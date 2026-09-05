# SPDX-License-Identifier: Apache-2.0
"""Однопоточный цикл генерации с MTP-спекуляцией для гибридных Qwen на MLX.

Схема always-advance, как в scheduler.py::_mtp_step:
  1. primary P = argmax(logits)
  2. draft  D = argmax(mtp_forward(hidden, P))     — MTP-голова без своего кэша
  3. verify: model([P, D]) одним вызовом
  4. accept (pred==D): выдаём P и D, следующий шаг берёт готовые logits/hidden
     reject: KV trim(2) + восстановление состояния рекуррентных слоёв из
     снимка + перепрогон только с P (гибридная модель: SSM-состояние не
     обрезается, его надо откатывать снимком)

"""

import time

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache


def raise_wired_limit():
    """Держать веса резидентными в Metal.

    mlx_lm.stream_generate делает это через контекст wired_limit, а свой цикл
    без него на длинном контексте валится: 82 -> 8 tok/s начиная с ~4k
    токенов, тогда как у stream_generate на тех же длинах ровно 64-83.
    """
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])


def _eos_ids(tokenizer):
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(ids)
    return {tokenizer.eos_token_id}


def _prefill(model, ids, cache, step, collect_hidden=False):
    """Префилл промпта. При collect_hidden возвращает hidden ВСЕХ позиций."""
    y = mx.array(ids)[None]
    chunks = []
    while y.shape[1] > step:
        logits, hidden = model(y[:, :step], cache=cache, return_hidden=True)
        if collect_hidden:
            mx.eval(hidden)
            chunks.append(hidden)
        else:
            mx.eval(logits)
        y = y[:, step:]
        mx.clear_cache()
    logits, hidden = model(y, cache=cache, return_hidden=True)
    if collect_hidden:
        chunks.append(hidden)
        return logits[:, -1, :], mx.concatenate(chunks, axis=1)
    return logits[:, -1, :], hidden[:, -1:, :]


def _mtp_prefill(model, hidden_all, ids, mtp_cache, step=2048):
    """Заполняет кэш MTP-головы позициями промпта.

    Позиция n головы — fc(norm(h_{n-1}), norm(emb(t_n))), поэтому берём
    hidden без последнего элемента и токены без первого.
    """
    h = hidden_all[:, :-1, :]
    t = mx.array(ids[1:])[None]
    for i in range(0, h.shape[1], step):
        mx.eval(
            model.mtp_forward(
                h[:, i : i + step, :],
                t[:, i : i + step],
                mtp_cache=mtp_cache,
                last_only=True,
            )
        )


def _mtp_step(model, hidden, tok_ids, mtp_cache):
    """Один шаг MTP-головы, возвращает и логиты, и выход слоя головы.

    Выход слоя (до финальной нормы) — замена настоящего hidden для
    цепочки драфтов: голова кормится собственным выходом, как в
    многошаговом MTP DeepSeek/vLLM.
    """
    from mlx_lm.models.base import create_attention_mask

    e = model.mtp.pre_fc_norm_embedding(model.model.embed_tokens(tok_ids))
    h = model.mtp.pre_fc_norm_hidden(hidden)
    x = model.mtp.fc(mx.concatenate([e, h], axis=-1))
    c = mtp_cache[0] if mtp_cache else None
    mask = create_attention_mask(x, c)
    x = model.mtp.layers[0](x, mask=mask, cache=c)
    normed = model.mtp.norm(x)
    if model.args.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(normed)
    else:
        logits = model.lm_head(normed)
    return logits, x


def _snapshot_recurrent(cache):
    """Снимок состояния нетримуемых (рекуррентных) слоёв.

    ArraysCache в mlx-lm заменяет элементы (`cache[0] = ...`), а не правит их
    на месте, поэтому снимок — это копия СПИСКА, сами массивы копировать не
    нужно. В scheduler.py тут стоит s.copy(), которого у mlx.core.array нет;
    их reject-ветка глохла бы в except Exception как MTP error.
    """
    snaps = {}
    for i, c in enumerate(cache):
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            continue
        state = getattr(c, "state", None)
        if state is None:
            continue
        snaps[i] = list(state)
    return snaps


def _ssm_recovery_supported(model, cache):
    """Only the injected, single-sequence Qwen cache contract is supported."""
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    if not getattr(model, "supports_ssm_recovery", False):
        return False
    layers = getattr(getattr(model, "model", None), "layers", ())
    if not layers or len(layers) != len(cache):
        return False
    found_ssm = False
    for layer, c in zip(layers, cache):
        if getattr(layer, "is_linear", False):
            found_ssm = True
            attn = getattr(layer, "linear_attn", None)
            if (
                type(c) is not ArraysCache
                or type(attn) is not GatedDeltaNet
                or c.lengths is not None
                or c.left_padding is not None
                or attn.sharding_group is not None
                or not callable(getattr(layer, "input_layernorm", None))
                or len(c.state) != 2
                or any(s is None or s.shape[0] != 1 for s in c.state)
            ):
                return False
        elif type(c) is not KVCache:
            return False
    return found_ssm


def _ssm_checkpoint_supported(model, cache, hidden, verification_tokens=2):
    """Select the exact supported kernel contract before verification mutates caches."""
    if not getattr(
        model, "supports_ssm_checkpoint", False
    ) or not _ssm_recovery_supported(model, cache):
        return False
    from .qwen3_5_checkpoint import validate_checkpoint_contract

    try:
        for layer, c in zip(model.model.layers, cache):
            if layer.is_linear:
                validate_checkpoint_contract(
                    layer.linear_attn,
                    (1, verification_tokens, hidden.shape[-1]),
                    hidden.dtype,
                    c,
                )
    except (TypeError, ValueError):
        return False
    return True


def _restore_checkpoint_prefix(cache, snapshots, checkpoints, prefix):
    """Restore a T3 prefix only after validating the complete capture and KV trim.

    Missing/unsupported captures return None without changing any cache, so the
    caller can replay from its pre-verification snapshot. Both prefix states
    must be valid even when only one is selected for this rejection.
    """
    if (
        prefix not in (1, 2)
        or not snapshots
        or not isinstance(checkpoints, dict)
        or checkpoints.keys() != snapshots.keys()
    ):
        return None
    for i, prefixes in checkpoints.items():
        if (
            type(cache[i]) is not ArraysCache
            or not isinstance(prefixes, dict)
            or set(prefixes) != {1, 2}
            or len(snapshots[i]) != 2
        ):
            return None
        for state in prefixes.values():
            if not isinstance(state, (list, tuple)) or len(state) != 2:
                return None
            for array, expected in zip(state, snapshots[i]):
                if (
                    array is None
                    or expected is None
                    or getattr(array, "shape", None) != expected.shape
                    or getattr(array, "dtype", None) != expected.dtype
                ):
                    return None
    trim = 3 - prefix
    kvs = [c for i, c in enumerate(cache) if i not in snapshots]
    if any(
        type(c) is not KVCache or type(c.offset) is not int or c.offset < trim
        for c in kvs
    ):
        return None
    recovered = tuple(
        array for prefixes in checkpoints.values() for array in prefixes[prefix]
    )
    for i, prefixes in checkpoints.items():
        cache[i].state = list(prefixes[prefix])
    for c in kvs:
        if c.trim(trim) != trim:
            raise RuntimeError(
                "KV cache did not discard exactly the rejected verification suffix"
            )
    return recovered


def _recover_ssm(model, cache, snapshots, first_inputs):
    """Rebuild only P's convolution/recurrent states, leaving verification KV.

    Unused linear-attention outputs remain lazy: its output projection and the
    decoder FFN are not dependencies of the two cache arrays. Build all states
    before committing them so an incomplete capture can use full replay.
    """
    if (
        not first_inputs
        or first_inputs.keys() != snapshots.keys()
        or any(x.ndim != 3 or x.shape[:2] != (1, 1) for x in first_inputs.values())
    ):
        return None
    recovered = {}
    for i, inputs in first_inputs.items():
        temporary = ArraysCache(2)
        temporary.state = list(snapshots[i])
        layer = model.model.layers[i]
        layer.linear_attn(layer.input_layernorm(inputs), cache=temporary)
        recovered[i] = temporary.state
    for i, state in recovered.items():
        cache[i].state = state
    return tuple(s for state in recovered.values() for s in state)


def decode(
    model,
    tokenizer,
    prompt,
    max_tokens,
    use_mtp,
    prefill_step=2048,
    mtp_history=True,
    mtp_norm_hidden=False,
    mtp_depth=1,
    *,
    ssm_recovery=False,
    ssm_checkpoint=False,
):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    return decode_ids(
        model,
        tokenizer,
        ids,
        max_tokens,
        use_mtp,
        prefill_step,
        mtp_history,
        mtp_norm_hidden,
        mtp_depth=mtp_depth,
        ssm_recovery=ssm_recovery,
        ssm_checkpoint=ssm_checkpoint,
    )


def _mtp_offsets_reusable(state):
    """Reject legacy or unreachable MTP boundaries before changing any cache."""
    mtp_cache = state.get("mtp_cache")
    if not mtp_cache:
        return True
    offsets = state.get("prompt_mtp_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != len(mtp_cache):
        return False
    return all(
        isinstance(c, KVCache)
        and type(offset) is int
        and type(c.offset) is int
        and 0 <= offset <= c.offset
        for c, offset in zip(mtp_cache, offsets)
    )


def decode_ids(
    model,
    tokenizer,
    ids,
    max_tokens,
    use_mtp,
    prefill_step=2048,
    mtp_history=True,
    mtp_norm_hidden=False,
    state=None,
    mtp_depth=1,
    *,
    ssm_recovery=False,
    ssm_checkpoint=False,
):
    """state: {"ids", "cache", "mtp_cache"} — переиспользуемый префикс.

    Переиспользуем только если сохранённые токены — РОВНО префикс новых:
    у гибридной модели рекуррентное состояние не обрезается назад.

    ssm_recovery is an opt-in depth-one experiment; unsupported caches retain
    the original full-model replay. Batched/single-token numerical equivalence
    must be verified on the loaded model before enabling it by default.
    ssm_checkpoint retains the accepted prefix states inside verification:
    P for depth one, P and P,D1 for depth two. It is opt-in and falls back to
    replay for unsupported cache contracts or incomplete prefix captures.
    """
    if ssm_recovery and ssm_checkpoint:
        raise ValueError("ssm_recovery and ssm_checkpoint are mutually exclusive")
    eos = _eos_ids(tokenizer)
    reuse = 0
    if state and state.get("ids"):
        prev = state["ids"]
        n = 0
        for x, y in zip(prev, ids):
            if x != y:
                break
            n += 1
        if n < len(prev):
            print(
                f"[prefix] промах: общий {n} из сохранённых {len(prev)}, "
                f"новый промпт {len(ids)}",
                flush=True,
            )
    if state and not _mtp_offsets_reusable(state):
        # A generated head tail cannot be distinguished from prompt history
        # without its recorded boundary. Leave the old caches untouched.
        state = None
    exact = False
    if (
        state
        and state.get("ids")
        and list(state["ids"]) == list(ids)
        and state.get("prompt_logits") is not None
        and bool(state.get("mtp_cache")) == (use_mtp and mtp_history)
    ):
        # Точное совпадение промпта (повтор той же задачи или первый запрос
        # новой сессии с тем же системным префиксом и текстом): откатываем кэш
        # к точке после префилла и берём сохранённые логиты — префилл не нужен.
        cache, mtp_cache = state["cache"], state["mtp_cache"]
        extra = state.get("kv_len", 0) - len(ids)
        if extra > 0:
            for c in cache:
                if (
                    hasattr(c, "is_trimmable")
                    and c.is_trimmable()
                    and hasattr(c, "trim")
                ):
                    c.trim(extra)
        # ArraysCache replaces entries in its state list during forward calls.
        # Keep the reusable prompt snapshot separate from that mutable list.
        for i, snap in (state.get("snap") or {}).items():
            cache[i].state = list(snap)
        reuse = len(ids)
        exact = True
    elif (
        state
        and state.get("ids")
        and len(state["ids"]) < len(ids)
        and ids[: len(state["ids"])] == state["ids"]
        and bool(state.get("mtp_cache")) == (use_mtp and mtp_history)
    ):
        cache, mtp_cache = state["cache"], state["mtp_cache"]
        reuse = len(state["ids"])
        # Кэш содержит и сгенерированный хвост прошлого шага, которого в новом
        # промпте нет (шаблон переупаковывает ответ в теги роли). Откатываем
        # кэш к состоянию сразу после префилла: KV тримится, рекуррентные
        # слои назад не обрезаются — их восстанавливаем из снимка.
        extra = state.get("kv_len", 0) - reuse
        if extra > 0:
            for c in cache:
                if (
                    hasattr(c, "is_trimmable")
                    and c.is_trimmable()
                    and hasattr(c, "trim")
                ):
                    c.trim(extra)
        # ArraysCache replaces entries in its state list during forward calls.
        # Keep the reusable prompt snapshot separate from that mutable list.
        for i, snap in (state.get("snap") or {}).items():
            cache[i].state = list(snap)
    else:
        cache = make_prompt_cache(model)
        mtp_cache = model.make_mtp_cache() if (use_mtp and mtp_history) else None

    if reuse and mtp_cache is not None:
        for c, offset in zip(mtp_cache, state["prompt_mtp_offsets"]):
            c.trim(c.offset - offset)

    def _h(x):
        # Голова обучалась на выходе последнего блока; ключ mtp_norm_hidden
        # проверяет альтернативу — hidden после финальной нормы модели.
        return model.model.norm(x) if mtp_norm_hidden else x

    t0 = time.perf_counter()
    tail = ids[reuse:]
    if exact:
        logits, hidden = state["prompt_logits"], state["prompt_hidden"]
        prompt_snap = state["snap"]
    else:
        logits, hidden = _prefill(
            model, tail, cache, prefill_step, collect_hidden=(mtp_cache is not None)
        )
        if mtp_cache is not None:
            # позиция n головы — (h_{n-1}, t_n); при переиспользовании префикса
            # первая позиция хвоста опирается на hidden, которого у нас нет,
            # поэтому её пропускаем: голова догонит со следующей.
            _mtp_prefill(model, _h(hidden), tail, mtp_cache, prefill_step)
            hidden = hidden[:, -1:, :]
        mx.eval(logits, hidden)
        # Точка отката для следующего запроса: кэш здесь соответствует ровно
        # промпту, без единого сгенерированного токена.
        prompt_snap = _snapshot_recurrent(cache)
    # Extension prefill skips a head position, so token count is not its offset.
    prompt_mtp_offsets = [c.offset for c in mtp_cache] if mtp_cache else None
    prefill_s = time.perf_counter() - t0
    prompt_logits, prompt_hidden = logits, hidden
    prompt_kv_len = len(ids)

    out = []
    attempted = accepted = 0
    recovery_enabled = bool(
        ssm_recovery
        and use_mtp
        and mtp_depth == 1
        and _ssm_recovery_supported(model, cache)
    )
    verification_tokens = 3 if mtp_depth >= 2 and mtp_cache is not None else 2
    checkpoint_enabled = bool(
        ssm_checkpoint
        and use_mtp
        and _ssm_checkpoint_supported(
            model, cache, hidden, verification_tokens=verification_tokens
        )
    )
    ssm_recovered = ssm_checkpointed = 0
    recovery_states = ()
    carry_h = carry_t = None  # позиции головы, пропущенные принятым драфтом
    t1 = time.perf_counter()
    while len(out) < max_tokens:
        primary = mx.argmax(logits, axis=-1)

        if not use_mtp:
            mx.eval(primary)
            tok = primary.item()
            out.append(tok)
            if tok in eos:
                break
            logits, hidden = model(primary[:, None], cache=cache, return_hidden=True)
            logits, hidden = logits[:, -1, :], hidden[:, -1:, :]
            continue

        h_in, t_in = hidden, primary[:, None]
        if carry_h is not None:
            h_in = mx.concatenate([carry_h, hidden], axis=1)
            t_in = mx.concatenate([carry_t, primary[:, None]], axis=1)

        if mtp_depth >= 2 and mtp_cache is not None:
            # Цепочка из двух драфтов: D2 рисуется из выхода слоя головы.
            d1_logits, x1 = _mtp_step(model, _h(h_in), t_in, mtp_cache)
            d1 = mx.argmax(d1_logits[:, -1, :], axis=-1)
            d2_logits, _ = _mtp_step(model, x1[:, -1:, :], d1[:, None], mtp_cache)
            d2 = mx.argmax(d2_logits[:, -1, :], axis=-1)
            # Start the head while Python constructs the verification graph.
            mx.async_eval(d2)
            # позиция D1-драфта в кэше головы спекулятивна — убираем; правду
            # донесёт carry следующего раунда
            mtp_cache[0].trim(1)

            snaps = _snapshot_recurrent(cache)
            verify_in = mx.concatenate(
                [primary[:, None], d1[:, None], d2[:, None]], axis=1
            )
            checkpoints = {} if checkpoint_enabled else None
            if checkpoint_enabled:
                vlogits, vhidden = model(
                    verify_in,
                    cache=cache,
                    return_hidden=True,
                    ssm_checkpoints=checkpoints,
                )
            else:
                vlogits, vhidden = model(verify_in, cache=cache, return_hidden=True)
            p0 = mx.argmax(vlogits[:, 0, :], axis=-1)
            p1 = mx.argmax(vlogits[:, 1, :], axis=-1)
            mx.eval(p0, p1, d1, d2, primary)

            tok = primary.item()
            out.append(tok)
            if tok in eos:
                break
            attempted += 1

            def _rollback(n_replay_tokens):
                for c in cache:
                    if (
                        hasattr(c, "is_trimmable")
                        and c.is_trimmable()
                        and hasattr(c, "trim")
                    ):
                        c.trim(3)
                for i, s in snaps.items():
                    cache[i].state = s
                return model(n_replay_tokens, cache=cache, return_hidden=True)

            if p0.item() == d1.item():
                accepted += 1
                d1_tok = d1.item()
                out.append(d1_tok)
                if d1_tok in eos:
                    break
                # D2 is attempted only when D1 did not terminate generation.
                attempted += 1
                if p1.item() == d2.item():
                    accepted += 1
                    d2_tok = d2.item()
                    out.append(d2_tok)
                    if d2_tok in eos:
                        break
                    logits, hidden = vlogits[:, 2, :], vhidden[:, 2:3, :]
                    carry_h = vhidden[:, 0:2, :]
                    carry_t = mx.concatenate([d1[:, None], d2[:, None]], axis=1)
                else:
                    # D1 принят, D2 нет: позиция D2 в кэше от неверного токена
                    recovered = _restore_checkpoint_prefix(cache, snaps, checkpoints, 2)
                    if recovered is not None:
                        ssm_checkpointed += 1
                        recovery_states = recovered
                        mx.async_eval(*recovery_states)
                        logits, hidden = vlogits[:, 1, :], vhidden[:, 1:2, :]
                        carry_h, carry_t = vhidden[:, 0:1, :], d1[:, None]
                    else:
                        rl, rh = _rollback(
                            mx.concatenate([primary[:, None], d1[:, None]], axis=1)
                        )
                        logits, hidden = rl[:, -1, :], rh[:, -1:, :]
                        carry_h, carry_t = rh[:, 0:1, :], d1[:, None]
            else:
                recovered = _restore_checkpoint_prefix(cache, snaps, checkpoints, 1)
                if recovered is not None:
                    ssm_checkpointed += 1
                    recovery_states = recovered
                    mx.async_eval(*recovery_states)
                    logits, hidden = vlogits[:, 0, :], vhidden[:, 0:1, :]
                else:
                    rl, rh = _rollback(primary[:, None])
                    logits, hidden = rl[:, -1, :], rh[:, -1:, :]
                carry_h = carry_t = None
            checkpoints = None
            continue

        draft_logits = model.mtp_forward(
            _h(h_in), t_in, mtp_cache=mtp_cache, last_only=True
        )
        draft = mx.argmax(draft_logits[:, -1, :], axis=-1)
        # Конвейер: голова стартует на GPU, пока Python снимает снимок и
        # строит граф verify — иначе GPU простаивает на постройке графа.
        mx.async_eval(draft)

        snaps = _snapshot_recurrent(cache)
        verify_in = mx.concatenate([primary[:, None], draft[:, None]], axis=1)
        first_inputs = {} if recovery_enabled else None
        checkpoints = {} if checkpoint_enabled else None
        if checkpoint_enabled:
            vlogits, vhidden = model(
                verify_in, cache=cache, return_hidden=True, ssm_checkpoints=checkpoints
            )
        elif recovery_enabled:
            vlogits, vhidden = model(
                verify_in, cache=cache, return_hidden=True, ssm_inputs=first_inputs
            )
        else:
            vlogits, vhidden = model(verify_in, cache=cache, return_hidden=True)
        pred = mx.argmax(vlogits[:, 0, :], axis=-1)
        mx.eval(pred, draft, primary)

        tok = primary.item()
        out.append(tok)
        if tok in eos:
            break
        attempted += 1

        if pred.item() == draft.item():
            accepted += 1
            dtok = draft.item()
            out.append(dtok)
            if dtok in eos:
                break
            logits, hidden = vlogits[:, 1, :], vhidden[:, 1:2, :]
            # позиция драфта в кэше головы ещё не посчитана — донесём её
            carry_h, carry_t = vhidden[:, 0:1, :], draft[:, None]
        else:
            recovered = None
            if checkpoints and checkpoints.keys() == snaps.keys():
                for i, checkpoint in checkpoints.items():
                    cache[i].state = checkpoint
                recovered = tuple(s for pair in checkpoints.values() for s in pair)
                ssm_checkpointed += 1
            elif recovery_enabled:
                recovered = _recover_ssm(model, cache, snaps, first_inputs)
                if recovered is not None:
                    ssm_recovered += 1
            if recovered is not None:
                for c in cache:
                    if isinstance(c, KVCache):
                        c.trim(1)
                logits, hidden = vlogits[:, 0, :], vhidden[:, 0:1, :]
                recovery_states = recovered
                mx.async_eval(*recovery_states)
            else:
                for c in cache:
                    if (
                        hasattr(c, "is_trimmable")
                        and c.is_trimmable()
                        and hasattr(c, "trim")
                    ):
                        c.trim(2)
                for i, s in snaps.items():
                    cache[i].state = s
                rl, rh = model(primary[:, None], cache=cache, return_hidden=True)
                logits, hidden = rl[:, -1, :], rh[:, -1:, :]
            carry_h = carry_t = None
        # Captures are bounded to this verification round.
        first_inputs = checkpoints = None

    mx.eval(logits, *recovery_states)
    gen_s = time.perf_counter() - t1
    return {
        "cached_tokens": reuse,
        "state": {
            "ids": list(ids),
            "cache": cache,
            "mtp_cache": mtp_cache,
            "snap": prompt_snap,
            "kv_len": prompt_kv_len + len(out),
            "prompt_mtp_offsets": prompt_mtp_offsets,
            "prompt_logits": prompt_logits,
            "prompt_hidden": prompt_hidden,
        },
        "tokens": out,
        "text": tokenizer.decode(out),
        "n_prompt": len(ids),
        "n_prefilled": len(tail),
        "n_gen": len(out),
        "prefill_s": prefill_s,
        "prefill_tok_s": len(tail) / prefill_s,
        "gen_s": gen_s,
        "gen_tok_s": len(out) / gen_s,
        "attempted": attempted,
        "accepted": accepted,
        "ssm_recovery_enabled": recovery_enabled,
        "ssm_recovered": ssm_recovered,
        "ssm_checkpoint_enabled": checkpoint_enabled,
        "ssm_checkpointed": ssm_checkpointed,
        "accept_rate": accepted / attempted if attempted else 0.0,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }
