# SPDX-License-Identifier: Apache-2.0
"""OpenAI-совместимый сервер для MLX-моделей с MTP-спекуляцией и pinned-префиксом.

  * /v1/models, /v1/chat/completions (JSON и SSE), /metrics
  * tools через chat template; оба диалекта tool calls Qwen (Hermes-JSON и XML)
  * MTP-драфтер: x1.5 к генерации при высоком acceptance
  * префикс-кэш: точное совпадение, расширение и pinned-префикс с диском —
    граница общего префикса сессий определяется автоматически по LCP

Генерация последовательная в одном MLX-потоке: модель одна, параллелить нечего.

    python -m mtpserve --model <путь к MLX-модели> --port 19234
"""

import argparse
from contextlib import nullcontext
from functools import wraps
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mlx.core as mx

from .engine import (
    _mtp_prefill,
    _prefill,
    decode_ids,
    make_prompt_cache,
    raise_wired_limit,
)
from .loader import load_model

MODEL = None
TOKENIZER = None
MODEL_ID = ""
LOCK = threading.Lock()
STATE = {"v": None}  # переиспользуемый префикс между запросами
USE_MTP = True
USE_CHECKPOINT_MTP = False
MTP_DEPTH = 1

# Все MLX-вычисления — в одном выделенном потоке. ThreadingHTTPServer даёт
# каждому запросу свежий поток; массивы из STATE (логиты, кэш) привязаны к
# стриму создавшего их потока, и после его смерти mx.eval падает с
# "There is no Stream(gpu, N) in current thread".
import queue  # noqa: E402

_JOBS = queue.Queue()


def _mlx_worker():
    while True:
        fn, args, out = _JOBS.get()
        try:
            out["res"] = fn(*args)
        except Exception as e:  # noqa: BLE001 — пробрасываем в поток запроса
            out["err"] = e
        out["done"].set()


threading.Thread(target=_mlx_worker, daemon=True).start()


def run_in_worker(fn, *args):
    out = {"done": threading.Event()}
    _JOBS.put((fn, args, out))
    out["done"].wait()
    if "err" in out:
        raise out["err"]
    return out["res"]


class IdleUnloadingHTTPServer(ThreadingHTTPServer):
    """Stop the process after a request-free interval, releasing model memory.

    ``serve_forever`` runs ``service_actions`` in its own thread.  Calling
    ``shutdown`` directly from there would deadlock, so expiry hands it to a
    separate short-lived thread.  Request accounting lives around endpoint
    handling, not a socket lifetime: an idle HTTP keep-alive connection must
    not hold a loaded model indefinitely.
    """

    def __init__(self, server_address, request_handler_class, *, idle_timeout=0):
        super().__init__(server_address, request_handler_class)
        self.idle_timeout = idle_timeout
        self._idle_lock = threading.Lock()
        self._active_requests = 0
        self._idle_since = time.monotonic()
        self._idle_shutdown_started = False

    def request_started(self):
        with self._idle_lock:
            self._active_requests += 1
            self._idle_since = None

    def request_finished(self):
        with self._idle_lock:
            if self._active_requests <= 0:
                raise RuntimeError("HTTP request activity underflow")
            self._active_requests -= 1
            if self._active_requests == 0:
                self._idle_since = time.monotonic()

    def service_actions(self):
        super().service_actions()
        with self._idle_lock:
            expired = (
                self.idle_timeout > 0
                and self._active_requests == 0
                and self._idle_since is not None
                and time.monotonic() - self._idle_since >= self.idle_timeout
                and not self._idle_shutdown_started
            )
            if expired:
                self._idle_shutdown_started = True
        if expired:
            print(
                f"idle timeout ({self.idle_timeout:g}s): unloading model and stopping server",
                flush=True,
            )
            threading.Thread(
                target=self.shutdown, name="mtpserve-idle-shutdown", daemon=True
            ).start()


def _track_request_activity(endpoint):
    """Keep an idle timer accurate for completed HTTP requests only."""

    @wraps(endpoint)
    def tracked(self, *args, **kwargs):
        tracker = getattr(self.server, "request_started", None)
        if tracker is None:
            return endpoint(self, *args, **kwargs)
        tracker()
        try:
            return endpoint(self, *args, **kwargs)
        finally:
            self.server.request_finished()

    return tracked


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# Qwen3.5/3.8: XML-диалект того же тега — <function=имя><parameter=ключ>...
TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM_XML_RE = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
# decode() не срезает спецтокены — они утекают в content
SPECIAL_RE = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>")


def build_ids(messages, tools):
    """Промпт через chat template. tools отдаём шаблону — он сам вставит
    описания в системную часть в том формате, которого ждёт модель."""
    # OpenAI-клиенты (pi) шлют tool_calls.function.arguments СТРОКОЙ, а шаблон
    # Qwen3.5 делает arguments|items и падает на не-mapping. Разворачиваем.
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    pass
    kw = {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
    if tools:
        kw["tools"] = tools
    return TOKENIZER.apply_chat_template(messages, **kw)


def _coerce_arg(name, value, schema_props):
    """XML-параметры приходят строками; числа/булевы/объекты приводим по схеме."""
    spec = (schema_props or {}).get(name, {})
    t = spec.get("type")
    if t in ("number", "integer", "boolean", "object", "array"):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _make_call(name, args):
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            # OpenAI ждёт arguments строкой, а не объектом
            "arguments": args
            if isinstance(args, str)
            else json.dumps(args, ensure_ascii=False),
        },
    }


def split_tool_calls(text, tools=None):
    """Оба диалекта Qwen -> OpenAI tool_calls. Возвращает (reasoning, текст, вызовы).

    Hermes (Qwen3-Next): <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    XML (Qwen3.5/3.8):   <tool_call><function=f><parameter=k>v</parameter>...
    Шаблон 3.5/3.8 открывает <think> в generation prompt — всё до </think>
    в выводе это рассуждение, не контент.
    """
    reasoning = ""
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()

    schemas = {}
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        schemas[fn.get("name", "")] = (fn.get("parameters") or {}).get("properties", {})

    calls = []
    for m in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        calls.append(_make_call(payload.get("name", ""), payload.get("arguments", {})))
    for m in TOOL_CALL_XML_RE.finditer(text):
        name = m.group(1)
        args = {
            k: _coerce_arg(k, v, schemas.get(name))
            for k, v in PARAM_XML_RE.findall(m.group(2))
        }
        calls.append(_make_call(name, args))

    clean = TOOL_CALL_XML_RE.sub("", TOOL_CALL_RE.sub("", text))
    clean = SPECIAL_RE.sub("", clean).strip()
    return (reasoning, clean, calls)


# --- pinned-префикс -----------------------------------------------------------
# Аналог prefix_pin движка Junie, но граница определяется сама: LCP промптов
# двух стартов сессий = стабильный системный префикс (system + схемы tools).
# На границе делается выделенный префилл и снимок состояний всех слоёв; каждая
# новая сессия стартует из КОПИИ снимка и префиллит только свой хвост.
PIN = {"ids": None, "states": None, "mtp_states": None}
LAST_INITIAL = {"ids": None}
PIN_MIN_TOKENS = 64
PIN_DIR = Path(os.environ.get("MTPSERVE_CACHE", Path.home() / ".cache" / "mtpserve"))

STATS = {
    "requests": 0,
    "gen_tokens": 0,
    "attempted": 0,
    "accepted": 0,
    "cached_tokens": 0,
    "computed_tokens": 0,
    "starts": {"exact": 0, "extension": 0, "pin": 0, "miss": 0},
    "started_at": time.time(),
}


def _pin_paths():
    key = hashlib.sha256(MODEL_ID.encode()).hexdigest()[:16]
    return PIN_DIR / f"pin-{key}.safetensors", PIN_DIR / f"pin-{key}.json"


def _flatten_states(states, prefix):
    """Состояния слоёв -> (тензоры для safetensors, json-описание структуры)."""
    tensors, meta = {}, []
    for i, st in enumerate(states):
        if isinstance(st, (list, tuple)):
            slots = []
            for j, x in enumerate(st):
                if isinstance(x, mx.array):
                    tensors[f"{prefix}{i}.{j}"] = x
                    slots.append("arr")
                else:
                    slots.append({"raw": x})
            meta.append(
                {"kind": "tuple" if isinstance(st, tuple) else "list", "slots": slots}
            )
        elif isinstance(st, mx.array):
            tensors[f"{prefix}{i}"] = st
            meta.append({"kind": "arr"})
        else:
            meta.append({"kind": "raw", "value": st})
    return tensors, meta


def _unflatten_states(tensors, meta, prefix):
    states = []
    for i, m in enumerate(meta):
        if m["kind"] in ("tuple", "list"):
            st = [
                tensors[f"{prefix}{i}.{j}"] if s == "arr" else s["raw"]
                for j, s in enumerate(m["slots"])
            ]
            states.append(tuple(st) if m["kind"] == "tuple" else st)
        elif m["kind"] == "arr":
            states.append(tensors[f"{prefix}{i}"])
        else:
            states.append(m["value"])
    return states


def _save_pin():
    """Пин на диск: переживает рестарт сервера."""
    try:
        PIN_DIR.mkdir(parents=True, exist_ok=True)
        st_path, meta_path = _pin_paths()
        tensors, meta = _flatten_states(PIN["states"], "L")
        mtp_meta = None
        if PIN["mtp_states"] is not None:
            mtp_tensors, mtp_meta = _flatten_states(PIN["mtp_states"], "M")
            tensors.update(mtp_tensors)
        mx.save_safetensors(str(st_path), tensors)
        meta_path.write_text(
            json.dumps(
                {
                    "model": MODEL_ID,
                    "ids": PIN["ids"],
                    "layers": meta,
                    "mtp_layers": mtp_meta,
                }
            )
        )
        print(
            f"pin: сохранён на диск ({st_path.stat().st_size / 1e6:.0f} MB)", flush=True
        )
    except Exception as e:  # noqa: BLE001 — пин это ускорение, не обязанность
        print(f"pin: не сохранился на диск: {e}", flush=True)


def _load_pin():
    st_path, meta_path = _pin_paths()
    if not (st_path.exists() and meta_path.exists()):
        return
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != MODEL_ID:
            return
        if bool(meta.get("mtp_layers")) != USE_MTP:
            return

        # Материализуем в MLX-воркере: ленивые массивы mx.load живут в
        # стриме создавшего потока, а пользоваться ими будет воркер.
        def _job():
            t = mx.load(str(st_path))
            mx.eval(*t.values())
            return t

        tensors = run_in_worker(_job)
        PIN["ids"] = list(meta["ids"])
        PIN["states"] = _unflatten_states(tensors, meta["layers"], "L")
        PIN["mtp_states"] = (
            _unflatten_states(tensors, meta["mtp_layers"], "M")
            if meta.get("mtp_layers")
            else None
        )
        print(f"pin: загружен с диска ({len(PIN['ids'])} токенов)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"pin: не загрузился с диска: {e}", flush=True)
        PIN["ids"] = PIN["states"] = PIN["mtp_states"] = None


def _copy_arr(x):
    # Гарантированно новый буфер: KV-кэш пишет в свои массивы на месте,
    # разделять их с пином нельзя.
    return mx.add(x, 0) if isinstance(x, mx.array) else x


def _copy_state(st):
    if isinstance(st, (list, tuple)):
        return type(st)(_copy_arr(x) for x in st)
    return _copy_arr(st)


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _pin_prefix(pin_ids):
    """Префилл границы в собственный кэш и снимок. Кэш пина больше не трогаем."""
    cache = make_prompt_cache(MODEL)
    mtp_cache = MODEL.make_mtp_cache() if USE_MTP else None
    t = time.time()
    _, hidden = _prefill(
        MODEL, pin_ids, cache, 2048, collect_hidden=(mtp_cache is not None)
    )
    if mtp_cache is not None:
        _mtp_prefill(MODEL, hidden, pin_ids, mtp_cache, 2048)
    PIN["ids"] = list(pin_ids)
    PIN["states"] = [c.state for c in cache]
    PIN["mtp_states"] = [c.state for c in mtp_cache] if mtp_cache else None
    print(
        f"pin: закреплён префикс {len(pin_ids)} токенов за {time.time() - t:.1f}s",
        flush=True,
    )
    _save_pin()


def _restore_pin():
    """Сессионный кэш из копии пина — в формате state для decode_ids."""
    cache = make_prompt_cache(MODEL)
    for c, st in zip(cache, PIN["states"]):
        c.state = _copy_state(st)
    mtp_cache = None
    if PIN["mtp_states"] is not None:
        mtp_cache = MODEL.make_mtp_cache()
        for c, st in zip(mtp_cache, PIN["mtp_states"]):
            c.state = _copy_state(st)
    return {
        "ids": list(PIN["ids"]),
        "cache": cache,
        "mtp_cache": mtp_cache,
        "prompt_mtp_offsets": [c.offset for c in mtp_cache] if mtp_cache else None,
        "snap": {},
        "kv_len": len(PIN["ids"]),
        "prompt_logits": None,
        "prompt_hidden": None,
    }


def _decode_job(ids, max_tokens):
    st = STATE["v"]
    ids_l = list(ids)
    extends_last = bool(
        st
        and st.get("ids")
        and (
            st["ids"] == ids_l
            or (len(st["ids"]) < len(ids_l) and ids_l[: len(st["ids"])] == st["ids"])
        )
    )
    start = (
        "exact"
        if (extends_last and st["ids"] == ids_l)
        else ("extension" if extends_last else "miss")
    )
    if not extends_last:
        # старт новой сессии: прошлое состояние бесполезно (назад не откатить)
        STATE["v"] = None
        prev = LAST_INITIAL["ids"]
        if prev is not None:
            # Кап: пин всегда строго короче промпта (иначе идентичные задачи
            # дают LCP = весь промпт, и extension-путь его не подхватит).
            # Гистерезис +64: не перезакреплять ради пары токенов — сам
            # префилл пина стоит ~9 секунд.
            n = min(_lcp(prev, ids_l), len(ids_l) - 8, len(prev) - 8)
            if n >= PIN_MIN_TOKENS and (PIN["ids"] is None or n > len(PIN["ids"]) + 64):
                _pin_prefix(ids_l[:n])
        LAST_INITIAL["ids"] = ids_l
        if (
            PIN["ids"]
            and len(PIN["ids"]) < len(ids_l)
            and ids_l[: len(PIN["ids"])] == PIN["ids"]
        ):
            STATE["v"] = _restore_pin()
            start = "pin"
    STATS["requests"] += 1
    STATS["starts"][start] += 1
    try:
        res = decode_ids(
            MODEL,
            TOKENIZER,
            ids,
            max_tokens,
            use_mtp=USE_MTP,
            state=STATE["v"],
            ssm_checkpoint=USE_CHECKPOINT_MTP,
            mtp_depth=MTP_DEPTH,
        )
        if USE_CHECKPOINT_MTP and (
            not res["ssm_checkpoint_enabled"]
            or res["ssm_checkpointed"] != res["attempted"] - res["accepted"]
        ):
            raise RuntimeError("Checkpoint MTP did not recover every rejected draft")
    except Exception:
        # decode_ids may already have mutated the previous cache by reference.
        if USE_CHECKPOINT_MTP:
            STATE["v"] = None
        raise
    STATE["v"] = res["state"]
    STATS["gen_tokens"] += res["n_gen"]
    STATS["attempted"] += res["attempted"]
    STATS["accepted"] += res["accepted"]
    STATS["cached_tokens"] += res["cached_tokens"]
    STATS["computed_tokens"] += res["n_prefilled"]
    return res


def generate(messages, tools, max_tokens):
    """Одна генерация (в MLX-воркере). Возвращает результат decode_ids + разбор."""
    ids = build_ids(messages, tools)
    with LOCK:
        res = run_in_worker(_decode_job, ids, max_tokens)
    reasoning, text, calls = split_tool_calls(res["text"], tools)
    res["reasoning"] = reasoning
    res["parsed_text"] = text
    res["tool_calls"] = calls
    return res


def message_from(res):
    msg = {"role": "assistant", "content": res["parsed_text"] or None}
    if res.get("reasoning"):
        msg["reasoning_content"] = res["reasoning"]
    if res["tool_calls"]:
        msg["tool_calls"] = res["tool_calls"]
    return msg


def usage_from(res):
    return {
        "prompt_tokens": res["n_prompt"],
        "completion_tokens": res["n_gen"],
        "total_tokens": res["n_prompt"] + res["n_gen"],
        "prompt_tokens_details": {"cached_tokens": res["cached_tokens"]},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        if USE_CHECKPOINT_MTP:
            # Non-daemon handlers must not wait forever on idle HTTP/1.1 sockets.
            # This limits socket I/O waits, not generation in the MLX worker.
            self.connection.settimeout(30)

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        blob = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    @_track_request_activity
    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                },
            )
        elif self.path.startswith("/health"):
            self._send(200, {"status": "ok"})
        elif self.path.startswith("/metrics"):
            att = STATS["attempted"]
            self._send(
                200,
                {
                    **{k: v for k, v in STATS.items() if k != "started_at"},
                    "accept_rate": STATS["accepted"] / att if att else None,
                    "uptime_s": round(time.time() - STATS["started_at"], 1),
                    "model": MODEL_ID,
                    "pin_tokens": len(PIN["ids"]) if PIN["ids"] else 0,
                },
            )
        else:
            self._send(404, {"error": {"message": "not found"}})

    @_track_request_activity
    def do_POST(self):
        if "chat/completions" not in self.path:
            return self._send(404, {"error": {"message": "not found"}})
        try:
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
        except Exception:
            return self._send(400, {"error": {"message": "invalid JSON"}})

        messages = body.get("messages", [])
        tools = body.get("tools")
        max_tokens = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or 1024
        )
        stream = bool(body.get("stream"))

        t0 = time.time()
        try:
            res = generate(messages, tools, max_tokens)
        except Exception as e:
            import traceback

            traceback.print_exc()
            return self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})

        print(
            f"req prompt={res['n_prompt']} cached={res['cached_tokens']} "
            f"gen={res['n_gen']} {time.time() - t0:.1f}s "
            f"gen={res['gen_tok_s']:.1f} tok/s accept={100 * res['accept_rate']:.0f}% "
            f"tools={len(res['tool_calls'])}",
            flush=True,
        )

        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(t0)
        finish = "tool_calls" if res["tool_calls"] else "stop"

        if not stream:
            return self._send(
                200,
                {
                    "id": cid,
                    "object": "chat.completion",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "message": message_from(res),
                            "finish_reason": finish,
                        }
                    ],
                    "usage": usage_from(res),
                },
            )

        # SSE: цикл не потоковый, поэтому отдаём готовый ответ чанками.
        # Поток закрываем разрывом соединения — Content-Length тут не задать.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta, finish_reason=None, usage=None):
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
            if usage is not None:
                payload["usage"] = usage
            self.wfile.write(
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
            )

        chunk({"role": "assistant"})
        if res.get("reasoning"):
            chunk({"reasoning_content": res["reasoning"]})
        if res["parsed_text"]:
            chunk({"content": res["parsed_text"]})
        for i, call in enumerate(res["tool_calls"]):
            # index обязателен: pi собирает вызовы по нему
            chunk({"tool_calls": [{"index": i, **call}]})
        chunk({}, finish, usage_from(res))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _idle_timeout(value):
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or not 0 <= seconds <= 86400:
        raise argparse.ArgumentTypeError("must be between 0 and 86400 seconds")
    return seconds


def main(argv=None):
    global MODEL, TOKENIZER, MODEL_ID, USE_MTP, USE_CHECKPOINT_MTP, MTP_DEPTH
    ap = argparse.ArgumentParser(prog="mtpserve")
    ap.add_argument(
        "--model",
        required=True,
        help="путь к MLX-модели (с mtp/weights.safetensors для MTP)",
    )
    ap.add_argument("--port", type=int, default=19234)
    ap.add_argument(
        "--idle-timeout",
        type=_idle_timeout,
        default=0,
        metavar="SECONDS",
        help="stop the server and release model memory after SECONDS without requests (0 disables; default: 0)",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--no-mtp", action="store_true")
    mode.add_argument(
        "--checkpoint-mtp",
        action="store_true",
        help="experimental Q4 verification and reject-state checkpoints",
    )
    ap.add_argument(
        "--mtp-depth",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of draft tokens per verification (default: 1)",
    )
    a = ap.parse_args(argv)
    if a.no_mtp and a.mtp_depth != 1:
        ap.error("--mtp-depth requires MTP")

    USE_MTP = not a.no_mtp
    USE_CHECKPOINT_MTP = a.checkpoint_mtp
    MTP_DEPTH = a.mtp_depth
    MODEL_ID = a.model

    t = time.time()
    MODEL, TOKENIZER = load_model(a.model, tokenizer_config={"eos_token": "<|im_end|>"})
    raise_wired_limit()
    has_mtp = getattr(MODEL, "mtp", None) is not None
    print(
        f"модель загружена за {time.time() - t:.1f}s, mtp={has_mtp}, use_mtp={USE_MTP}",
        flush=True,
    )
    if USE_MTP and not has_mtp:
        print("ВНИМАНИЕ: MTP-голова не найдена, идём без спекуляции", flush=True)

    context = nullcontext()
    if USE_CHECKPOINT_MTP:
        if not has_mtp or not getattr(MODEL, "supports_ssm_checkpoint", False):
            raise ValueError(
                "--checkpoint-mtp requires an MTP head and SSM checkpoint support"
            )
        from .q4_pair import paired_quantized_linears

        context = paired_quantized_linears(
            MODEL, verification_only=True, verification_rows=MTP_DEPTH + 1
        )

    with context as report:
        if USE_CHECKPOINT_MTP and report["patched_projection_count"] <= 0:
            raise ValueError("--checkpoint-mtp requires paired Q4 projections")
        _load_pin()

        srv = IdleUnloadingHTTPServer(
            ("127.0.0.1", a.port), Handler, idle_timeout=a.idle_timeout
        )
        if USE_CHECKPOINT_MTP:
            # Keep patched classes until every active request has completed.
            srv.daemon_threads = False
        try:
            print(f"READY http://127.0.0.1:{a.port}", flush=True)
            srv.serve_forever()
        finally:
            srv.server_close()


if __name__ == "__main__":
    main()
