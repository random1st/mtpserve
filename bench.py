# SPDX-License-Identifier: Apache-2.0
"""Фиксированный стенд: медиана gen tok/s (с MTP) и prefill tok/s.

3 серии x 4 коротких код-промпта, max_tokens=200, прогрев одним прогоном,
порядок промптов чередуется между сериями. Состояние кэша не переиспользуется
(state=None) — префилл честный в каждом прогоне.

Запуск: cd ~/src/mtpserve && uv run python bench.py
"""

import argparse
from contextlib import nullcontext
import json
import statistics
import sys
import time

MODEL = "/Users/random1st/.lmstudio/models/orcarouter/Qwen3.8-27B-Uncensored-MLX"
MAX_TOKENS = 200

PROMPTS = [
    "Write a Python function that parses a semver string into a tuple.",
    "Implement binary search over a sorted list in Python with tests.",
    "Write a Rust function that reverses words in a string in place.",
    "Implement an LRU cache class in Python using OrderedDict.",
]

ORDERS = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-mtp",
        action="store_true",
        help="experimental M1-arithmetic Q4 verification and reject-state checkpoints",
    )
    parser.add_argument(
        "--mtp-depth",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of draft tokens per verification (default: 1)",
    )
    args = parser.parse_args(argv)

    from mtpserve.engine import raise_wired_limit
    from mtpserve.loader import load_model

    raise_wired_limit()
    t0 = time.perf_counter()
    model, tokenizer = load_model(MODEL)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    if args.checkpoint_mtp:
        print("experimental checkpoint MTP enabled", flush=True)
    return _run_benchmark(
        model, tokenizer, checkpoint=args.checkpoint_mtp, mtp_depth=args.mtp_depth
    )


def _validate_pair_warmup(report, mtp_depth=1):
    key = "pair_calls_by_projection" if mtp_depth == 1 else "triple_calls_by_projection"
    calls = report.get(key, {})
    count = report.get("patched_projection_count", 0)
    if (
        count <= 0
        or report.get("supported_projection_count") != count
        or len(calls) != count
        or any(value <= 0 for value in calls.values())
        or not report.get("classes_restored")
        or not report.get("model_class_restored")
        or not report.get("parameter_objects_unchanged")
    ):
        raise RuntimeError(
            "Checkpoint warmup did not exercise every paired Q4 projection safely"
        )


def _run_benchmark(model, tokenizer, *, checkpoint=False, mtp_depth=1):
    import mlx.core as mx

    from mtpserve.engine import decode_ids

    def ids_for(p):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
        )

    def run(prompt):
        result = decode_ids(
            model,
            tokenizer,
            ids_for(prompt),
            MAX_TOKENS,
            use_mtp=True,
            ssm_checkpoint=checkpoint,
            mtp_depth=mtp_depth,
        )
        if checkpoint and (
            not result["ssm_checkpoint_enabled"]
            or result["ssm_checkpointed"] != result["attempted"] - result["accepted"]
        ):
            raise RuntimeError(
                "Checkpoint benchmark did not recover every rejected draft"
            )
        return result

    def context(count_calls):
        if not checkpoint:
            return nullcontext()
        from mtpserve.q4_pair import paired_quantized_linears

        return paired_quantized_linears(
            model,
            count_calls=count_calls,
            verification_only=True,
            verification_rows=mtp_depth + 1,
        )

    # Count actual pair coverage only during the excluded warmup.
    with context(True) as warmup_report:
        r = run(PROMPTS[0])
    if checkpoint:
        _validate_pair_warmup(warmup_report, mtp_depth)
    print(
        f"warmup: gen {r['gen_tok_s']:.2f} prefill {r['prefill_tok_s']:.1f} "
        f"accept {r['accept_rate']:.3f}",
        flush=True,
    )

    gens, prefills, accepts = [], [], []
    series_medians = []
    totals = dict(n_gen=0, gen_s=0.0, attempted=0, accepted=0, ssm_checkpointed=0)
    with context(False):
        for s, order in enumerate(ORDERS):
            sg = []
            for i in order:
                r = run(PROMPTS[i])
                for key in totals:
                    totals[key] += r[key]
                gens.append(r["gen_tok_s"])
                prefills.append(r["prefill_tok_s"])
                accepts.append(r["accept_rate"])
                sg.append(r["gen_tok_s"])
                print(
                    f"s{s} p{i}: gen {r['gen_tok_s']:.2f} "
                    f"prefill {r['prefill_tok_s']:.1f} "
                    f"accept {r['accept_rate']:.3f} n_gen {r['n_gen']}",
                    flush=True,
                )
            series_medians.append(statistics.median(sg))

    out = {
        "mtp_depth": mtp_depth,
        "gen_median": round(statistics.median(gens), 2),
        "gen_aggregate": round(totals["n_gen"] / totals["gen_s"], 2),
        "accept_aggregate": (
            totals["accepted"] / totals["attempted"] if totals["attempted"] else 0.0
        ),
        "totals": totals,
        "gen_series_medians": [round(x, 2) for x in series_medians],
        "gen_min": round(min(gens), 2),
        "gen_max": round(max(gens), 2),
        "prefill_median": round(statistics.median(prefills), 1),
        "accept_median": round(statistics.median(accepts), 3),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 1),
    }
    if checkpoint:
        out["experimental_checkpoint_mtp"] = True
    print("RESULT " + json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
