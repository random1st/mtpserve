# SPDX-License-Identifier: Apache-2.0
"""Фиксированный стенд: медиана gen tok/s (с MTP) и prefill tok/s.

3 серии x 4 коротких код-промпта, max_tokens=200, прогрев одним прогоном,
порядок промптов чередуется между сериями. Состояние кэша не переиспользуется
(state=None) — префилл честный в каждом прогоне.

Запуск: cd ~/src/mtpserve && uv run python bench.py
"""

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


def main():
    import mlx.core as mx

    from mtpserve.engine import decode_ids, raise_wired_limit
    from mtpserve.loader import load_model

    raise_wired_limit()
    t0 = time.perf_counter()
    model, tokenizer = load_model(MODEL)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    def ids_for(p):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=True,
        )

    # прогрев
    r = decode_ids(model, tokenizer, ids_for(PROMPTS[0]), MAX_TOKENS, use_mtp=True)
    print(f"warmup: gen {r['gen_tok_s']:.2f} prefill {r['prefill_tok_s']:.1f} "
          f"accept {r['accept_rate']:.3f}", flush=True)

    gens, prefills, accepts = [], [], []
    series_medians = []
    for s, order in enumerate(ORDERS):
        sg = []
        for i in order:
            r = decode_ids(model, tokenizer, ids_for(PROMPTS[i]), MAX_TOKENS,
                           use_mtp=True)
            gens.append(r["gen_tok_s"])
            prefills.append(r["prefill_tok_s"])
            accepts.append(r["accept_rate"])
            sg.append(r["gen_tok_s"])
            print(f"s{s} p{i}: gen {r['gen_tok_s']:.2f} "
                  f"prefill {r['prefill_tok_s']:.1f} "
                  f"accept {r['accept_rate']:.3f} n_gen {r['n_gen']}", flush=True)
        series_medians.append(statistics.median(sg))

    out = {
        "gen_median": round(statistics.median(gens), 2),
        "gen_series_medians": [round(x, 2) for x in series_medians],
        "gen_min": round(min(gens), 2),
        "gen_max": round(max(gens), 2),
        "prefill_median": round(statistics.median(prefills), 1),
        "accept_median": round(statistics.median(accepts), 3),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 1),
    }
    print("RESULT " + json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
