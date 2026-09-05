"""CPU MTP history regression: fake head arithmetic, real KVCache offsets.

No model loading/GPU. Uses generation-free controls to measure the actual MTP
prompt boundary: extension deliberately skips its first new head position.
"""

import ast
from pathlib import Path
import sys
import unittest

import mlx.core as mx

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402
from mtpserve.engine import decode_ids  # noqa: E402
from test_engine_checkpoint import FakeModel, PROMPT, Tokenizer  # noqa: E402


class HistoryModel(FakeModel):
    def make_mtp_cache(self):
        return [KVCache()]

    def mtp_forward(self, hidden, tokens, mtp_cache=None, **kwargs):
        if mtp_cache is not None:
            keys = tokens.astype(mx.float32)[:, None, :, None]
            values = hidden.sum(axis=-1)[:, None, :, None]
            mtp_cache[0].update_and_fetch(keys, values)
        return super().mtp_forward(hidden, tokens, **kwargs)


def generate(model, ids=PROMPT, count=0, state=None):
    return decode_ids(
        model, Tokenizer(), ids, count, use_mtp=True, mtp_history=True, state=state
    )


def mtp(result):
    return result["state"]["mtp_cache"][0]


class MTPReuseTests(unittest.TestCase):
    def assert_mtp_equal(self, actual, expected):
        self.assertEqual(
            actual.offset,
            expected.offset,
            f"MTP offset {actual.offset} != prompt boundary {expected.offset}",
        )
        for a, b in zip(actual.state, expected.state):
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(mx.all(a == b).item())

    def test_exact_reuse_restores_head_prompt_boundary(self):
        control = generate(HistoryModel())
        model = HistoryModel()
        generated = generate(model, count=4)
        self.assertGreater(mtp(generated).offset, mtp(control).offset)
        reused = generate(model, state=generated["state"])
        self.assertEqual(reused["cached_tokens"], len(PROMPT))
        self.assertEqual(reused["n_prefilled"], 0)
        self.assert_mtp_equal(mtp(reused), mtp(control))

    def test_extension_restores_head_then_appends_only_its_actual_positions(self):
        extended = PROMPT + [8, 6, 7]
        control_model = HistoryModel()
        control_start = generate(control_model)
        control = generate(control_model, extended, state=control_start["state"])
        # Current extension skips one boundary position: 2 + (3 - 1) = 4, not 5.
        self.assertEqual(mtp(control).offset, 4)
        self.assertNotEqual(mtp(control).offset, len(extended) - 1)
        model = HistoryModel()
        generated = generate(model, count=4)
        reused = generate(model, extended, state=generated["state"])
        self.assertEqual(reused["cached_tokens"], len(PROMPT))
        self.assertEqual(reused["n_prefilled"], 3)
        self.assert_mtp_equal(mtp(reused), mtp(control))

    def test_exact_after_extension_uses_measured_not_inferred_boundary(self):
        extended = PROMPT + [8, 6, 7]
        control_model = HistoryModel()
        control_start = generate(control_model)
        control = generate(control_model, extended, state=control_start["state"])
        model = HistoryModel()
        start = generate(model)
        generated = generate(model, extended, count=4, state=start["state"])
        reused = generate(model, extended, state=generated["state"])
        self.assertEqual(mtp(control).offset, 4)
        self.assert_mtp_equal(mtp(reused), mtp(control))

    def test_saved_offsets_and_repeated_exact_reuse(self):
        model = HistoryModel()
        prior = generate(model, count=4)
        self.assertEqual(prior["state"]["prompt_mtp_offsets"], [2])
        control = generate(HistoryModel(), count=4)
        for _ in range(3):
            prior = generate(model, count=4, state=prior["state"])
            self.assertEqual(prior["cached_tokens"], len(PROMPT))
            self.assertEqual(prior["n_prefilled"], 0)
            self.assertEqual(prior["state"]["prompt_mtp_offsets"], [2])
            self.assertEqual(prior["tokens"], control["tokens"])
            self.assert_mtp_equal(mtp(prior), mtp(control))

    def test_legacy_malformed_and_unreachable_offsets_fall_back_without_mutation(self):
        invalid = (
            "missing",
            None,
            [],
            [2, 2],
            [-1],
            [None],
            ["2"],
            [2.0],
            [True],
            [999],
        )
        for offsets in invalid:
            with self.subTest(offsets=offsets):
                model = HistoryModel()
                generated = generate(model, count=4)
                old = generated["state"]
                if offsets == "missing":
                    old.pop("prompt_mtp_offsets")
                else:
                    old["prompt_mtp_offsets"] = offsets
                old_kv_offset = old["cache"][1].offset
                old_head_offset = old["mtp_cache"][0].offset
                before = [[array.tolist() for array in c.state] for c in old["cache"]]
                for ids in (PROMPT, PROMPT + [8, 6, 7]):
                    result = generate(model, ids, state=old)
                    control = generate(HistoryModel(), ids)
                    self.assertEqual(result["cached_tokens"], 0)
                    self.assertEqual(result["n_prefilled"], len(ids))
                    self.assertIsNot(result["state"]["cache"], old["cache"])
                    self.assert_mtp_equal(mtp(result), mtp(control))
                    self.assertEqual(old["cache"][1].offset, old_kv_offset)
                    self.assertEqual(old["mtp_cache"][0].offset, old_head_offset)
                    self.assertEqual(
                        [[a.tolist() for a in c.state] for c in old["cache"]], before
                    )

    def test_all_heads_validated_before_any_cache_changes(self):
        class TwoHeadHistory(HistoryModel):
            def make_mtp_cache(self):
                return [KVCache(), KVCache()]

            def mtp_forward(self, hidden, tokens, mtp_cache=None, **kwargs):
                logits = super().mtp_forward(
                    hidden, tokens, mtp_cache=mtp_cache, **kwargs
                )
                keys = tokens.astype(mx.float32)[:, None, :, None]
                mtp_cache[1].update_and_fetch(keys, keys)
                return logits

        model = TwoHeadHistory()
        generated = generate(model, count=4)
        old = generated["state"]
        self.assertEqual(old["prompt_mtp_offsets"], [2, 2])
        old["prompt_mtp_offsets"] = [2, 999]
        result = generate(model, state=old)
        self.assertEqual(result["cached_tokens"], 0)
        self.assertEqual([c.offset for c in old["mtp_cache"]], [6, 6])
        self.assertEqual(old["cache"][1].offset, 7)

    def test_canonical_pin_restore_recovers_original_head_and_copies_buffers(self):
        model = HistoryModel()
        seed = generate(model)
        pin = {
            "ids": PROMPT,
            "states": [c.state for c in seed["state"]["cache"]],
            "mtp_states": [c.state for c in seed["state"]["mtp_cache"]],
        }
        # Execute exactly the canonical pure helpers without starting server threads.
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "mtpserve/server.py").read_text()
        )
        names = {"_copy_arr", "_copy_state", "_restore_pin"}
        module = ast.Module(
            [
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in names
            ],
            type_ignores=[],
        )
        namespace = {
            "mx": mx,
            "MODEL": model,
            "PIN": pin,
            "make_prompt_cache": make_prompt_cache,
        }
        exec(compile(module, "<canonical-pin-helpers>", "exec"), namespace)
        restored = namespace["_restore_pin"]()
        self.assertEqual(restored["prompt_mtp_offsets"], [2])
        self.assert_mtp_equal(restored["mtp_cache"][0], mtp(seed))
        values = mx.zeros((1, 1, 1, 1))
        restored["mtp_cache"][0].update_and_fetch(values, values)
        self.assertEqual(restored["mtp_cache"][0].offset, 3)
        self.assertEqual(pin["mtp_states"][0][0].shape[2], 2)
        self.assertEqual(mtp(seed).offset, 2)
        extended = generate(
            model, PROMPT + [8, 6, 7], state=namespace["_restore_pin"]()
        )
        self.assertEqual(extended["cached_tokens"], len(PROMPT))
        self.assertEqual(extended["state"]["prompt_mtp_offsets"], [4])
        self.assertEqual(mtp(extended).offset, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
