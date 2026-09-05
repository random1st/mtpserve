"""CPU depth2 controls using fake arithmetic and real MLX cache containers.

Positive cases patch the support predicate and MTP step; they do not validate
real model/kernel arithmetic. Cache and continuation equality are exact here.
"""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mlx.core as mx

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_lm.models.cache import KVCache  # noqa: E402
from mtpserve import engine  # noqa: E402
from test_engine_checkpoint import FakeModel, PROMPT, Tokenizer, VOCAB, continue_greedy  # noqa: E402


class Depth2Model(FakeModel):
    def __init__(self, accepts=((True, True),), capture="complete"):
        super().__init__()
        self.pattern, self.capture, self.draft_calls = accepts, capture, 0

    def make_mtp_cache(self):
        return [KVCache()]

    @staticmethod
    def write_head(hidden, tokens, mtp_cache):
        mtp_cache[0].update_and_fetch(
            tokens.astype(mx.float32)[:, None, :, None],
            hidden.sum(axis=-1).astype(mx.float32)[:, None, :, None],
        )

    def mtp_forward(self, hidden, tokens, mtp_cache=None, **kwargs):
        self.write_head(hidden, tokens, mtp_cache)
        return self.logits(0)[None, None]

    def draft_step(self, hidden, tokens, mtp_cache):
        self.write_head(hidden, tokens, mtp_cache)
        token = tokens[0, -1].item()
        a, b = hidden[0, -1].tolist()
        a, b = (a * 3 + token) % 97, (b * 5 + token * 2) % 89
        predicted = self.next_token(token, a, b)
        stage = self.draft_calls % 2
        accept = self.pattern[(self.draft_calls // 2) % len(self.pattern)][stage]
        self.draft_calls += 1
        draft = predicted if accept else (predicted + 1) % VOCAB
        return self.logits(draft)[None, None], mx.array([[[a, b]]])

    def __call__(self, inputs, cache, return_hidden=False, ssm_checkpoints=None):
        # Reuse the existing recurrent arithmetic one position at a time, while
        # recording the actual external model calls for replay assertions.
        start, outputs, hiddens = len(self.calls), [], []
        for position in range(inputs.shape[1]):
            output, hidden = super().__call__(
                inputs[:, position : position + 1], cache, True
            )
            outputs.append(output)
            hiddens.append(hidden)
            if ssm_checkpoints is not None and position < 2:
                for index in (0, 2):
                    ssm_checkpoints.setdefault(index, {})[position + 1] = list(
                        cache[index].state
                    )
        del self.calls[start:]
        self.calls.append((inputs.shape[1], ssm_checkpoints is not None))
        if ssm_checkpoints is not None:
            if self.capture == "empty":
                ssm_checkpoints.clear()
            elif self.capture == "partial":
                ssm_checkpoints.pop(2)
            elif self.capture == "missing_second":
                ssm_checkpoints[2].pop(2)
            elif self.capture == "bad_dtype":
                ssm_checkpoints[2][2][1] = ssm_checkpoints[2][2][1].astype(mx.bfloat16)
        result = mx.concatenate(outputs, axis=1), mx.concatenate(hiddens, axis=1)
        return result if return_hidden else result[0]


def run(
    model,
    count=6,
    *,
    checkpoint=False,
    supported=True,
    tokenizer=None,
    state=None,
    ids=PROMPT,
):
    with (
        patch(
            "mtpserve.engine._mtp_step",
            side_effect=lambda model, h, t, c: model.draft_step(h, t, c),
        ),
        patch("mtpserve.engine._ssm_checkpoint_supported", return_value=supported),
    ):
        return engine.decode_ids(
            model,
            tokenizer or Tokenizer(),
            ids,
            count,
            use_mtp=True,
            mtp_depth=2,
            ssm_checkpoint=checkpoint,
            state=state,
        )


class Checkpoint3Tests(unittest.TestCase):
    def assert_cache_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual(getattr(a, "offset", None), getattr(b, "offset", None))
            for x, y in zip(a.state, b.state):
                self.assertEqual(x.shape, y.shape)
                self.assertEqual(x.dtype, y.dtype)
                self.assertTrue(mx.all(x == y).item())

    def assert_results_equal(self, left, right):
        for key in ("tokens", "attempted", "accepted", "n_gen", "n_prompt"):
            self.assertEqual(left[key], right[key], key)
        for key in ("cache", "mtp_cache"):
            self.assert_cache_equal(left["state"][key], right["state"][key])
        self.assertEqual(
            left["state"]["prompt_mtp_offsets"], right["state"]["prompt_mtp_offsets"]
        )

    def compare(
        self, pattern, count, *, capture="complete", supported=True, tokenizer=None
    ):
        a, b = Depth2Model(pattern), Depth2Model(pattern, capture)
        baseline = run(a, count, tokenizer=tokenizer)
        candidate = run(
            b, count, checkpoint=True, supported=supported, tokenizer=tokenizer
        )
        self.assert_results_equal(baseline, candidate)
        # Independent continuation from copies checks the selected SSM and KV prefix.
        ca, cb = (
            copy.deepcopy(baseline["state"]["cache"]),
            copy.deepcopy(candidate["state"]["cache"]),
        )
        self.assertEqual(continue_greedy(a, ca), continue_greedy(b, cb))
        self.assert_cache_equal(ca, cb)
        return candidate, b.calls[:-6]  # Drop the six independent continuation calls.

    def test_d1_reject_restores_p_without_replay(self):
        result, calls = self.compare(((False, False),), 1)
        self.assertEqual(
            (result["attempted"], result["accepted"], result["ssm_checkpointed"]),
            (1, 0, 1),
        )
        self.assertEqual(calls, [(3, False), (3, True)])

    def test_d2_reject_restores_pd_without_replay(self):
        result, calls = self.compare(((True, False),), 2)
        self.assertEqual(
            (result["attempted"], result["accepted"], result["ssm_checkpointed"]),
            (2, 1, 1),
        )
        self.assertEqual(calls, [(3, False), (3, True)])

    def test_all_accept_keeps_full_state(self):
        result, calls = self.compare(((True, True),), 6)
        self.assertEqual(
            (result["attempted"], result["accepted"], result["ssm_checkpointed"]),
            (4, 4, 0),
        )
        self.assertEqual(calls, [(3, False), (3, True), (3, True)])

    def test_mixed_accept_both_rejects_and_continuation(self):
        result, calls = self.compare(((True, True), (False, False), (True, False)), 12)
        self.assertEqual(result["ssm_checkpointed"], 4)
        self.assertEqual(result["attempted"] - result["accepted"], 4)
        self.assertTrue(all(length == 3 for length, _ in calls))

    def test_eos_at_primary_d1_d2_counts_only_actual_attempts(self):
        # Fake model's first greedy tokens from PROMPT are 5,11,4.
        for eos, expected in ((5, (1, 0, 0)), (11, (2, 1, 1)), (4, (3, 2, 2))):
            with self.subTest(eos=eos):
                tokenizer = Tokenizer()
                tokenizer.eos_token_id = eos
                result, _ = self.compare(((True, True),), 6, tokenizer=tokenizer)
                self.assertEqual(
                    (result["n_gen"], result["attempted"], result["accepted"]), expected
                )
                self.assertEqual(result["ssm_checkpointed"], 0)
                self.assertEqual(result["attempted"] - result["accepted"], 0)

    def test_incomplete_capture_falls_back_for_both_reject_paths(self):
        for capture in ("empty", "partial", "missing_second", "bad_dtype"):
            for pattern, count in ((((False, False),), 1), (((True, False),), 2)):
                with self.subTest(capture=capture, pattern=pattern):
                    result, calls = self.compare(pattern, count, capture=capture)
                    self.assertEqual(result["ssm_checkpointed"], 0)
                    self.assertEqual(calls[-1], (count, False))
                    self.assertTrue(result["ssm_checkpoint_enabled"])

    def test_unsupported_uses_stock_verification_and_replay(self):
        result, calls = self.compare(((True, False),), 2, supported=False)
        self.assertFalse(result["ssm_checkpoint_enabled"])
        self.assertEqual(calls, [(3, False), (3, False), (2, False)])

    def test_invalid_unused_prefix_or_unreachable_kv_is_atomic(self):
        model = Depth2Model()
        for invalid in ("unused_prefix", "kv_offset"):
            cache = model.make_cache()
            key = mx.zeros((1, 1, 3, 1))
            cache[1].update_and_fetch(key, key)
            snapshots = engine._snapshot_recurrent(cache)
            checkpoints = {
                i: {
                    1: [mx.ones_like(x) for x in state],
                    2: [mx.ones_like(x) for x in state],
                }
                for i, state in snapshots.items()
            }
            if invalid == "unused_prefix":
                checkpoints[2][2][1] = mx.zeros((1, 2))
            else:
                cache[1].offset = 1
            old_states = {i: list(cache[i].state) for i in snapshots}
            # KVCache.state creates logical slice views on each property read;
            # compare backing object identity and offset to detect mutation.
            kv_arrays = cache[1].keys, cache[1].values
            offset = cache[1].offset
            self.assertIsNone(
                engine._restore_checkpoint_prefix(cache, snapshots, checkpoints, 1)
            )
            self.assertEqual(cache[1].offset, offset)
            self.assertIs(cache[1].keys, kv_arrays[0])
            self.assertIs(cache[1].values, kv_arrays[1])
            for i, before in old_states.items():
                self.assertTrue(all(a is b for a, b in zip(cache[i].state, before)))

    def test_actual_verification_width_passed_to_support_predicate(self):
        for history, width in ((True, 3), (False, 2)):
            with patch(
                "mtpserve.engine._ssm_checkpoint_supported", return_value=False
            ) as support:
                engine.decode_ids(
                    Depth2Model(),
                    Tokenizer(),
                    PROMPT,
                    0,
                    use_mtp=True,
                    mtp_history=history,
                    mtp_depth=2,
                    ssm_checkpoint=True,
                )
            self.assertEqual(support.call_args.kwargs["verification_tokens"], width)

    def test_repeated_exact_prompt_reuse_matches_fresh(self):
        pattern = ((True, True), (False, False), (True, False))
        a, b = Depth2Model(pattern), Depth2Model(pattern)
        baseline, candidate = run(a), run(b, checkpoint=True)
        for _ in range(2):
            baseline = run(a, state=baseline["state"])
            candidate = run(b, checkpoint=True, state=candidate["state"])
            self.assert_results_equal(baseline, candidate)
            self.assertEqual(candidate["cached_tokens"], len(PROMPT))
            self.assertEqual(candidate["n_prefilled"], 0)
            fresh = run(Depth2Model(pattern), checkpoint=True)
            self.assert_results_equal(fresh, candidate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
