"""CPU control-flow tests: fake arithmetic, real ArraysCache/KVCache.

Positive cases patch only the support predicate; these are NOT real-model or
kernel-contract tests. Greedy predictions depend on both fake SSM states.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mlx.core as mx

mx.set_default_device(mx.cpu)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402
from mtpserve.engine import decode_ids  # noqa: E402


VOCAB = 31
PROMPT = [2, 5, 1]


class Tokenizer:
    eos_token_id = -1

    def decode(self, ids):
        return str(ids)


class FakeModel:
    def __init__(self, accepts=(False,), capture="complete"):
        self.accepts, self.capture = accepts, capture
        self.calls, self.rounds, self.cache_creations = [], 0, 0

    def make_cache(self):
        self.cache_creations += 1
        result = [ArraysCache(2), KVCache(), ArraysCache(2)]
        for i in (0, 2):
            result[i].state = [mx.zeros((1, 3, 1)), mx.zeros((1, 1, 1, 1))]
        return result

    @staticmethod
    def next_token(token, state0, state2):
        return int((token + state0 + state2) % VOCAB)

    @staticmethod
    def logits(token):
        return mx.where(mx.arange(VOCAB) == token, 1.0, -1.0)

    def __call__(self, inputs, cache, return_hidden=False, ssm_checkpoints=None):
        self.calls.append((inputs.shape[1], ssm_checkpoints is not None))
        logits, hidden = [], []
        for position, token in enumerate(inputs.tolist()[0]):
            for i, multiplier, token_scale, modulus in ((0, 3, 1, 97), (2, 5, 2, 89)):
                conv, recurrent = cache[i].state
                conv = mx.concatenate([conv[:, 1:], mx.full((1, 1, 1), token)], axis=1)
                recurrent = (recurrent * multiplier + token * token_scale) % modulus
                cache[i][0] = conv
                cache[i][1] = recurrent
                if (
                    ssm_checkpoints is not None
                    and position == 0
                    and self.capture != "empty"
                    and (self.capture != "partial" or i == 0)
                ):
                    ssm_checkpoints[i] = list(cache[i].state)
            a, b = cache[0][1].item(), cache[2][1].item()
            key = mx.full((1, 1, 1, 1), token, dtype=mx.float32)
            value = mx.full((1, 1, 1, 1), a + b, dtype=mx.float32)
            cache[1].update_and_fetch(key, value)
            logits.append(self.logits(self.next_token(token, a, b)))
            hidden.append(mx.array([a, b]))
        output, states = mx.stack(logits)[None], mx.stack(hidden)[None]
        return (output, states) if return_hidden else output

    def mtp_forward(self, hidden, tokens, **kwargs):
        token = tokens[0, -1].item()
        a, b = hidden[0, -1].tolist()
        predicted = self.next_token(
            token, (a * 3 + token) % 97, (b * 5 + token * 2) % 89
        )
        accept = self.accepts[self.rounds % len(self.accepts)]
        self.rounds += 1
        return self.logits(predicted if accept else (predicted + 1) % VOCAB)[None, None]


def run_decode(model, count=9, ids=None, **kwargs):
    return decode_ids(
        model,
        Tokenizer(),
        PROMPT if ids is None else ids,
        count,
        use_mtp=True,
        mtp_history=False,
        **kwargs,
    )


def continue_greedy(model, cache, count=6):
    token = cache[1].state[0][0, 0, -1, 0].item()
    next_id = model.next_token(token, cache[0][1].item(), cache[2][1].item())
    result = []
    for _ in range(count):
        result.append(next_id)
        logits = model(mx.array([[next_id]]), cache=cache)
        next_id = mx.argmax(logits[0, -1]).item()
    return result


class EngineCheckpointTests(unittest.TestCase):
    def assert_caches_equal(self, left, right):
        self.assertEqual(left[1].offset, right[1].offset)
        for a, b in zip(left, right):
            for x, y in zip(a.state, b.state):
                self.assertEqual(x.shape, y.shape)
                self.assertTrue(mx.all(x == y).item())

    def compare(self, accepts, count, capture="complete", supported=True):
        base_model, candidate_model = FakeModel(accepts), FakeModel(accepts, capture)
        baseline = run_decode(base_model, count)
        if supported:
            with patch("mtpserve.engine._ssm_checkpoint_supported", return_value=True):
                candidate = run_decode(candidate_model, count, ssm_checkpoint=True)
        else:
            # Real support predicate rejects this intentionally unsupported fake model.
            candidate = run_decode(candidate_model, count, ssm_checkpoint=True)
        self.assertEqual(baseline["tokens"], candidate["tokens"])
        self.assertEqual(baseline["attempted"], candidate["attempted"])
        self.assertEqual(baseline["accepted"], candidate["accepted"])
        ca, cb = baseline["state"]["cache"], candidate["state"]["cache"]
        self.assertEqual(cb[1].offset, len(PROMPT) + candidate["n_gen"])
        self.assert_caches_equal(ca, cb)
        # Record replay evidence before the independent continuation adds model calls.
        replays = sum(length == 1 for length, _ in candidate_model.calls)
        captures = sum(captured for _, captured in candidate_model.calls)
        self.assertEqual(
            continue_greedy(base_model, ca), continue_greedy(candidate_model, cb)
        )
        self.assert_caches_equal(ca, cb)
        return candidate, replays, captures

    def test_last_token_reject_keeps_p_without_full_replay(self):
        result, replays, captures = self.compare((False,), 9)
        self.assertEqual((result["ssm_checkpointed"], replays, captures), (9, 0, 9))
        self.assertEqual(result["accepted"], 0)

    def test_mixed_accept_reject_and_continuation(self):
        # A,R,A,R,A,R emits 9 tokens and ends on a rejected draft.
        result, replays, captures = self.compare((True, False), 9)
        self.assertEqual((result["accepted"], result["ssm_checkpointed"]), (3, 3))
        self.assertEqual((replays, captures), (0, 6))

    def test_accepted_verification_keeps_both_tokens_and_full_state(self):
        result, replays, captures = self.compare((True,), 6)
        self.assertEqual((result["accepted"], result["ssm_checkpointed"]), (3, 0))
        self.assertEqual((replays, captures), (0, 3))

    def test_incomplete_checkpoint_falls_back_atomically(self):
        for capture in ("empty", "partial"):
            with self.subTest(capture=capture):
                result, replays, _ = self.compare((False,), 5, capture=capture)
                self.assertTrue(result["ssm_checkpoint_enabled"])
                self.assertEqual((result["ssm_checkpointed"], replays), (0, 5))

    def test_unsupported_contract_uses_full_replay(self):
        result, replays, captures = self.compare((False,), 5, supported=False)
        self.assertFalse(result["ssm_checkpoint_enabled"])
        self.assertEqual((result["ssm_checkpointed"], replays, captures), (0, 5, 0))

    def test_default_false_preserves_reference(self):
        a, b = run_decode(FakeModel()), run_decode(FakeModel(), ssm_checkpoint=False)
        self.assertFalse(a["ssm_checkpoint_enabled"])
        self.assertEqual(a["tokens"], b["tokens"])
        self.assert_caches_equal(a["state"]["cache"], b["state"]["cache"])

    def assert_result_and_continuation_equal(
        self, model_a, result_a, model_b, result_b
    ):
        for field in ("tokens", "attempted", "accepted", "n_prompt", "n_gen"):
            self.assertEqual(result_a[field], result_b[field], field)
        ca, cb = result_a["state"]["cache"], result_b["state"]["cache"]
        self.assert_caches_equal(ca, cb)
        self.assertEqual(continue_greedy(model_a, ca), continue_greedy(model_b, cb))
        self.assert_caches_equal(ca, cb)

    def test_exact_prompt_reuse_matches_fresh_and_continuation(self):
        candidate_model, fresh_model = (
            FakeModel((True, False)),
            FakeModel((True, False)),
        )
        with patch("mtpserve.engine._ssm_checkpoint_supported", return_value=True):
            initial = run_decode(candidate_model, ssm_checkpoint=True)
            reused = run_decode(
                candidate_model, state=initial["state"], ssm_checkpoint=True
            )
            fresh = run_decode(fresh_model, ssm_checkpoint=True)
        self.assertEqual(reused["cached_tokens"], len(PROMPT))
        self.assertEqual(reused["n_prefilled"], 0)
        self.assert_result_and_continuation_equal(
            candidate_model, reused, fresh_model, fresh
        )

    def test_repeated_exact_reuse_preserves_prompt_snapshot(self):
        for checkpoint in (False, True):
            with self.subTest(ssm_checkpoint=checkpoint):
                candidate_model = FakeModel((True, False))
                fresh_model = FakeModel((True, False))
                with patch(
                    "mtpserve.engine._ssm_checkpoint_supported", return_value=True
                ):
                    prior = run_decode(candidate_model, ssm_checkpoint=checkpoint)
                    for _ in range(2):
                        prior = run_decode(
                            candidate_model,
                            state=prior["state"],
                            ssm_checkpoint=checkpoint,
                        )
                        self.assertEqual(prior["cached_tokens"], len(PROMPT))
                        self.assertEqual(prior["n_prefilled"], 0)
                    fresh = run_decode(fresh_model, ssm_checkpoint=checkpoint)
                self.assert_result_and_continuation_equal(
                    candidate_model, prior, fresh_model, fresh
                )

    def test_prefix_extension_reuse_matches_fresh_and_continuation(self):
        candidate_model, fresh_model = (
            FakeModel((True, False)),
            FakeModel((True, False)),
        )
        extended = PROMPT + [8, 6, 7]
        with patch("mtpserve.engine._ssm_checkpoint_supported", return_value=True):
            initial = run_decode(candidate_model, ssm_checkpoint=True)
            reused = run_decode(
                candidate_model,
                ids=extended,
                state=initial["state"],
                prefill_step=2,
                ssm_checkpoint=True,
            )
            fresh = run_decode(fresh_model, ids=extended, ssm_checkpoint=True)
        self.assertEqual(reused["cached_tokens"], len(PROMPT))
        self.assertEqual(reused["n_prefilled"], len(extended) - len(PROMPT))
        self.assert_result_and_continuation_equal(
            candidate_model, reused, fresh_model, fresh
        )

    def test_chunked_prefill_matches_one_chunk_and_continuation(self):
        a, b = FakeModel((True, False)), FakeModel((True, False))
        ids = PROMPT + [8, 6, 7, 3]
        with patch("mtpserve.engine._ssm_checkpoint_supported", return_value=True):
            chunked = run_decode(a, ids=ids, prefill_step=2, ssm_checkpoint=True)
            single = run_decode(b, ids=ids, prefill_step=2048, ssm_checkpoint=True)
        self.assertEqual(chunked["cached_tokens"], 0)
        self.assertEqual(chunked["n_prefilled"], len(ids))
        self.assert_result_and_continuation_equal(a, chunked, b, single)

    def test_mutual_exclusion_precedes_model_or_cache_creation(self):
        model = FakeModel()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            run_decode(model, ssm_recovery=True, ssm_checkpoint=True)
        self.assertEqual(model.calls, [])
        self.assertEqual(model.cache_creations, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
