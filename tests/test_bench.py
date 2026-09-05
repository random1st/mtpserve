"""Benchmark accounting excludes warmup and rejects an ineffective experiment."""

import contextlib
import io
import types
import unittest
from unittest.mock import patch

import bench


class BenchmarkTests(unittest.TestCase):
    def run_fake(self, checkpoint, *, effective=True, pair_count=1, mtp_depth=1):
        calls, results = [], []
        core = types.ModuleType("mlx.core")
        core.get_peak_memory = lambda: 123
        mlx = types.ModuleType("mlx")
        mlx.core = core
        engine = types.ModuleType("mtpserve.engine")

        def decode(model, tokenizer, ids, max_tokens, **kwargs):
            calls.append((ids, max_tokens, kwargs))
            i = len(calls)
            result = dict(
                n_gen=200 + i % 2,
                gen_s=float(1 + i % 3),
                prefill_tok_s=100.0,
                accept_rate=0.8,
                attempted=100,
                accepted=80,
                ssm_checkpoint_enabled=checkpoint and effective,
                ssm_checkpointed=20 if checkpoint and effective else 0,
            )
            result["gen_tok_s"] = result["n_gen"] / result["gen_s"]
            results.append(result)
            return result

        engine.decode_ids = decode
        pair_module = types.ModuleType("mtpserve.q4_pair")
        contexts = []

        @contextlib.contextmanager
        def paired(model, *, count_calls, verification_only, verification_rows):
            self.assertEqual(verification_rows, mtp_depth + 1)
            self.assertTrue(verification_only)
            contexts.append(count_calls)
            report = dict(
                patched_projection_count=pair_count,
                supported_projection_count=pair_count,
                pair_calls_by_projection={"projection": 1} if pair_count else {},
            )
            if mtp_depth == 2:
                report["triple_calls_by_projection"] = report.pop(
                    "pair_calls_by_projection"
                )
            try:
                yield report
            finally:
                report.update(
                    classes_restored=True,
                    model_class_restored=True,
                    parameter_objects_unchanged=True,
                )

        pair_module.paired_quantized_linears = paired
        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kw: messages[0]["content"]
        )
        with patch.dict(
            "sys.modules",
            {
                "mlx": mlx,
                "mlx.core": core,
                "mtpserve.engine": engine,
                "mtpserve.q4_pair": pair_module,
            },
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                output = bench._run_benchmark(
                    object(), tokenizer, checkpoint=checkpoint, mtp_depth=mtp_depth
                )
        self.assertEqual(contexts, [True, False] if checkpoint else [])
        return calls, results, output

    def test_weighted_accounting_and_order_exclude_warmup(self):
        calls, results, output = self.run_fake(False)
        expected = [bench.PROMPTS[0]] + [
            bench.PROMPTS[i] for order in bench.ORDERS for i in order
        ]
        self.assertEqual([call[0] for call in calls], expected)
        self.assertTrue(
            all(
                call[1] == bench.MAX_TOKENS and call[2]["ssm_checkpoint"] is False
                for call in calls
            )
        )
        total_tokens = sum(row["n_gen"] for row in results[1:])
        total_seconds = sum(row["gen_s"] for row in results[1:])
        self.assertEqual(output["totals"]["n_gen"], total_tokens)
        self.assertEqual(output["totals"]["gen_s"], total_seconds)
        self.assertEqual(
            output["gen_aggregate"], round(total_tokens / total_seconds, 2)
        )
        self.assertEqual(output["accept_aggregate"], 0.8)
        self.assertNotIn("experimental_checkpoint_mtp", output)

    def test_checkpoint_counts_exclude_warmup(self):
        calls, _, output = self.run_fake(True)
        self.assertTrue(all(call[2]["ssm_checkpoint"] for call in calls))
        self.assertEqual(output["totals"]["ssm_checkpointed"], 12 * 20)
        self.assertTrue(output["experimental_checkpoint_mtp"])

    def test_depth_two_routes_three_row_coverage_and_decode(self):
        calls, _, output = self.run_fake(True, mtp_depth=2)
        self.assertTrue(all(call[2]["mtp_depth"] == 2 for call in calls))
        self.assertEqual(output["mtp_depth"], 2)
        self.assertEqual(output["totals"]["ssm_checkpointed"], 12 * 20)

    def test_missing_pair_coverage_cannot_produce_a_benchmark_result(self):
        with self.assertRaisesRegex(RuntimeError, "every paired Q4"):
            self.run_fake(True, pair_count=0)

    def test_partial_pair_coverage_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "every paired Q4"):
            self.run_fake(True, pair_count=2)

    def test_unsupported_checkpoint_cannot_produce_a_benchmark_result(self):
        with self.assertRaisesRegex(RuntimeError, "every rejected draft"):
            self.run_fake(True, effective=False)


if __name__ == "__main__":
    unittest.main()
