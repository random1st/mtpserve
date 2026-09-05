"""Server option/lifecycle tests using canonical AST and stubs; no MLX or sockets."""

import argparse
import ast
import contextlib
import io
from pathlib import Path
import time
import types
import unittest
from unittest.mock import Mock, patch


class ServerCheckpointTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "mtpserve/server.py"
        tree = ast.parse(path.read_text())
        selected = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "main",
                "_decode_job",
                "_idle_timeout",
            ):
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id in ("USE_MTP", "USE_CHECKPOINT_MTP", "MTP_DEPTH")
                for target in node.targets
            ):
                selected.append(node)
        self.ns = dict(
            __package__="mtpserve",
            argparse=argparse,
            nullcontext=contextlib.nullcontext,
            time=time,
        )
        exec(compile(ast.Module(selected, type_ignores=[]), str(path), "exec"), self.ns)
        self.events, self.active = [], False
        self.model = types.SimpleNamespace(mtp=object(), supports_ssm_checkpoint=True)
        self.server = types.SimpleNamespace(
            daemon_threads=True,
            serve_forever=self.serve,
            server_close=lambda: self.events.append("close"),
        )
        self.old_state = {"ids": [1, 2]}
        self.result = dict(
            state={"ids": [1, 2], "new": True},
            n_gen=3,
            attempted=2,
            accepted=1,
            cached_tokens=2,
            n_prefilled=0,
            ssm_checkpoint_enabled=True,
            ssm_checkpointed=1,
        )
        self.ns.update(
            MODEL=self.model,
            TOKENIZER=object(),
            STATE={"v": self.old_state},
            load_model=Mock(side_effect=lambda *a, **kw: (self.model, object())),
            raise_wired_limit=Mock(),
            _load_pin=lambda: self.event("pin"),
            IdleUnloadingHTTPServer=lambda *a, **kw: self.bind(),
            Handler=object,
            decode_ids=Mock(return_value=self.result),
            STATS=dict(
                requests=0,
                gen_tokens=0,
                attempted=0,
                accepted=0,
                cached_tokens=0,
                computed_tokens=0,
                starts=dict(exact=0, extension=0, pin=0, miss=0),
            ),
        )
        self.pair_error, self.pair_count = None, 1
        pair_module = types.ModuleType("mtpserve.q4_pair")

        @contextlib.contextmanager
        def paired(model, *, verification_only, count_calls=False, verification_rows=2):
            self.assertEqual(verification_rows, self.ns["MTP_DEPTH"] + 1)
            self.assertIs(model, self.model)
            self.assertTrue(verification_only)
            self.assertFalse(count_calls)
            self.events.append("pair_enter")
            if self.pair_error:
                raise self.pair_error
            self.active = True
            try:
                yield {"patched_projection_count": self.pair_count}
            finally:
                self.active = False
                self.events.append("pair_exit")

        pair_module.paired_quantized_linears = paired
        modules = patch.dict("sys.modules", {"mtpserve.q4_pair": pair_module})
        modules.start()
        self.addCleanup(modules.stop)
        self.tree, self.path = tree, path

    def event(self, name):
        self.events.append(name)
        self.assertEqual(self.active, self.ns["USE_CHECKPOINT_MTP"])

    def bind(self):
        self.event("bind")
        return self.server

    def serve(self):
        self.event("serve")

    def main(self, *flags):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.ns["main"](["--model", "/fake/model", *flags])

    def test_default_and_no_mtp_do_not_enter_pair_context(self):
        self.assertFalse(self.ns["USE_CHECKPOINT_MTP"])
        self.main()
        self.assertTrue(self.ns["USE_MTP"])
        self.assertTrue(self.server.daemon_threads)
        self.assertEqual(self.events, ["pin", "bind", "serve", "close"])
        self.events.clear()
        self.main("--no-mtp")
        self.assertFalse(self.ns["USE_MTP"])
        self.assertFalse(self.ns["USE_CHECKPOINT_MTP"])
        self.assertNotIn("pair_enter", self.events)

    def test_conflicting_flags_fail_before_model_load(self):
        with self.assertRaises(SystemExit) as error:
            self.main("--no-mtp", "--checkpoint-mtp")
        self.assertEqual(error.exception.code, 2)
        self.ns["load_model"].assert_not_called()
        self.assertEqual(self.events, [])

    def test_pair_lifetime_covers_pin_and_checkpoint_decode(self):
        def serve():
            self.event("serve")
            self.assertFalse(self.server.daemon_threads)
            self.assertIs(self.ns["_decode_job"]([1, 2], 3), self.result)

        self.server.serve_forever = serve
        self.main("--checkpoint-mtp")
        self.assertEqual(
            self.events, ["pair_enter", "pin", "bind", "serve", "close", "pair_exit"]
        )
        call = self.ns["decode_ids"].call_args
        self.assertTrue(call.kwargs["use_mtp"])
        self.assertTrue(call.kwargs["ssm_checkpoint"])
        self.assertIs(self.ns["STATE"]["v"], self.result["state"])
        self.assertFalse(self.active)

    def test_depth_two_context_and_decode_use_three_rows(self):
        def serve():
            self.event("serve")
            self.ns["_decode_job"]([1, 2], 3)

        self.server.serve_forever = serve
        self.main("--checkpoint-mtp", "--mtp-depth", "2")
        self.assertEqual(self.ns["decode_ids"].call_args.kwargs["mtp_depth"], 2)
        self.assertEqual(self.events[-2:], ["close", "pair_exit"])

    def test_no_mtp_depth_two_fails_before_model_load(self):
        with self.assertRaises(SystemExit):
            self.main("--no-mtp", "--mtp-depth", "2")
        self.ns["load_model"].assert_not_called()

    def test_serve_exception_closes_server_before_pair_cleanup(self):
        self.server.serve_forever = Mock(side_effect=RuntimeError("serve failed"))
        with self.assertRaisesRegex(RuntimeError, "serve failed"):
            self.main("--checkpoint-mtp")
        self.assertEqual(self.events[-2:], ["close", "pair_exit"])
        self.assertFalse(self.active)

    def test_unsupported_feature_fails_before_bind_or_pin(self):
        for head, support in ((None, True), (object(), False)):
            with self.subTest(head=head, support=support):
                self.events.clear()
                self.model.mtp, self.model.supports_ssm_checkpoint = head, support
                with self.assertRaisesRegex(ValueError, "MTP head and SSM checkpoint"):
                    self.main("--checkpoint-mtp")
                self.assertEqual(self.events, [])

    def test_invalid_weights_or_missing_projections_fail_before_bind(self):
        self.pair_error = ValueError("unsupported weights")
        with self.assertRaisesRegex(ValueError, "unsupported weights"):
            self.main("--checkpoint-mtp")
        self.assertEqual(self.events, ["pair_enter"])
        self.pair_error, self.pair_count = None, 0
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "paired Q4 projections"):
            self.main("--checkpoint-mtp")
        self.assertEqual(self.events, ["pair_enter", "pair_exit"])

    def test_default_decode_has_no_effective_mode_requirement(self):
        self.result.pop("ssm_checkpoint_enabled")
        self.result.pop("ssm_checkpointed")
        self.ns["_decode_job"]([1, 2], 3)
        self.assertFalse(self.ns["decode_ids"].call_args.kwargs["ssm_checkpoint"])
        self.assertIs(self.ns["STATE"]["v"], self.result["state"])

    def test_ineffective_decode_discards_mutated_state_before_stats_update(self):
        self.ns["USE_CHECKPOINT_MTP"] = True
        for enabled, recovered in ((False, 1), (True, 0)):
            with self.subTest(enabled=enabled, recovered=recovered):
                self.ns["STATE"]["v"] = self.old_state
                self.result.update(
                    ssm_checkpoint_enabled=enabled, ssm_checkpointed=recovered
                )

                def decode(*args, **kwargs):
                    kwargs["state"]["mutated"] = True
                    return self.result

                self.ns["decode_ids"].side_effect = decode
                with self.assertRaisesRegex(RuntimeError, "every rejected draft"):
                    self.ns["_decode_job"]([1, 2], 3)
                self.assertIsNone(self.ns["STATE"]["v"])
                self.assertTrue(self.old_state["mutated"])
                self.assertEqual(self.ns["STATS"]["gen_tokens"], 0)

    def test_checkpoint_exception_discards_state_default_exception_preserves_it(self):
        self.ns["decode_ids"].side_effect = ValueError("projection failed")
        for checkpoint in (False, True):
            self.ns["USE_CHECKPOINT_MTP"] = checkpoint
            self.ns["STATE"]["v"] = self.old_state
            with self.assertRaisesRegex(ValueError, "projection failed"):
                self.ns["_decode_job"]([1, 2], 3)
            self.assertIs(self.ns["STATE"]["v"], None if checkpoint else self.old_state)

    def test_opt_in_socket_timeout_bounds_idle_http_keepalive(self):
        handler = next(
            n
            for n in self.tree.body
            if isinstance(n, ast.ClassDef) and n.name == "Handler"
        )
        setup = next(
            n
            for n in handler.body
            if isinstance(n, ast.FunctionDef) and n.name == "setup"
        )
        handler.body = [setup]
        base = type("Base", (), {"setup": lambda instance: None})
        self.ns["BaseHTTPRequestHandler"] = base
        exec(
            compile(ast.Module([handler], type_ignores=[]), str(self.path), "exec"),
            self.ns,
        )
        instance = self.ns["Handler"]()
        instance.connection = Mock()
        instance.setup()
        instance.connection.settimeout.assert_not_called()
        self.ns["USE_CHECKPOINT_MTP"] = True
        instance.setup()
        instance.connection.settimeout.assert_called_once_with(30)


if __name__ == "__main__":
    unittest.main()
