"""Unit tests for the idle model-unload lifecycle without importing MLX."""

import argparse
import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import unittest
from functools import wraps
from unittest.mock import Mock, patch


class IdleUnloadingServerTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "mtpserve/server.py"
        tree = ast.parse(path.read_text())
        selected = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "IdleUnloadingHTTPServer"
            )
            or (
                isinstance(node, ast.FunctionDef)
                and node.name in ("_track_request_activity", "_idle_timeout")
            )
        ]
        self.ns = {
            "argparse": argparse,
            "math": __import__("math"),
            "threading": threading,
            "time": time,
            "ThreadingHTTPServer": ThreadingHTTPServer,
            "wraps": wraps,
        }
        exec(compile(ast.Module(selected, type_ignores=[]), str(path), "exec"), self.ns)
        self.server = self.ns["IdleUnloadingHTTPServer"](
            ("127.0.0.1", 0), BaseHTTPRequestHandler, idle_timeout=10
        )
        self.addCleanup(self.server.server_close)

    def test_expired_idle_interval_starts_shutdown_once(self):
        self.server._idle_since = 100
        with (
            patch.object(self.ns["time"], "monotonic", return_value=110),
            patch.object(self.ns["threading"], "Thread") as thread,
        ):
            self.server.service_actions()
            self.server.service_actions()
        thread.assert_called_once_with(
            target=self.server.shutdown,
            name="mtpserve-idle-shutdown",
            daemon=True,
        )
        thread.return_value.start.assert_called_once_with()

    def test_active_request_defers_idle_shutdown_until_completion(self):
        self.server._idle_since = 100
        self.server.request_started()
        with (
            patch.object(self.ns["time"], "monotonic", return_value=1000),
            patch.object(self.ns["threading"], "Thread") as thread,
        ):
            self.server.service_actions()
            self.server.request_finished()
            self.server.service_actions()
        thread.assert_not_called()
        self.assertEqual(self.server._active_requests, 0)
        self.assertEqual(self.server._idle_since, 1000)

    def test_server_loop_stops_after_idle_timeout(self):
        self.server.idle_timeout = 0.01
        loop = threading.Thread(target=self.server.serve_forever)
        loop.start()
        loop.join(timeout=2)
        self.assertFalse(loop.is_alive())

    def test_endpoint_wrapper_tracks_completed_request(self):
        request_started, request_finished = Mock(), Mock()
        server = type(
            "Server",
            (),
            {"request_started": request_started, "request_finished": request_finished},
        )()

        @self.ns["_track_request_activity"]
        def endpoint(handler):
            self.assertEqual(handler.server, server)
            return "ok"

        self.assertEqual(endpoint(type("Handler", (), {"server": server})()), "ok")
        request_started.assert_called_once_with()
        request_finished.assert_called_once_with()

    def test_timeout_parser_accepts_zero_or_positive_seconds(self):
        parse = self.ns["_idle_timeout"]
        self.assertEqual(parse("0"), 0)
        self.assertEqual(parse("0.5"), 0.5)
        self.assertEqual(parse("300"), 300)
        for value in ("-1", "nan", "86401", "not-a-number"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                parse(value)


if __name__ == "__main__":
    unittest.main()
