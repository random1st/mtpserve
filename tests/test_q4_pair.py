"""CPU-only adapter/launch contract tests with stub MLX; no Metal/model execution.

These tests check dispatch, validation and cleanup, not numerical kernel parity.
Real-device parity needs separate verification against the installed stock kernel.
"""

from math import prod
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtpserve import q4_pair, q4_triple  # noqa: E402


class Array:
    def __init__(self, shape, dtype="bfloat16", values=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.size = len(shape), prod(shape)
        self.values = [0] * self.size if values is None else list(values)

    def reshape(self, *shape):
        assert prod(shape) == self.size
        return Array(shape, self.dtype, self.values)

    def __add__(self, other):
        return Array(
            self.shape,
            self.dtype,
            [
                value + other.values[i % other.size]
                for i, value in enumerate(self.values)
            ],
        )


class Module(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        if name == "__class__":
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def named_modules(self):
        result, stack = [], [("", self)]
        while stack:
            name, module = stack.pop()
            result.append((name, module))
            stack.extend(
                (f"{name}.{key}".lstrip("."), value)
                for key, value in module.items()
                if isinstance(value, Module)
            )
        return result


class QuantizedLinear(Module):
    def __init__(self, k=512, n=8, bias=False):
        self.mode, self.bits, self.group_size = "affine", 4, 64
        self.weight = Array((n, k // 8), "uint32")
        self.scales, self.biases = Array((n, k // 64)), Array((n, k // 64))
        if bias:
            self.bias = Array((n,), values=range(n))
        self.stock_calls = []

    def __call__(self, x):
        self.stock_calls.append(x)
        shape = (*x.shape[:-1], self.weight.shape[0])
        result = Array(shape, x.dtype, [7] * prod(shape))
        return result + self.bias if "bias" in self else result


class CheckpointModel(Module):
    supports_ssm_checkpoint = True

    def __call__(self, x, *, ssm_checkpoints=None, fail=False):
        result = self.projection(x)
        if fail:
            raise RuntimeError("model failure")
        return result

    def mtp_forward(self, x):
        return self.projection(x)


class Q4PairTests(unittest.TestCase):
    def setUp(self):
        core, nn, mlx = (ModuleType(name) for name in ("mlx.core", "mlx.nn", "mlx"))
        core.bfloat16, core.uint32 = "bfloat16", "uint32"
        core.fast = ModuleType("mlx.core.fast")
        core.fast.metal_kernel = Mock(name="metal_kernel")
        nn.Module, nn.QuantizedLinear = Module, QuantizedLinear
        mlx.core, mlx.nn = core, nn
        self.modules = patch.dict(
            sys.modules,
            {
                "mlx": mlx,
                "mlx.core": core,
                "mlx.nn": nn,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        q4_pair._kernel.cache_clear()
        self.addCleanup(q4_pair._kernel.cache_clear)
        q4_triple._kernel.cache_clear()
        self.addCleanup(q4_triple._kernel.cache_clear)
        self.core = core

    def fixture(self, *, bias=False):
        layer = QuantizedLinear(bias=bias)
        model = Module(projection=layer, alias=layer)
        params = {
            key: layer[key]
            for key in ("weight", "scales", "biases", "bias")
            if key in layer
        }
        return model, layer, params

    def assert_restored(self, layer, params, report=None):
        self.assertIs(type(layer), QuantizedLinear)
        for key, value in params.items():
            self.assertIs(layer[key], value)
        if report is not None:
            self.assertTrue(report["classes_restored"])
            self.assertTrue(report["model_class_restored"])
            self.assertTrue(report["parameter_objects_unchanged"])

    def test_shape_contract(self):
        self.assertEqual(
            q4_pair._validate_shapes((2, 512), (8, 64), (8, 8), (8, 8)), (8, 512)
        )
        self.assertEqual(
            q4_pair._validate_shapes((2, 1024), (16, 128), (16, 16), (16, 16)),
            (16, 1024),
        )
        invalid = [
            ((1, 512), (8, 64), (8, 8), (8, 8)),
            ((1, 2, 512), (8, 64), (8, 8), (8, 8)),
            ((2, 0), (8, 0), (8, 0), (8, 0)),
            ((2, 256), (8, 32), (8, 4), (8, 4)),
            ((2, 512), (0, 64), (0, 8), (0, 8)),
            ((2, 512), (7, 64), (7, 8), (7, 8)),
            ((2, 512), (8,), (8, 8), (8, 8)),
            ((2, 512), (8, 63), (8, 8), (8, 8)),
            ((2, 512), (8, 64), (8, 7), (8, 8)),
            ((2, 512), (8, 64), (8, 8), (8, 8, 1)),
        ]
        for shapes in invalid:
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                q4_pair._validate_shapes(*shapes)

    def test_launch_preserves_fixed_layout(self):
        x, layer = Array((2, 512)), QuantizedLinear()
        output = Array((2, 8))
        launch = Mock(return_value=[output])
        with patch.object(q4_pair, "_kernel", return_value=launch):
            self.assertIs(
                q4_pair.qmv_fast_pair(x, layer.weight, layer.scales, layer.biases),
                output,
            )
        self.assertEqual(
            launch.call_args.kwargs,
            {
                "inputs": [layer.weight, layer.scales, layer.biases, x],
                "template": [("T", "bfloat16"), ("K", 512), ("N", 8), ("M", 2)],
                "grid": (32, 2, 1),
                "threadgroup": (32, 2, 1),
                "output_shapes": [(2, 8)],
                "output_dtypes": ["bfloat16"],
            },
        )
        q4_pair._kernel()
        options = self.core.fast.metal_kernel.call_args.kwargs
        self.assertTrue(options["ensure_row_contiguous"])
        self.assertEqual(options["compile_options"], {"math_mode": "safe"})
        self.assertEqual(options["header"], q4_pair.HEADER)
        self.assertEqual(options["source"], q4_pair.SOURCE)

    def test_invalid_dtypes_do_not_create_kernel(self):
        for index in range(4):
            layer = QuantizedLinear()
            args = [Array((2, 512)), layer.weight, layer.scales, layer.biases]
            args[index].dtype = "float32"
            with self.subTest(index=index), patch.object(q4_pair, "_kernel") as kernel:
                with self.assertRaises(ValueError):
                    q4_pair.qmv_fast_pair(*args)
                kernel.assert_not_called()

    def test_all_weights_validated_before_install(self):
        mutations = {
            "mode": "mxfp4",
            "bits": 8,
            "group_size": 32,
            "weight": Array((8, 32), "uint32"),
            "scales": Array((8, 8), "float32"),
            "biases": None,
            "bias": Array((7,)),
        }
        real_validation = q4_pair._validate_weights
        for key, value in mutations.items():
            model, good, params = self.fixture()
            model.bad = QuantizedLinear()
            model.bad[key] = value
            model.last_good = good  # Traverse this valid layer before bad.

            def validate(module, mx):
                self.assertIs(type(good), QuantizedLinear)
                return real_validation(module, mx)

            with (
                self.subTest(key=key),
                patch.object(q4_pair, "_validate_weights", validate),
            ):
                with self.assertRaisesRegex(ValueError, "'bad'"):
                    with q4_pair.paired_quantized_linears(model):
                        self.fail("Invalid weights entered context")
            self.assert_restored(good, params)
            self.assertIs(type(model.bad), QuantizedLinear)

    def test_subclass_and_nested_context_rejected(self):
        class Derived(QuantizedLinear):
            pass

        model, layer, params = self.fixture()
        model.custom = Derived()
        model.custom.child = Module()  # Intermediate subclasses must also be validated.
        with self.assertRaisesRegex(ValueError, "subclass"):
            with q4_pair.paired_quantized_linears(model):
                self.fail("Subclass was silently skipped")
        self.assert_restored(layer, params)
        del model["custom"]
        with q4_pair.paired_quantized_linears(model):
            paired_type = type(layer)
            with self.assertRaisesRegex(ValueError, "subclass"):
                with q4_pair.paired_quantized_linears(model):
                    self.fail("Nested context entered")
            self.assertIs(type(layer), paired_type)
        self.assert_restored(layer, params)

    def test_standalone_quantized_linear_is_included(self):
        _, layer, params = self.fixture()
        with patch.object(q4_pair, "qmv_fast_pair", return_value=Array((2, 8))) as pair:
            with q4_pair.paired_quantized_linears(layer) as report:
                layer(Array((2, 512)))
                self.assertEqual(report["patched_projection_count"], 1)
                pair.assert_called_once()
        self.assert_restored(layer, params, report)

    def test_pair_preserves_bias_shape_alias_and_parameter_identity(self):
        model, layer, params = self.fixture(bias=True)
        with patch.object(q4_pair, "qmv_fast_pair", return_value=Array((2, 8))) as pair:
            with q4_pair.paired_quantized_linears(model) as report:
                output = model.alias(Array((1, 2, 512)))
                self.assertEqual(output.shape, (1, 2, 8))
                self.assertEqual(output.values, list(range(8)) * 2)
                self.assertEqual(report["patched_projection_count"], 1)
                self.assertEqual(report["supported_projection_count"], 1)
                self.assertNotIn("pair_calls_by_projection", report)
                self.assertNotIn("fallback_calls_by_projection", report)
                for value, key in zip(
                    pair.call_args.args[1:], ("weight", "scales", "biases")
                ):
                    self.assertIs(value, params[key])
        self.assert_restored(layer, params, report)

    def test_stock_fallback_and_optional_counts(self):
        model, layer, params = self.fixture(bias=True)
        inputs = [
            Array((1, 512)),
            Array((3, 512)),
            Array((512,)),
            Array((2, 512), "float32"),
        ]
        expected = [layer(x).values for x in inputs]
        layer.stock_calls.clear()
        with patch.object(q4_pair, "qmv_fast_pair", return_value=Array((2, 8))) as pair:
            with q4_pair.paired_quantized_linears(model, count_calls=True) as report:
                for x, values in zip(inputs, expected):
                    self.assertEqual(layer(x).values, values)
                self.assertEqual(layer.stock_calls, inputs)
                pair.assert_not_called()
                layer(Array((2, 512)))
                model.alias(Array((2, 512)))
                self.assertEqual(report["pair_calls_by_projection"], {"alias": 2})
                self.assertEqual(report["fallback_calls_by_projection"], {"alias": 4})
        self.assert_restored(layer, params, report)

    def test_verification_scope_keeps_prefill_and_mtp_stock(self):
        for count_calls in (False, True):
            _, layer, params = self.fixture(bias=True)
            model = CheckpointModel(projection=layer)
            with self.subTest(count_calls=count_calls):
                with patch.object(
                    q4_pair, "qmv_fast_pair", return_value=Array((2, 8))
                ) as pair:
                    with q4_pair.paired_quantized_linears(
                        model, count_calls=count_calls, verification_only=True
                    ) as report:
                        expected = [7 + i for i in range(8)] * 2
                        self.assertEqual(model(Array((2, 512))).values, expected)
                        self.assertEqual(
                            model(Array((2, 512)), ssm_checkpoints=None).values,
                            expected,
                        )
                        self.assertEqual(
                            model.mtp_forward(Array((2, 512))).values, expected
                        )
                        pair.assert_not_called()
                        output = model(Array((2, 512)), ssm_checkpoints={})
                        self.assertEqual(output.values, list(range(8)) * 2)
                        pair.assert_called_once()
                        self.assertEqual(model(Array((1, 512))).values, expected[:8])
                        self.assertEqual(len(layer.stock_calls), 4)
                        self.assertTrue(report["verification_only"])
                        if count_calls:
                            self.assertEqual(
                                report["pair_calls_by_projection"], {"projection": 1}
                            )
                            self.assertEqual(
                                report["fallback_calls_by_projection"],
                                {"projection": 4},
                            )
                        else:
                            self.assertNotIn("pair_calls_by_projection", report)
                self.assertIs(type(model), CheckpointModel)
                self.assert_restored(layer, params, report)

    def test_verification_invalid_input_raises_without_fallback(self):
        _, layer, params = self.fixture()
        model = CheckpointModel(projection=layer)
        with patch.object(q4_pair, "qmv_fast_pair") as pair:
            with q4_pair.paired_quantized_linears(
                model, count_calls=True, verification_only=True
            ) as report:
                for x in (
                    Array((1, 512)),
                    Array((2, 512), "float32"),
                    Array((2, 256)),
                    Array((512,)),
                ):
                    with self.subTest(shape=x.shape, dtype=x.dtype):
                        with self.assertRaisesRegex(ValueError, "BF16 two-row"):
                            model(x, ssm_checkpoints={})
                pair.assert_not_called()
                self.assertEqual(layer.stock_calls, [])
                self.assertEqual(report["pair_calls_by_projection"], {})
                self.assertEqual(report["fallback_calls_by_projection"], {})
                layer(Array((2, 512)))  # Failure must reset the active scope.
                self.assertEqual(len(layer.stock_calls), 1)
        self.assertIs(type(model), CheckpointModel)
        self.assert_restored(layer, params, report)

    def test_verification_exceptions_reset_scope_and_restore_both_classes(self):
        for kernel_error in (False, True):
            _, layer, params = self.fixture()
            model = CheckpointModel(projection=layer)
            effect = RuntimeError("kernel failure") if kernel_error else None
            with self.subTest(kernel_error=kernel_error):
                with patch.object(
                    q4_pair,
                    "qmv_fast_pair",
                    return_value=Array((2, 8)),
                    side_effect=effect,
                ):
                    with self.assertRaisesRegex(RuntimeError, "body failure"):
                        with q4_pair.paired_quantized_linears(
                            model, verification_only=True
                        ) as report:
                            with self.assertRaisesRegex(RuntimeError, "failure"):
                                model(Array((2, 512)), ssm_checkpoints={}, fail=True)
                            self.assertEqual(layer(Array((2, 512))).values, [7] * 16)
                            raise RuntimeError("body failure")
                self.assertIs(type(model), CheckpointModel)
                self.assert_restored(layer, params, report)

    def test_verification_root_guard_precedes_any_mutation(self):
        model, layer, params = self.fixture()
        with self.assertRaisesRegex(ValueError, "root model"):
            with q4_pair.paired_quantized_linears(model, verification_only=True):
                self.fail("Unsupported root entered verification context")
        self.assertIs(type(model), Module)
        self.assert_restored(layer, params)

        model = CheckpointModel(projection=layer, bad=QuantizedLinear(k=256))
        with self.assertRaisesRegex(ValueError, "Unsupported QuantizedLinear"):
            with q4_pair.paired_quantized_linears(model, verification_only=True):
                self.fail("Unsupported weight entered verification context")
        self.assertIs(type(model), CheckpointModel)
        self.assert_restored(layer, params)

    def test_verification_nested_context_rejected_without_disturbing_outer(self):
        _, layer, params = self.fixture()
        model = CheckpointModel(projection=layer)
        with q4_pair.paired_quantized_linears(model, verification_only=True) as report:
            model_class, layer_class = type(model), type(layer)
            for verification_only in (False, True):
                with self.assertRaisesRegex(ValueError, "Nested"):
                    with q4_pair.paired_quantized_linears(
                        model, verification_only=verification_only
                    ):
                        self.fail("Nested context entered")
                self.assertIs(type(model), model_class)
                self.assertIs(type(layer), layer_class)
        self.assertIs(type(model), CheckpointModel)
        self.assert_restored(layer, params, report)

    def test_cleanup_on_body_and_kernel_exceptions(self):
        for kernel_error in (False, True):
            model, layer, params = self.fixture()
            with self.subTest(kernel_error=kernel_error):
                with patch.object(
                    q4_pair, "qmv_fast_pair", side_effect=RuntimeError("deliberate")
                ):
                    with self.assertRaisesRegex(RuntimeError, "deliberate"):
                        with q4_pair.paired_quantized_linears(model) as report:
                            if kernel_error:
                                layer(Array((2, 512)))
                            raise RuntimeError("deliberate")
                self.assert_restored(layer, params, report)

    def test_triple_shape_and_dtype_guards(self):
        self.assertEqual(
            q4_triple._validate_shapes((3, 512), (8, 64), (8, 8), (8, 8)), (8, 512)
        )
        invalid = [
            ((2, 512), (8, 64), (8, 8), (8, 8)),
            ((1, 3, 512), (8, 64), (8, 8), (8, 8)),
            ((3, 0), (8, 0), (8, 0), (8, 0)),
            ((3, 256), (8, 32), (8, 4), (8, 4)),
            ((3, 512), (7, 64), (7, 8), (7, 8)),
            ((3, 512), (8,), (8, 8), (8, 8)),
            ((3, 512), (8, 63), (8, 8), (8, 8)),
            ((3, 512), (8, 64), (8, 7), (8, 8)),
            ((3, 512), (8, 64), (8, 8), (8, 8, 1)),
        ]
        for shapes in invalid:
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                q4_triple._validate_shapes(*shapes)
        for index in range(4):
            layer = QuantizedLinear()
            args = [Array((3, 512)), layer.weight, layer.scales, layer.biases]
            args[index].dtype = "float32"
            with (
                self.subTest(index=index),
                patch.object(q4_triple, "_kernel") as kernel,
            ):
                with self.assertRaises(ValueError):
                    q4_triple.qmv_triple(*args)
                kernel.assert_not_called()

    def test_triple_launch_owns_every_output_once(self):
        for n, k in ((8, 512), (16, 1024), (17408, 5120), (5120, 17408)):
            # Metadata-only arguments keep this a small CPU test for real model shapes.
            class Metadata:
                def __init__(self, shape, dtype):
                    self.shape, self.dtype = shape, dtype

            x = Metadata((3, k), "bfloat16")
            parts = [
                Metadata((n, k // 8), "uint32"),
                Metadata((n, k // 64), "bfloat16"),
                Metadata((n, k // 64), "bfloat16"),
            ]
            output, launch = object(), Mock()
            launch.return_value = [output]
            with (
                self.subTest(n=n, k=k),
                patch.object(q4_triple, "_kernel", return_value=launch),
            ):
                self.assertIs(q4_triple.qmv_triple(x, *parts), output)
                call = launch.call_args.kwargs
                self.assertEqual(call["inputs"], [*parts, x])
                self.assertEqual(call["grid"], (32, n // 2, 1))
                self.assertEqual(call["threadgroup"], (32, 2, 1))
                self.assertEqual(call["output_shapes"], [(3, n)])
                self.assertEqual(
                    call["template"], [("T", "bfloat16"), ("K", k), ("N", n)]
                )
                groups = call["grid"][1] // call["threadgroup"][1]
                outputs = [
                    vector * n + group * 4 + simd * 2 + row
                    for vector in range(3)
                    for group in range(groups)
                    for simd in range(2)
                    for row in range(2)
                ]
                self.assertEqual(outputs, list(range(3 * n)))
        q4_triple._kernel()
        options = self.core.fast.metal_kernel.call_args.kwargs
        self.assertTrue(options["ensure_row_contiguous"])
        self.assertEqual(options["compile_options"], {"math_mode": "safe"})
        self.assertEqual(options["header"], q4_triple.HEADER)
        self.assertEqual(options["source"], q4_triple.SOURCE)

    def test_verification_rows_guards_precede_any_mutation(self):
        _, layer, params = self.fixture()
        model = CheckpointModel(projection=layer)
        for rows, scoped in (
            (1, True),
            (4, True),
            (True, True),
            (3.0, True),
            ("3", True),
            (3, False),
        ):
            with (
                self.subTest(rows=rows, scoped=scoped),
                self.assertRaisesRegex(ValueError, "verification_rows"),
            ):
                with q4_pair.paired_quantized_linears(
                    model, verification_rows=rows, verification_only=scoped
                ):
                    self.fail("Invalid row option entered context")
            self.assertIs(type(model), CheckpointModel)
            self.assert_restored(layer, params)
        unsupported_root = Module(projection=layer)
        with self.assertRaisesRegex(ValueError, "root model"):
            with q4_pair.paired_quantized_linears(
                unsupported_root, verification_only=True, verification_rows=3
            ):
                self.fail("Unsupported root accepted for triple verification")
        self.assert_restored(layer, params)

    def test_triple_scope_bias_counts_and_parameter_identity(self):
        for count_calls in (False, True):
            _, layer, params = self.fixture(bias=True)
            model = CheckpointModel(projection=layer, alias=layer)
            with (
                self.subTest(count_calls=count_calls),
                patch.object(
                    q4_triple, "qmv_triple", return_value=Array((3, 8))
                ) as triple,
                patch.object(q4_pair, "qmv_fast_pair") as pair,
            ):
                with q4_pair.paired_quantized_linears(
                    model,
                    count_calls=count_calls,
                    verification_only=True,
                    verification_rows=3,
                ) as report:
                    x = Array((1, 3, 512))
                    expected = [7 + i for i in range(8)] * 3
                    self.assertEqual(model(x).values, expected)
                    self.assertEqual(model(x, ssm_checkpoints=None).values, expected)
                    self.assertEqual(model.mtp_forward(x).values, expected)
                    triple.assert_not_called()
                    output = model(x, ssm_checkpoints={})
                    self.assertEqual(output.shape, (1, 3, 8))
                    self.assertEqual(output.values, list(range(8)) * 3)
                    self.assertEqual(triple.call_args.args[0].shape, (3, 512))
                    for value, key in zip(
                        triple.call_args.args[1:], ("weight", "scales", "biases")
                    ):
                        self.assertIs(value, params[key])
                    self.assertEqual(model(Array((1, 512))).values, expected[:8])
                    self.assertEqual(report["verification_rows"], 3)
                    self.assertEqual(report["patched_projection_count"], 1)
                    self.assertNotIn("pair_calls_by_projection", report)
                    if count_calls:
                        self.assertEqual(
                            report["triple_calls_by_projection"], {"alias": 1}
                        )
                        self.assertEqual(
                            report["fallback_calls_by_projection"], {"alias": 4}
                        )
                    else:
                        self.assertNotIn("triple_calls_by_projection", report)
                        self.assertNotIn("fallback_calls_by_projection", report)
                    pair.assert_not_called()
                self.assertIs(type(model), CheckpointModel)
                self.assert_restored(layer, params, report)

    def test_triple_invalid_active_inputs_do_not_fall_back(self):
        _, layer, params = self.fixture()
        model = CheckpointModel(projection=layer)
        with patch.object(q4_triple, "qmv_triple") as triple:
            with q4_pair.paired_quantized_linears(
                model, count_calls=True, verification_only=True, verification_rows=3
            ) as report:
                for x in (
                    Array((1, 512)),
                    Array((2, 512)),
                    Array((4, 512)),
                    Array((3, 512), "float32"),
                    Array((3, 256)),
                    Array((512,)),
                ):
                    with (
                        self.subTest(shape=x.shape, dtype=x.dtype),
                        self.assertRaisesRegex(ValueError, "BF16 three-row"),
                    ):
                        model(x, ssm_checkpoints={})
                triple.assert_not_called()
                self.assertEqual(layer.stock_calls, [])
                self.assertEqual(report["triple_calls_by_projection"], {})
                self.assertEqual(report["fallback_calls_by_projection"], {})
                layer(Array((3, 512)))
                self.assertEqual(len(layer.stock_calls), 1)
        self.assertIs(type(model), CheckpointModel)
        self.assert_restored(layer, params, report)

    def test_triple_weight_validation_and_cross_row_nested_context(self):
        _, layer, params = self.fixture()
        model = CheckpointModel(
            projection=layer, bad=QuantizedLinear(k=256), last_good=layer
        )
        real_validation = q4_pair._validate_weights

        def validate(module, mx):
            self.assertIs(type(layer), QuantizedLinear)
            self.assertIs(type(model), CheckpointModel)
            return real_validation(module, mx)

        with (
            patch.object(q4_pair, "_validate_weights", validate),
            self.assertRaisesRegex(ValueError, "'bad'"),
        ):
            with q4_pair.paired_quantized_linears(
                model, verification_only=True, verification_rows=3
            ):
                self.fail("Unsupported weight entered triple context")
        self.assert_restored(layer, params)
        del model["bad"]
        for outer_rows in (2, 3):
            with q4_pair.paired_quantized_linears(
                model, verification_only=True, verification_rows=outer_rows
            ) as report:
                classes = type(model), type(layer)
                for inner_rows in (2, 3):
                    with self.assertRaisesRegex(ValueError, "Nested"):
                        with q4_pair.paired_quantized_linears(
                            model, verification_only=True, verification_rows=inner_rows
                        ):
                            self.fail("Nested projection context entered")
                    self.assertEqual((type(model), type(layer)), classes)
            self.assertIs(type(model), CheckpointModel)
            self.assert_restored(layer, params, report)

    def test_triple_kernel_and_body_exceptions_restore_scope(self):
        for kernel_error in (False, True):
            _, layer, params = self.fixture()
            model = CheckpointModel(projection=layer)
            with (
                self.subTest(kernel_error=kernel_error),
                patch.object(
                    q4_triple,
                    "qmv_triple",
                    return_value=Array((3, 8)),
                    side_effect=RuntimeError("kernel failure")
                    if kernel_error
                    else None,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "body failure"):
                    with q4_pair.paired_quantized_linears(
                        model, verification_only=True, verification_rows=3
                    ) as report:
                        with self.assertRaisesRegex(RuntimeError, "failure"):
                            model(Array((3, 512)), ssm_checkpoints={}, fail=True)
                        self.assertEqual(layer(Array((3, 512))).values, [7] * 24)
                        raise RuntimeError("body failure")
                self.assertIs(type(model), CheckpointModel)
                self.assert_restored(layer, params, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
