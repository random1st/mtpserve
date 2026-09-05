# SPDX-License-Identifier: MIT
# MIT License
# Copyright © 2023 Apple Inc. (upstream file: Copyright © 2023-2024 Apple Inc.)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Experimental BF16/Q4/group64 projections for two input rows.

Derived from MLX v0.32.2, mlx/backend/metal/kernels/quantized.h:
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h
Upstream helpers: load_vector (62-69), qdot (235-243/289), qmv_fast_impl (758-821).
Source SHA256: 2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f.
Modification: share packed integer loads/masks between two rows while preserving
separate floating-point expression order and the two-SIMD-group launch layout.
Numerical equivalence depends on the MLX/compiler/device combination; this is
an opt-in experimental path, not a general replacement for quantized_matmul.

Importing this module does not import MLX. The first qmv_fast_pair call creates
a Metal kernel; evaluation and synchronization remain the caller's responsibility.
"""

from contextlib import contextmanager
from functools import lru_cache

HEADER = r"""
template <typename T>
inline float pair_load_vector(const device T* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < 16; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
    return sum;
}
inline float clone_qdot(const device uint8_t* w, const thread float* x_thread,
                        float scale, float bias, float sum) {
    float accum = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < 4; i++) {
        accum +=
            (x_thread[4 * i] * (ws[i] & 0x000f) +
             x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
             x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
             x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
    return scale * accum + sum * bias;
}
inline void pair_qdot(const device uint8_t* w,
                      const thread float* x0, const thread float* x1,
                      float scale, float bias, float sum0, float sum1,
                      thread float& q0, thread float& q1) {
    float accum0 = 0, accum1 = 0;
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < 4; i++) {
        uint16_t packed = ws[i];
        auto a = packed & 0x000f;
        auto b = packed & 0x00f0;
        auto c = packed & 0x0f00;
        auto d = packed & 0xf000;
        accum0 += (x0[4 * i] * a + x0[4 * i + 1] * b +
                   x0[4 * i + 2] * c + x0[4 * i + 3] * d);
        accum1 += (x1[4 * i] * a + x1[4 * i + 1] * b +
                   x1[4 * i + 2] * c + x1[4 * i + 3] * d);
    }
    q0 = scale * accum0 + sum0 * bias;
    q1 = scale * accum1 + sum1 * bias;
}
"""

SOURCE = r"""
uint lid = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
int out_row = threadgroup_position_in_grid.y * 8 + sg * 4;
const device uint8_t* ws = (const device uint8_t*)w + out_row * (K / 2) + lid * 8;
const device T* ss = scales + out_row * (K / 64) + lid / 4;
const device T* bb = biases + out_row * (K / 64) + lid / 4;
const device T* xp0 = x + lid * 16;
const device T* xp1 = xp0;
if constexpr (M == 2) xp1 += K;
thread float x0[16], x1[16];
thread float result0[4] = {0};
thread float result1[4] = {0};
for (int k = 0; k < K; k += 512) {
    float sum0 = pair_load_vector<T>(xp0, x0);
    float sum1 = 0;
    if constexpr (M == 2) sum1 = pair_load_vector<T>(xp1, x1);
    for (int row = 0; row < 4; row++) {
        auto wl = (const device uint8_t*)(ws + row * (K / 2));
        float scale = ss[row * (K / 64)];
        float bias = bb[row * (K / 64)];
        if constexpr (M == 1) {
            result0[row] += clone_qdot(wl, x0, scale, bias, sum0);
        } else {
            float q0, q1;
            pair_qdot(wl, x0, x1, scale, bias, sum0, sum1, q0, q1);
            result0[row] += q0;
            result1[row] += q1;
        }
    }
    ws += 256;
    ss += 8;
    bb += 8;
    xp0 += 512;
    if constexpr (M == 2) xp1 += 512;
}
for (int row = 0; row < 4; row++) {
    result0[row] = simd_sum(result0[row]);
    if (lid == 0) y[out_row + row] = static_cast<T>(result0[row]);
    if constexpr (M == 2) {
        result1[row] = simd_sum(result1[row]);
        if (lid == 0) y[N + out_row + row] = static_cast<T>(result1[row]);
    }
}
"""


def _validate_shapes(x, w, scales, biases):
    """Validate packed affine Q4 layout without importing MLX."""
    if len(x) != 2 or x[0] != 2:
        raise ValueError("Expected rank-2 input with exactly two rows")
    k = x[1]
    if len(w) != 2 or k <= 0 or k % 512 or w[0] <= 0 or w[0] % 8:
        raise ValueError(
            "Q4 pair requires positive K divisible by 512 and N divisible by 8"
        )
    n = w[0]
    if w != (n, k // 8) or scales != (n, k // 64) or biases != (n, k // 64):
        raise ValueError(
            "Expected packed Q4 weight (N,K/8) and group64 metadata (N,K/64)"
        )
    return n, k


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="qmv_fast_pair_experimental",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        source=SOURCE,
        header=HEADER,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def qmv_fast_pair(x, w, scales, biases):
    """Project BF16 (2,K) through packed Q4/group64 weights using fixed G2.

    K must be divisible by 512 and N by 8. Weight dtype is uint32; affine
    scales/biases are BF16. Noncontiguous arrays are normalized by metal_kernel;
    no implicit dtype conversion or device synchronization is performed here.
    """
    import mlx.core as mx

    n, k = _validate_shapes(x.shape, w.shape, scales.shape, biases.shape)
    if (
        x.dtype != mx.bfloat16
        or scales.dtype != mx.bfloat16
        or biases.dtype != mx.bfloat16
    ):
        raise ValueError(
            "Only BF16 input/scales/biases are supported; no implicit dtype casts"
        )
    if w.dtype != mx.uint32:
        raise ValueError("Packed Q4 weight must be uint32")
    return _kernel()(
        inputs=[w, scales, biases, x],
        template=[("T", mx.bfloat16), ("K", k), ("N", n), ("M", 2)],
        grid=(32, 2 * (n // 8), 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(2, n)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _validate_weights(module, mx):
    if (
        module.mode != "affine"
        or module.bits != 4
        or module.group_size != 64
        or module.weight.dtype != mx.uint32
        or module.scales.dtype != mx.bfloat16
        or module.get("biases") is None
        or module.biases.dtype != mx.bfloat16
    ):
        raise ValueError(
            "Requires affine Q4/group64 uint32 weights and BF16 scales/biases"
        )
    weight_shape = module.weight.shape
    if len(weight_shape) != 2:
        raise ValueError("Packed Q4 weight must have rank 2")
    n, _ = _validate_shapes(
        (2, weight_shape[1] * 8), weight_shape, module.scales.shape, module.biases.shape
    )
    if "bias" in module and module.bias.shape != (n,):
        raise ValueError("Linear bias must have shape (N,)")


@contextmanager
def paired_quantized_linears(
    model, *, count_calls=False, verification_only=False, verification_rows=2
):
    """Temporarily replace every stock QuantizedLinear's selected projection.

    Validate all observed QuantizedLinear weights before mutating any instance.
    Unsupported weights or subclasses raise ValueError, preventing a silently
    mixed projection path. Other input sizes/dtypes use the stock implementation.
    Classes are restored on exit without copying/replacing any parameter arrays.

    With verification_only=True, the root model must support SSM checkpoints.
    Dispatch is active only inside root calls with a non-None ssm_checkpoints
    keyword; prefill and standalone MTP calls remain stock. verification_rows=3
    selects the R2G2 triple kernel and requires verification_only=True. The
    default remains two rows. Active verification requires BF16 inputs with the
    selected row count at every projection, raising on incompatible inputs.

    The yielded report contains static projection counts and cleanup results.
    Set count_calls=True for per-projection pair/triple and fallback counts; the default
    dispatch performs no diagnostic counter or dictionary operations. Counting
    adds Python overhead and should be disabled for timing comparisons.

    The model must not be used concurrently, re-quantized, or have its parameters
    replaced within the context. Nested contexts on the same model are rejected.
    """
    if type(verification_rows) is not int or verification_rows not in (2, 3):
        raise ValueError("verification_rows must be integer 2 or 3")
    if verification_rows == 3 and not verification_only:
        raise ValueError("verification_rows=3 requires verification_only=True")

    import mlx.core as mx
    import mlx.nn as nn

    project = qmv_fast_pair
    counter_key, row_description = "pair_calls_by_projection", "two-row"
    if verification_rows == 3:
        from .q4_triple import qmv_triple

        project = qmv_triple
        counter_key, row_description = "triple_calls_by_projection", "three-row"

    original_call = nn.QuantizedLinear.__call__
    model_class = type(model)
    if getattr(model_class, "_q4_pair_verification_context", False):
        raise ValueError(
            "Nested paired projection contexts on the same model are unsupported"
        )
    if verification_only and (
        not getattr(model, "supports_ssm_checkpoint", False)
        or isinstance(model, nn.QuantizedLinear)
    ):
        raise ValueError(
            "verification_only requires a root model supporting SSM checkpoints"
        )
    if verification_only:
        original_model_call = model_class.__call__
    active = False
    modules = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear) or id(module) in modules:
            continue
        if type(module) is not nn.QuantizedLinear:
            raise ValueError(f"Unsupported QuantizedLinear subclass at {name!r}")
        try:
            _validate_weights(module, mx)
        except (ValueError, AttributeError, KeyError) as error:
            raise ValueError(
                f"Unsupported QuantizedLinear at {name!r}: {error}"
            ) from error
        modules[id(module)] = (name, module)

    report = {
        "patched_projection_count": len(modules),
        "supported_projection_count": len(modules),
        "verification_only": bool(verification_only),
        "verification_rows": verification_rows,
    }

    class PairQuantizedLinear(nn.QuantizedLinear):
        def __call__(self, x):
            if verification_only and not active:
                return original_call(self, x)
            if (
                x.dtype != mx.bfloat16
                or x.ndim < 2
                or x.size != verification_rows * x.shape[-1]
                or x.shape[-1] != self.weight.shape[-1] * 8
            ):
                if verification_only:
                    raise ValueError(
                        f"Checkpoint verification requires BF16 {row_description} projection inputs"
                    )
                return original_call(self, x)
            result = project(
                x.reshape(verification_rows, x.shape[-1]),
                self.weight,
                self.scales,
                self.biases,
            )
            result = result.reshape(*x.shape[:-1], self.weight.shape[0])
            return result + self["bias"] if "bias" in self else result

    replacement = PairQuantizedLinear
    if count_calls:
        report[counter_key] = {}
        report["fallback_calls_by_projection"] = {}

        class CountingPairQuantizedLinear(PairQuantizedLinear):
            def __call__(self, x):
                use_projection = (
                    (not verification_only or active)
                    and x.dtype == mx.bfloat16
                    and x.ndim >= 2
                    and x.size == verification_rows * x.shape[-1]
                    and x.shape[-1] == self.weight.shape[-1] * 8
                )
                result = super().__call__(x)
                name = modules[id(self)][0]
                counts = report[
                    counter_key if use_projection else "fallback_calls_by_projection"
                ]
                counts[name] = counts.get(name, 0) + 1
                return result

        replacement = CountingPairQuantizedLinear

    if verification_only:

        class VerificationModel(model_class):
            _q4_pair_verification_context = True

            def __call__(self, *args, **kwargs):
                nonlocal active
                previous_active = active
                active = kwargs.get("ssm_checkpoints") is not None
                try:
                    return original_model_call(self, *args, **kwargs)
                finally:
                    active = previous_active

    originals = []
    try:
        for _, module in modules.values():
            params = {
                key: module[key]
                for key in ("weight", "scales", "biases", "bias")
                if key in module
            }
            originals.append((module, type(module), params))
            module.__class__ = replacement
        if verification_only:
            model.__class__ = VerificationModel
        yield report
    finally:
        for module, cls, _ in originals:
            module.__class__ = cls
        if verification_only:
            model.__class__ = model_class
        report["model_class_restored"] = type(model) is model_class
        report["classes_restored"] = all(
            type(module) is cls for module, cls, _ in originals
        )
        report["parameter_objects_unchanged"] = all(
            module.get(key) is value
            for module, _, params in originals
            for key, value in params.items()
        )
        if (
            not report["classes_restored"]
            or not report["model_class_restored"]
            or not report["parameter_objects_unchanged"]
        ):
            raise AssertionError("Adapter cleanup failed or a parameter object changed")
