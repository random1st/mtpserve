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

"""Experimental BF16/Q4/group64 projection for three input rows.

Derived from MLX v0.32.2, mlx/backend/metal/kernels/quantized.h:
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized.h
Upstream helpers: load_vector (62-69), qdot (235-243/289), qmv_fast_impl (758-821).
Source SHA256: 2a007016da606afe569adb9adcc05e00f14558ad2e094bcb4f8974beb53c316f.
Modification: share scalar uint16 packed loads/masks across three independent
vectors, preserving each M1 floating-point expression and SIMD reduction. Each
SIMD computes two output rows, with two SIMD groups per threadgroup (R2G2).
Numerical parity depends on the MLX/compiler/device combination; this remains an
opt-in experiment. Import does not load MLX; evaluation belongs to the caller.
"""

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

inline void triple_qdot(const device uint8_t* w,
                        const thread float* x0, const thread float* x1,
                        const thread float* x2,
                        float scale, float bias, float sum0, float sum1, float sum2,
                        thread float& q0, thread float& q1, thread float& q2) {
    float accum0 = 0, accum1 = 0, accum2 = 0;
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
        accum2 += (x2[4 * i] * a + x2[4 * i + 1] * b +
                   x2[4 * i + 2] * c + x2[4 * i + 3] * d);
    }
    q0 = scale * accum0 + sum0 * bias;
    q1 = scale * accum1 + sum1 * bias;
    q2 = scale * accum2 + sum2 * bias;
}
"""

SOURCE = r"""
uint lid = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
int out_row = threadgroup_position_in_grid.y * 4 + sg * 2;
const device uint8_t* ws = (const device uint8_t*)w + out_row * (K / 2) + lid * 8;
const device T* ss = scales + out_row * (K / 64) + lid / 4;
const device T* bb = biases + out_row * (K / 64) + lid / 4;
const device T* xp0 = x + lid * 16;
const device T* xp1 = xp0 + K;
const device T* xp2 = xp0 + 2 * K;
thread float x0[16], x1[16], x2[16];
thread float result0[2] = {0};
thread float result1[2] = {0};
thread float result2[2] = {0};
for (int k = 0; k < K; k += 512) {
    float sum0 = pair_load_vector<T>(xp0, x0);
    float sum1 = pair_load_vector<T>(xp1, x1);
    float sum2 = pair_load_vector<T>(xp2, x2);
    for (int row = 0; row < 2; row++) {
        auto wl = (const device uint8_t*)(ws + row * (K / 2));
        float scale = ss[row * (K / 64)];
        float bias = bb[row * (K / 64)];
        float q0, q1, q2;
        triple_qdot(wl, x0, x1, x2, scale, bias, sum0, sum1, sum2, q0, q1, q2);
        result0[row] += q0;
        result1[row] += q1;
        result2[row] += q2;
    }
    ws += 256;
    ss += 8;
    bb += 8;
    xp0 += 512;
    xp1 += 512;
    xp2 += 512;
}
for (int row = 0; row < 2; row++) {
    result0[row] = simd_sum(result0[row]);
    if (lid == 0) y[out_row + row] = static_cast<T>(result0[row]);
    result1[row] = simd_sum(result1[row]);
    if (lid == 0) y[N + out_row + row] = static_cast<T>(result1[row]);
    result2[row] = simd_sum(result2[row]);
    if (lid == 0) y[2 * N + out_row + row] = static_cast<T>(result2[row]);
}
"""


def _validate_shapes(x, w, scales, biases):
    """Validate the fixed M3 affine Q4/group64 geometry without importing MLX."""
    if len(x) != 2 or x[0] != 3:
        raise ValueError("Expected rank-2 input with exactly three rows")
    k = x[1]
    if len(w) != 2 or k <= 0 or k % 512 or w[0] <= 0 or w[0] % 8:
        raise ValueError(
            "Q4 triple requires positive K divisible by 512 and N divisible by 8"
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
        name="qmv_triple_rows2_experimental",
        input_names=["w", "scales", "biases", "x"],
        output_names=["y"],
        source=SOURCE,
        header=HEADER,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def qmv_triple(x, w, scales, biases):
    """Project BF16 (3,K) through packed uint32 Q4/group64 weights with R2G2.

    K must be divisible by 512, N by 8, and scales/biases must be BF16.
    The kernel normalizes noncontiguous rows; this function neither casts input
    dtypes nor evaluates/synchronizes the result.
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
        template=[("T", mx.bfloat16), ("K", k), ("N", n)],
        grid=(32, n // 2, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(3, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
