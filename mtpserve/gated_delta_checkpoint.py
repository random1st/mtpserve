# SPDX-License-Identifier: MIT
# Derived from mlx-lm 0.31.3, mlx_lm/models/gated_delta.py (scalar, unmasked).
# Modification: expose the FP32 recurrent state after the first of two tokens.
# Copyright (c) 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Two-position, unmasked gated delta verification with a prefix checkpoint.

The checkpoint uses the same batched q/k/v as verification. It does not rerun
projections or mix a single-token recomputation with batched hidden states.
"""

from functools import lru_cache

import mlx.core as mx


_SOURCE = """
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;
    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;
    auto checkpoint = after_first + (n * Dv + dv_idx) * Dk;
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;
    for (int t = 0; t < T; ++t) {
      if (true) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * g_[hv_idx];
          kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);
        auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + k_[s_idx] * delta;
          out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y[dv_idx] = static_cast<InT>(out);
        }
      } else {
        y[dv_idx] = static_cast<InT>(0);
      }
      if (t == 0) {
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          checkpoint[s_idx] = static_cast<StT>(state[i]);
        }
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtpserve_gated_delta_checkpoint_t2_v1",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out", "after_first"],
        source=_SOURCE,
    )


def gated_delta_checkpoint(q, k, v, g, beta, state):
    """Return (y, final_state, state_after_first) for B=1, T=2.

    Supports FP32/BF16 activations, scalar FP32 decay, FP32 recurrent state,
    and no mask. Unsupported contracts raise before constructing a kernel.
    CPU uses the upstream ops reference and makes no Metal-kernel claim.
    """
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k must share [B,T,Hk,Dk]; v must be [B,T,Hv,Dv]")
    batch, length, hk, dk = q.shape
    hv, dv = v.shape[2:]
    if (
        (batch, length) != (1, 2)
        or v.shape[:2] != (1, 2)
        or min(hk, hv, dk, dv) <= 0
        or hv % hk
        or dk % 32
        or dv % 4
    ):
        raise ValueError("requires B=1,T=2,Hv%Hk=0,Dk%32=0,Dv%4=0,positive heads/dims")
    if g.shape != (1, 2, hv) or beta.shape != g.shape:
        raise ValueError("g/beta must be scalar gates shaped [1,2,Hv]")
    if state.shape != (1, hv, dv, dk):
        raise ValueError("state must be shaped [1,Hv,Dv,Dk]")
    if (
        q.dtype not in (mx.float32, mx.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or g.dtype != mx.float32
        or beta.dtype not in (q.dtype, mx.float32)
        or state.dtype != mx.float32
    ):
        raise TypeError(
            "requires matching FP32/BF16 q/k/v, FP32 g/state, FP32 or q-dtype beta"
        )
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        from mlx_lm.models.gated_delta import gated_delta_ops

        y, final = gated_delta_ops(q, k, v, g, beta, state)
        _, first = gated_delta_ops(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], state
        )
        return y, final, first
    return tuple(
        _kernel()(
            inputs=[q, k, v, g, beta, state, length],
            template=[
                ("InT", q.dtype),
                ("StT", state.dtype),
                ("Dk", dk),
                ("Dv", dv),
                ("Hk", hk),
                ("Hv", hv),
            ],
            grid=(32, dv, batch * hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(batch, length, hv, dv), state.shape, state.shape],
            output_dtypes=[q.dtype, state.dtype, state.dtype],
        )
    )
