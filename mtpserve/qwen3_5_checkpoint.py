# SPDX-License-Identifier: MIT
# Derived from mlx-lm 0.31.3, mlx_lm/models/qwen3_5.py: GatedDeltaNet.__call__.
# Modification: retain convolution and recurrent states after verification's P and, for T3, P,D1.
# Copyright (c) 2023 Apple Inc. (distribution license)
# Copyright (c) 2026 Apple Inc. (qwen3_5.py)
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
"""Restricted Qwen3.5 linear-attention verification with one or two prefix checkpoints."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g
from mlx_lm.models.qwen3_5 import GatedDeltaNet

from .gated_delta_checkpoint import gated_delta_checkpoint


def validate_checkpoint_contract(attn, input_shape, input_dtype, cache):
    """Validate supported metadata without constructing arrays or mutating cache."""
    if type(attn) is not GatedDeltaNet or type(cache) is not ArraysCache:
        raise TypeError("checkpoint requires stock GatedDeltaNet and ArraysCache")
    if attn.training or attn.sharding_group is not None:
        raise ValueError("checkpoint requires unsharded inference (eval mode)")
    if cache.lengths is not None or cache.left_padding is not None:
        raise ValueError("checkpoint does not support padding or per-sequence lengths")
    if tuple(input_shape) not in (
        (1, 2, attn.hidden_size),
        (1, 3, attn.hidden_size),
    ) or input_dtype not in (mx.float32, mx.bfloat16):
        raise ValueError(
            "checkpoint inputs must be FP32/BF16 [1,T,hidden_size], T=2 or T=3"
        )
    n_keep = attn.conv_kernel_size - 1
    if (
        n_keep < 1
        or attn.head_k_dim % 32
        or attn.head_v_dim % 4
        or min(attn.head_k_dim, attn.head_v_dim, attn.num_k_heads, attn.num_v_heads)
        <= 0
        or attn.num_v_heads % attn.num_k_heads
    ):
        raise ValueError("checkpoint requires conv_kernel>=2, Dk%32=0, Dv%4=0, Hv%Hk=0")
    if len(cache.state) != 2 or any(s is None for s in cache.state):
        raise ValueError(
            "checkpoint requires populated convolution and recurrent states"
        )
    conv_state, state = cache.state
    if (
        conv_state.shape != (1, n_keep, attn.conv_dim)
        or conv_state.dtype != input_dtype
        or state.shape != (1, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim)
        or state.dtype != mx.float32
    ):
        raise ValueError(
            "checkpoint cache shape/dtype differs from the attention contract"
        )


def linear_attention_checkpoint(attn, inputs, cache):
    """Return verification output and recurrent/convolution prefix checkpoints.

    Mutates the live cache to the full verification state, as GatedDeltaNet does.
    T2 returns [conv_P, state_P]; T3 returns {1: [conv_P, state_P],
    2: [conv_PD1, state_PD1]}. Supports populated, unpadded, unsharded
    inference caches and B=1 with T=2 or T=3.
    Contract failures raise before mutating cache; the caller selects fallback.
    """
    validate_checkpoint_contract(attn, inputs.shape, inputs.dtype, cache)
    n_keep = attn.conv_kernel_size - 1
    conv_state, state = cache.state
    batch, length, _ = inputs.shape
    qkv = attn.in_proj_qkv(inputs)
    z = attn.in_proj_z(inputs).reshape(batch, length, attn.num_v_heads, attn.head_v_dim)
    b = attn.in_proj_b(inputs)
    a = attn.in_proj_a(inputs)

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_after_last = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_after_first = mx.contiguous(conv_input[:, 1 : n_keep + 1, :])
    conv_after_second = (
        mx.contiguous(conv_input[:, 2 : n_keep + 2, :]) if length == 3 else None
    )
    conv_out = nn.silu(attn.conv1d(conv_input))
    q, k, v = [
        t.reshape(batch, length, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [attn.key_dim, 2 * attn.key_dim], -1),
            [attn.num_k_heads, attn.num_k_heads, attn.num_v_heads],
            [attn.head_k_dim, attn.head_k_dim, attn.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    beta = mx.sigmoid(b)
    g = compute_g(attn.A_log, a, attn.dt_bias)
    if length == 3:
        from .gated_delta_checkpoint3 import gated_delta_checkpoint3

        out, final_state, state_after_first, state_after_second = (
            gated_delta_checkpoint3(q, k, v, g, beta, state)
        )
        checkpoints = {
            1: [conv_after_first, state_after_first],
            2: [conv_after_second, state_after_second],
        }
    else:
        out, final_state, state_after_first = gated_delta_checkpoint(
            q, k, v, g, beta, state
        )
        checkpoints = [conv_after_first, state_after_first]

    # Delay committing the full cache until the kernel's contract checks pass.
    cache[0] = conv_after_last
    cache[1] = final_state
    cache.advance(length)
    out = attn.norm(out, z)
    out = attn.out_proj(out.reshape(batch, length, -1))
    return out, checkpoints
