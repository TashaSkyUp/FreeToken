"""Fixed-tile half-precision linear with FP32 accumulation."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(
    X, W, B, Y, M,
    N: tl.constexpr, K: tl.constexpr,
    SXM: tl.constexpr, SXK: tl.constexpr,
    SWN: tl.constexpr, SWK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACC_FP64: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float64 if ACC_FP64 else tl.float32)
    for start in range(tl.cdiv(K, BK)):
        ks = start * BK + kk
        x = tl.load(
            X + rows[:, None] * SXM + ks[None, :] * SXK,
            mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0,
        )
        w = tl.load(
            W + cols[None, :] * SWN + ks[:, None] * SWK,
            mask=(cols[None, :] < N) & (ks[:, None] < K), other=0.0,
        )
        if ACC_FP64:
            # Reset each fixed K tile so splitting K preserves the dot products.
            acc += tl.dot(x, w).to(tl.float64)
        else:
            acc = tl.dot(x, w, acc)
    if HAS_BIAS:
        acc += tl.load(B + cols, mask=cols < N, other=0.0)[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :], acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def linear(x, weight, bias=None, *, out_dtype=None, accumulate_fp64=False):
    assert x.ndim == 2 and weight.ndim == 2
    assert x.dtype == weight.dtype and x.dtype in (torch.bfloat16, torch.float16)
    m, k = x.shape
    n = weight.shape[0]
    assert weight.shape[1] == k
    out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    # A fixed accumulation layout avoids algorithm changes at cold/cached batch sizes.
    _linear[(triton.cdiv(m, 16), triton.cdiv(n, 64))](
        x, weight, bias if bias is not None else weight, out, m, n, k,
        *x.stride(), *weight.stride(), bias is not None, accumulate_fp64, 16, 64, 64,
        num_warps=4, num_stages=2,
    )
    return out


def fp32_linear(x, weight, bias=None):
    if os.environ.get("FREETOKEN_STABLE_LINEAR"):
        return linear(x, weight, bias, out_dtype=torch.float32)
    y = torch.mm(x, weight.T, out_dtype=torch.float32)
    return y + bias.float() if bias is not None else y


def reduction_linear(x, weight, bias=None):
    if os.environ.get("FREETOKEN_TP_REDUCTION_FP64"):
        return linear(x, weight, bias, out_dtype=torch.float64, accumulate_fp64=True)
    return fp32_linear(x, weight, bias)
