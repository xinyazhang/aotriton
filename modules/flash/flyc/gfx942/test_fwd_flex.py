#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forward bodies on flex_gemm against the originals: fwd_flex.py vs fwd.py
(narrow), fwd_wide_flex.py vs fwd_wide.py (wide) and fwd_xwide_flex.py vs
fwd_xwide.py (xwide). O and LSE must be
bit-identical for every feature build, including windows, varlen, GQA, NaN
slack in padded heads and rows that see no key.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_fwd_flex.py
"""

import functools
import importlib

import pytest

from .jit_common import np, pytestmark, torch, with_nan_slack  # noqa: F401

import fmha_utils as abi  # noqa: E402

# body -> head dims: narrow at 64/128; wide at 160 (early K/V^T) and 256 (streamed);
# xwide at 384 and 512.
HEAD_DIMS = {'narrow': (64, 128), 'wide': (160, 256), 'xwide': (384, 512)}
BUILDERS = {'narrow': (('fwd', 'build_fmha_fwd_gfx942'), ('fwd_flex', 'build_fmha_fwd_gfx942_flex')),
            'wide': (('fwd_wide', 'build_fmha_fwd_gfx942_wide'),
                     ('fwd_wide_flex', 'build_fmha_fwd_gfx942_wide_flex')),
            'xwide': (('fwd_xwide', 'build_fmha_fwd_gfx942_xwide'),
                      ('fwd_xwide_flex', 'build_fmha_fwd_gfx942_xwide_flex'))}
CASES = [(body, d) for body, dims in HEAD_DIMS.items() for d in dims]


def _builder(body, variant):
    mod, fn = BUILDERS[body][0 if variant == 'old' else 1]
    return getattr(importlib.import_module(f'.{mod}', __package__), fn)


@functools.cache
def _build(variant, body, head_dim, causal_type, bias, dropout, padded_head, dtype_str, rescale_threshold):
    build = _builder(body, variant)
    return build(head_dim,
                 causal_type=causal_type,
                 bias=bias,
                 dropout=dropout,
                 padded_head=padded_head,
                 dtype_str=dtype_str,
                 rescale_threshold=rescale_threshold)


def _run_both(key, qkv, batch, seqlen_q, out, lse, kwargs):
    """Run old and new on the same inputs; return their (O, LSE)."""
    got = []
    for variant in ('old', 'new'):
        out.fill_(float('nan'))
        lse.fill_(float('nan'))
        _build(variant, *key)(*qkv, out, batch, seqlen_q, **kwargs)
        torch.cuda.synchronize()
        got.append((out.clone(), lse.clone()))
    return got


def _assert_same(got):
    (o0, l0), (o1, l1) = got
    # NaN-aware bitwise equality.
    assert torch.equal(o0.view(torch.int16), o1.view(torch.int16)), 'O differs'
    assert torch.equal(l0.view(torch.int32), l1.view(torch.int32)), 'LSE differs'


# Window cases: causal_type 1/2 (top-left/bottom-right causal), 3 with an explicit window.
WINDOWS = ((0, None), (1, None), (2, None), (3, (100, 37)), (3, (0, 0)))


@pytest.mark.parametrize("rescale_threshold", (0.0, 8.0))
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dropout", (False, True))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_dense(body, head_dim, window, bias, dropout, padded, dtype, rescale_threshold):
    causal_type, win = window
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    b, h, hk, sq, sk = 2, 4, 2, 333, 517
    d = head_dim - 5 if padded else head_dim     # NaN in the slack up to ceil8(d)
    torch.manual_seed(head_dim + causal_type)
    if padded:
        q = with_nan_slack(b, h, sq, d, dt=dt)
        k = with_nan_slack(b, hk, sk, d, dt=dt)
        v = with_nan_slack(b, hk, sk, d, dt=dt)
    else:
        q = torch.randn(b, h, sq, d, device='cuda', dtype=dt)
        k = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
        v = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
    out = with_nan_slack(b, h, sq, d, dt=dt) if padded else torch.empty_like(q)
    lse = torch.empty(b * h, sq, device='cuda')
    kwargs = dict(seqlen_k=sk, lse=lse)
    if bias:
        kwargs['bias'] = torch.randn(b, h, sq, sk, device='cuda', dtype=dt)
    if dropout:
        kwargs.update(dropout_p=0.2, philox_seed=1234, philox_offset2=5)
    if win is not None:
        kwargs['window'] = win
    key = (body, head_dim, causal_type, bias, dropout, padded, dtype, rescale_threshold)
    _assert_same(_run_both(key, (q, k, v), b, sq, out, lse, kwargs))


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("causal_type", (0, 2))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_varlen(body, head_dim, causal_type, bias):
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    h = 2
    lq = [a for a, _ in SEQS]
    lk = [b for _, b in SEQS]
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device='cuda', dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device='cuda', dtype=torch.int32)
    tq, tk = sum(lq), sum(lk)
    torch.manual_seed(head_dim)
    q = torch.randn(1, h, tq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, h, tk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=abi.VARLEN_LSE_LAYOUT_HT)
    out = torch.empty_like(q)
    lse = torch.empty(h, tq, device='cuda')
    kwargs = dict(seqlen_k=max(lk), num_seqlens=len(SEQS), lse=lse, varlen=varlen)
    if bias:
        kwargs['bias'] = torch.randn(1, h, tq, tk, device='cuda', dtype=torch.bfloat16)
    key = (body, head_dim, causal_type, bias, False, False, 'bf16', 0.0)
    _assert_same(_run_both(key, (q, k, v), 1, max(lq), out, lse, kwargs))
