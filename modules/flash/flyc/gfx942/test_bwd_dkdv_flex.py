#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dK/dV bodies on flex_gemm against the originals: dkdv<body>_flex.py vs
dkdv<body>.py. dK and dV must be bit-identical for every feature build, including
windows, varlen, GQA, NaN slack in padded heads and keys no query sees; and both
must write every element they own (dK = dV = 0 on keys no query sees).

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_bwd_dkdv_flex.py
"""

import functools
import importlib
import math

import pytest

from .jit_common import assert_written, np, pytestmark, torch, with_nan_slack  # noqa: F401

import fmha_utils as abi  # noqa: E402

# body -> head dims it serves (hd96: waves_per_eu 2 without bias, 1 with).
HEAD_DIMS = {'narrow': (32, 64, 96, 128)}
SUFFIX = {'narrow': '', 'wide': '_wide', 'xwide': '_xwide', 'xxwide': '_xxwide'}
BUILDERS = {b: ((f'dkdv{s}', f'build_fmha_bwd_dkdv_gfx942{s}'), (f'dkdv{s}_flex', f'build_fmha_bwd_dkdv_gfx942{s}_flex'))
            for b, s in SUFFIX.items()}
CASES = [(body, d) for body, dims in HEAD_DIMS.items() for d in dims]


def _builder(body, variant):
    mod, fn = BUILDERS[body][0 if variant == 'old' else 1]
    return getattr(importlib.import_module(f'.{mod}', __package__), fn)


@functools.cache
def _build(variant, body, head_dim, causal_type, bias, dropout, padded_head, dtype_str):
    return _builder(body, variant)(head_dim,
                                   causal_type=causal_type,
                                   bias=bias,
                                   dropout=dropout,
                                   padded_head=padded_head,
                                   dtype_str=dtype_str)


def _run_both(key, args, dk, dv, kwargs):
    """Run old and new on the same inputs; return their (dK, dV)."""
    got = []
    for variant in ('old', 'new'):
        dk.fill_(float('nan'))
        dv.fill_(float('nan'))
        _build(variant, *key)(*args, **kwargs)
        torch.cuda.synchronize()
        assert_written(f'{variant} body', dK=dk, dV=dv)
        got.append((dk.clone(), dv.clone()))
    return got


def _assert_same(got):
    (k0, v0), (k1, v1) = got
    # NaN-aware bitwise equality.
    assert torch.equal(k0.view(torch.int16), k1.view(torch.int16)), 'dK differs'
    assert torch.equal(v0.view(torch.int16), v1.view(torch.int16)), 'dV differs'


def _rows(n_rows, sq, sk):
    """LSE near log(sk) and a small Delta: P stays in range, as from a real forward."""
    lse = math.log(sk) + 0.3 * torch.randn(n_rows, sq, device='cuda')
    delta = 0.1 * torch.randn(n_rows, sq, device='cuda')
    return lse, delta


# Window cases: causal_type 1/2 (top-left/bottom-right causal), 3 with an explicit window.
WINDOWS = ((0, None), (1, None), (2, None), (3, (100, 37)), (3, (0, 0)))
# Sq < Sk: top-left causal and small windows leave keys no query sees; Sq > Sk:
# bottom-right causal leaves rows that see no key.
SHAPES = ((333, 517), (517, 333))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dropout", (False, True))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_dense(body, head_dim, window, bias, dropout, padded, dtype, shape):
    causal_type, win = window
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    if shape != SHAPES[0] and (not causal_type or dtype != 'bf16'):
        pytest.skip("Sq > Sk only matters to masked rows")
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    b, h, hk = 2, 4, 2
    sq, sk = shape
    d = head_dim - 5 if padded else head_dim     # NaN in the slack up to ceil8(d)
    torch.manual_seed(head_dim + causal_type)
    if padded:
        q = with_nan_slack(b, h, sq, d, dt=dt)
        k = with_nan_slack(b, hk, sk, d, dt=dt)
        v = with_nan_slack(b, hk, sk, d, dt=dt)
        do = with_nan_slack(b, h, sq, d, dt=dt)
    else:
        q = torch.randn(b, h, sq, d, device='cuda', dtype=dt)
        k = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
        v = torch.randn(b, hk, sk, d, device='cuda', dtype=dt)
        do = torch.randn(b, h, sq, d, device='cuda', dtype=dt)
    dk = with_nan_slack(b, hk, sk, d, dt=dt) if padded else torch.empty_like(k)
    dv = with_nan_slack(b, hk, sk, d, dt=dt) if padded else torch.empty_like(v)
    lse, delta = _rows(b * h, sq, sk)
    kwargs = dict(seqlen_k=sk)
    if bias:
        kwargs['bias'] = torch.randn(b, h, sq, sk, device='cuda', dtype=dt)
    if dropout:
        kwargs.update(dropout_p=0.2, philox_seed=1234, philox_offset2=5)
    if win is not None:
        kwargs['window'] = win
    key = (body, head_dim, causal_type, bias, dropout, padded, dtype)
    _assert_same(_run_both(key, (q, k, v, do, dk, dv, lse, delta, b, sq), dk, dv, kwargs))


# (seqlen_q, seqlen_k) per sequence: decode, shorter and longer K, ragged tiles.
SEQS = ((1, 300), (64, 320), (130, 130), (300, 37), (37, 200))


@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("causal_type", (0, 2))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_varlen(body, head_dim, causal_type, bias):
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    h, hk = 4, 2
    lq = [a for a, _ in SEQS]
    lk = [b for _, b in SEQS]
    cu_q = torch.tensor([0] + list(np.cumsum(lq)), device='cuda', dtype=torch.int32)
    cu_k = torch.tensor([0] + list(np.cumsum(lk)), device='cuda', dtype=torch.int32)
    tq, tk = sum(lq), sum(lk)
    torch.manual_seed(head_dim)
    q = torch.randn(1, h, tq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, hk, tk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=tq, lse_layout=abi.VARLEN_LSE_LAYOUT_HT)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    lse, delta = _rows(h, tq, max(lk))
    kwargs = dict(seqlen_k=max(lk), num_seqlens=len(SEQS), varlen=varlen)
    if bias:
        kwargs['bias'] = torch.randn(1, h, tq, tk, device='cuda', dtype=torch.bfloat16)
    key = (body, head_dim, causal_type, bias, False, False, 'bf16')
    _assert_same(_run_both(key, (q, k, v, do, dk, dv, lse, delta, 1, max(lq)), dk, dv, kwargs))
