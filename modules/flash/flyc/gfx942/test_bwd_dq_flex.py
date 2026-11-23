#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The dQ bodies on flex_gemm against the originals: dq_flex.py vs dq.py (narrow)
and dq_wide_flex.py vs dq_wide.py (wide).
dQ and dB must be bit-identical for every feature build, including windows,
varlen, GQA, NaN slack in padded heads and rows that see no key; and both must
write every element they own (dQ = 0 on rows that see no key).

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_bwd_dq_flex.py
"""

import functools
import importlib
import math

import pytest

from .jit_common import assert_written, np, pytestmark, torch, with_nan_slack  # noqa: F401

import fmha_utils as abi  # noqa: E402

# body -> head dims; the narrow body's dQ groups differ at each (1, 2, 2+1, 4 columns).
# The wide body: 32x32x8 at 160/256, 16x16x16 at 384/512.
HEAD_DIMS = {'narrow': (32, 64, 96, 128), 'wide': (160, 256, 384, 512)}
BUILDERS = {'narrow': (('dq', 'build_fmha_bwd_dq_gfx942'), ('dq_flex', 'build_fmha_bwd_dq_gfx942_flex')),
            'wide': (('dq_wide', 'build_fmha_bwd_dq_gfx942_wide'),
                     ('dq_wide_flex', 'build_fmha_bwd_dq_gfx942_wide_flex'))}
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


def _run_both(key, args, dq, dbias, kwargs, db_owned=None):
    """Run old and new on the same inputs; return their (dQ, dB). `db_owned` lists
    the (row slice, column slice) blocks of dB a varlen launch owns."""
    got = []
    for variant in ('old', 'new'):
        dq.fill_(float('nan'))
        if dbias is not None:
            dbias.fill_(float('nan'))
        _build(variant, *key)(*args, **kwargs)
        torch.cuda.synchronize()
        assert_written(f'{variant} body', dQ=dq)
        if dbias is not None:
            for rows, cols in db_owned or [(slice(None), slice(None))]:
                assert_written(f'{variant} body', dB=dbias[..., rows, cols])
        got.append((dq.clone(), None if dbias is None else dbias.clone()))
    return got


def _assert_same(got):
    (q0, b0), (q1, b1) = got
    # NaN-aware bitwise equality.
    assert torch.equal(q0.view(torch.int16), q1.view(torch.int16)), 'dQ differs'
    if b0 is not None:
        assert torch.equal(b0.view(torch.int16), b1.view(torch.int16)), 'dB differs'


def _rows(n_rows, sq, sk):
    """LSE near log(sk) and a small Delta: P stays in range, as from a real forward."""
    lse = math.log(sk) + 0.3 * torch.randn(n_rows, sq, device='cuda')
    delta = 0.1 * torch.randn(n_rows, sq, device='cuda')
    return lse, delta


# Window cases: causal_type 1/2 (top-left/bottom-right causal), 3 with an explicit window.
WINDOWS = ((0, None), (1, None), (2, None), (3, (100, 37)), (3, (0, 0)))
# Odd Sk: bias and dB rows start mid-dword; even Sk: clean dB tiles store dwords.
# Sq > Sk: under causal masks some rows see no key.
SHAPES = ((333, 517), (300, 520), (517, 333))


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
    if shape == SHAPES[1] and not bias:
        pytest.skip("the even pitch only matters to dB")
    if shape == SHAPES[2] and (not causal_type or dtype != 'bf16'):
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
    dq = with_nan_slack(b, h, sq, d, dt=dt) if padded else torch.empty_like(q)
    lse, delta = _rows(b * h, sq, sk)
    kwargs = dict(seqlen_k=sk)
    dbias = None
    if bias:
        kwargs['bias'] = torch.randn(b, h, sq, sk, device='cuda', dtype=dt)
        dbias = torch.empty(b, h, sq, sk, device='cuda', dtype=dt)
        kwargs['dbias'] = dbias
    if dropout:
        kwargs.update(dropout_p=0.2, philox_seed=1234, philox_offset2=5)
    if win is not None:
        kwargs['window'] = win
    key = (body, head_dim, causal_type, bias, dropout, padded, dtype)
    _assert_same(_run_both(key, (q, k, v, do, dq, lse, delta, b, sq), dq, dbias, kwargs))


# principal_diagonal_choice (VarlenBits bit 18): with d = seqlen_k - seqlen_q per
# sequence and p the bit, a numeric bound becomes L = wl - p*d, R = wr + p*d;
# sentinels are not shifted. The windows put L below 0, R past seqlen_k, a band off
# the diagonal and bands that miss every key of some rows (tests/
# test_bwd_dq_principal_diagonal.py); the shapes are decode, chunked prefill,
# square and Sq > Sk, dense and mixed in one varlen batch.
PD_WINDOWS = ((abi.WINDOW_BOTRIGHT, 0), (8, 0), (0, 8), (4, 4), (-1, 3))
PD_SHAPES = ((1, 200), (48, 200), (150, 150), (200, 70), 'varlen')
PD_SEQS = ((1, 130), (37, 160), (96, 96), (150, 40), (20, 3))
PD_BITS = (abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT, abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)


def _pd_varlen(shape, p_bit):
    """(batch, lq, lk, varlen dict, cu_q, cu_k) of a principal-diagonal case."""
    if shape == 'varlen':
        lq = [a for a, _ in PD_SEQS]
        lk = [b for _, b in PD_SEQS]
        cu_q = torch.tensor([0] + list(np.cumsum(lq)), device='cuda', dtype=torch.int32)
        cu_k = torch.tensor([0] + list(np.cumsum(lk)), device='cuda', dtype=torch.int32)
        varlen = abi.varlen_compact(cu_q, cu_k, max(lq), max(lk), lse_tokens=sum(lq), lse_layout=abi.VARLEN_LSE_LAYOUT_HT)
        varlen['bits'] = abi.principal_diagonal_bits(varlen['bits'], p_bit)
        return 1, lq, lk, varlen, cu_q, cu_k
    sq, sk = shape
    return 2, [sq], [sk], abi.varlen_dense(principal_diagonal=p_bit), None, None


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
    dq = torch.empty_like(q)
    lse, delta = _rows(h, tq, max(lk))
    kwargs = dict(seqlen_k=max(lk), num_seqlens=len(SEQS), varlen=varlen)
    dbias = None
    if bias:
        kwargs['bias'] = torch.randn(1, h, tq, tk, device='cuda', dtype=torch.bfloat16)
        dbias = torch.empty(1, h, tq, tk, device='cuda', dtype=torch.bfloat16)
        kwargs['dbias'] = dbias
    key = (body, head_dim, causal_type, bias, False, False, 'bf16')
    owned = [(slice(int(cu_q[i]), int(cu_q[i + 1])), slice(int(cu_k[i]), int(cu_k[i + 1]))) for i in range(len(SEQS))]
    _assert_same(_run_both(key, (q, k, v, do, dq, lse, delta, 1, max(lq)), dq, dbias, kwargs, owned))


@pytest.mark.parametrize("shape", PD_SHAPES)
@pytest.mark.parametrize("window", PD_WINDOWS)
@pytest.mark.parametrize("p_bit", PD_BITS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_principal_diagonal(body, head_dim, p_bit, window, shape):
    """Numeric windows under both principal diagonals: bit-identical dQ, and every
    owned element written (dQ = 0 where a row sees no key)."""
    batch, lq, lk, varlen, _, _ = _pd_varlen(shape, p_bit)
    h, hk = 4, 2
    torch.manual_seed(head_dim + sum(lq))
    tq, tk = (sum(lq), sum(lk)) if shape == 'varlen' else (lq[0], lk[0])
    q = torch.randn(batch, h, tq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch, hk, tk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dq = torch.empty_like(q)
    lse, delta = _rows(batch * h, tq, max(lk))
    kwargs = dict(seqlen_k=max(lk), varlen=varlen, window=window)
    if shape == 'varlen':
        kwargs['num_seqlens'] = len(lq)
    key = (body, head_dim, 3, False, False, False, 'bf16')
    _assert_same(_run_both(key, (q, k, v, do, dq, lse, delta, batch, max(lq)), dq, None, kwargs))
