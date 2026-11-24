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

from .jit_common import assert_written, bottom_right, keep_mask, np, philox_4x32_10, pytestmark, torch, with_nan_slack  # noqa: F401
from . import test_bwd_dkdv as narrow_ref
from . import test_bwd_dkdv_wide as wide_ref

import fmha_utils as abi  # noqa: E402
import philox  # noqa: E402

# body -> head dims it serves, with its policy's knobs (tuning_dkdv.policy): narrow
# hd96 waves_per_eu 2 without bias, 1 with; wide kv_subs 2; xwide kv_subs 4;
# xxwide kv_subs 4 at 384, 3 at 512.
HEAD_DIMS = {'narrow': (32, 64, 96, 128), 'wide': (160, 192), 'xwide': (224, 256), 'xxwide': (384, 512)}
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
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    lse, delta = _rows(h, tq, max(lk))
    kwargs = dict(seqlen_k=max(lk), num_seqlens=len(SEQS), varlen=varlen)
    if bias:
        kwargs['bias'] = torch.randn(1, h, tq, tk, device='cuda', dtype=torch.bfloat16)
    key = (body, head_dim, causal_type, bias, False, False, 'bf16')
    _assert_same(_run_both(key, (q, k, v, do, dk, dv, lse, delta, 1, max(lq)), dk, dv, kwargs))


# Where dropout's 1/(1-p) meets P for dV (traits `dropout_scale_at`): each body's
# default keeps its original's bits above; the other is held to fp64 here.
OTHER_SCALE_AT = {'narrow': 'elem', 'wide': 'store', 'xwide': 'store', 'xxwide': 'store'}


@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("body,head_dim", CASES)
def test_dropout_scale_at(body, head_dim, causal):
    """The non-default `dropout_scale_at` against fp64, at the tolerance of the body's
    own reference test (test_bwd_dkdv.py: narrow; test_bwd_dkdv_wide.py: the rest)."""
    sq, sk = 300, 517
    b, h, p = 2, 3, 0.3
    seed, off1, off2 = 0x123456789ABCDEF0, 1000, 77
    dt = torch.bfloat16
    torch.manual_seed(head_dim + causal)
    q, k, v, do = (torch.randn(b, h, n, head_dim, device='cuda', dtype=dt) for n in (sq, sk, sk, sq))
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off1 + off2) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    keep = torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk)
    live = bottom_right(sq, sk) if causal else None
    scale = head_dim ** -0.5
    if body == 'narrow':
        dk_r, dv_r, dk_f, dv_f, lse, delta = narrow_ref._reference(q, k, v, do, scale, dt, live=live,
                                                                   keep=keep.double(), p=p)
    else:
        dk_r, dv_r, lse, delta = wide_ref._reference(q, k, v, do, scale, keep_mask=live, keep=keep, p=p)
    fn = _builder(body, 'new')(head_dim,
                               causal_type=2 if causal else 0,
                               dropout=True,
                               dropout_scale_at=OTHER_SCALE_AT[body])
    dk = torch.full_like(k, float('nan'))
    dv = torch.full_like(v, float('nan'))
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       lse.reshape(b * h, sq).contiguous(),
       delta.reshape(b * h, sq).contiguous(),
       b,
       sq,
       seqlen_k=sk,
       scale=scale,
       dropout_p=p,
       philox_seed=seed,
       philox_offset1=torch.tensor([off1], dtype=torch.int64, device='cuda'),
       philox_offset2=off2)
    torch.cuda.synchronize()
    assert_written('new body', dK=dk, dV=dv)
    tag = f'{body} d{head_dim} dropout_scale_at={OTHER_SCALE_AT[body]} causal={causal}'
    if body == 'narrow':
        narrow_ref._check(f'dV {tag}', dv, dv_r, dv_f)
        narrow_ref._check(f'dK {tag}', dk, dk_r, dk_f)
    else:
        wide_ref._check(f'dV {tag}', dv, dv_r)
        wide_ref._check(f'dK {tag}', dk, dk_r)


@pytest.mark.parametrize("shape", PD_SHAPES)
@pytest.mark.parametrize("window", PD_WINDOWS)
@pytest.mark.parametrize("p_bit", PD_BITS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_principal_diagonal(body, head_dim, p_bit, window, shape):
    """Numeric windows under both principal diagonals: bit-identical dK and dV, and
    every owned element written (dK = dV = 0 where no query sees a key)."""
    batch, lq, lk, varlen, _, _ = _pd_varlen(shape, p_bit)
    h, hk = 4, 2
    torch.manual_seed(head_dim + sum(lq))
    tq, tk = (sum(lq), sum(lk)) if shape == 'varlen' else (lq[0], lk[0])
    q = torch.randn(batch, h, tq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch, hk, tk, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    lse, delta = _rows(batch * h, tq, max(lk))
    kwargs = dict(seqlen_k=max(lk), varlen=varlen, window=window)
    if shape == 'varlen':
        kwargs['num_seqlens'] = len(lq)
    key = (body, head_dim, 3, False, False, False, 'bf16')
    _assert_same(_run_both(key, (q, k, v, do, dk, dv, lse, delta, batch, max(lq)), dk, dv, kwargs))


# ---------------------------------------------------------------- the mask point against fp64
# `mask_at` (softmax.MASK_SCORE / MASK_PROB) is a traits field of every dK/dV body.
# Each value is held to fp64 at the tolerance of the body's own reference test
# (narrow: test_bwd_dkdv.py's rounding floor; the rest: test_bwd_dkdv_wide.py's
# REL_TOL), over windows under both principal diagonals, Sq <, == and > Sk, decode,
# chunked prefill, mixed varlen, bias, dropout, NaN slack in padded heads, f16 and
# bf16. Rows that see no key get the forward's +inf LSE, or a finite 0.0, so that
# the mask alone has to zero their P; keys no row sees must come back exactly 0.
MASK_ATS = ('score', 'prob')
REF_WINDOWS = (None, (abi.WINDOW_BOTRIGHT, 0), (8, 0), (0, 8), (-1, 3), (100, 37))
REF_SHAPES = ((1, 200), (48, 200), (150, 150), (200, 70))


@functools.cache
def _build_at(body, head_dim, causal_type, bias, dropout, padded_head, dtype_str, mask_at):
    return _builder(body, 'new')(head_dim,
                                 causal_type=causal_type,
                                 bias=bias,
                                 dropout=dropout,
                                 padded_head=padded_head,
                                 dtype_str=dtype_str,
                                 mask_at=mask_at)


def _ref_check(body, tag, got, exact, floor, unseen):
    if body == 'narrow':
        narrow_ref._check(tag, got, exact, floor, unseen)
    else:
        wide_ref._check(tag, got, exact, dead_cols=unseen)


def _dead_lse(lse, lse_mode):
    """The forward's +inf LSE on rows that see no key, or a finite 0.0 there."""
    return lse if lse_mode == 'inf' else torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)


def _keep(b, h, sq, sk, seed, off, p):
    """The forward's dropout keep: offset = base + plane*sq*stride + row*stride + col//4."""
    stride = -(-sk // 4)
    zh = np.arange(b * h, dtype=np.uint64)[:, None, None]
    m = np.arange(sq, dtype=np.uint64)[None, :, None]
    g = np.arange(stride, dtype=np.uint64)[None, None, :]
    offs = np.uint64(off) + zh * np.uint64(sq * stride) + m * np.uint64(stride) + g
    rand = philox_4x32_10(seed, offs).reshape(b * h, sq, stride * 4)[:, :, :sk]
    return torch.from_numpy(rand > philox.dropout_threshold(p)).cuda().view(b, h, sq, sk).double()


@pytest.mark.parametrize("lse_mode", ("inf", "finite"))
@pytest.mark.parametrize("shape", REF_SHAPES)
@pytest.mark.parametrize("p_bit", (0, 1))
@pytest.mark.parametrize("window", REF_WINDOWS)
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("padded", (False, True))
@pytest.mark.parametrize("dropout", (False, True))
@pytest.mark.parametrize("bias", (False, True))
@pytest.mark.parametrize("mask_at", MASK_ATS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_mask_at_reference(body, head_dim, mask_at, bias, dropout, padded, dtype, window, p_bit, shape, lse_mode):
    causal_type = 0 if window is None else 3
    if bias and causal_type:
        pytest.skip("bias and causal are mutually exclusive")
    if window is None and (p_bit or lse_mode == 'finite'):
        pytest.skip("no window: no shift, and every row sees a key")
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    b, h, hk = 2, 4, 2
    sq, sk = shape
    d = head_dim - 5 if padded else head_dim
    torch.manual_seed(head_dim + sq + sk)
    if padded:
        q, do = (with_nan_slack(b, h, sq, d, dt=dt) for _ in range(2))
        k, v = (with_nan_slack(b, hk, sk, d, dt=dt) for _ in range(2))
        dk, dv = (with_nan_slack(b, hk, sk, d, dt=dt) for _ in range(2))
    else:
        q, do = (torch.randn(b, h, sq, d, device='cuda', dtype=dt) for _ in range(2))
        k, v = (torch.randn(b, hk, sk, d, device='cuda', dtype=dt) for _ in range(2))
        dk, dv = torch.empty_like(k), torch.empty_like(v)
    dk.fill_(float('nan'))
    dv.fill_(float('nan'))
    live = None if window is None else keep_mask(window, p_bit, sq, sk)
    bias_t = torch.randn(b, h, sq, sk, device='cuda', dtype=dt) if bias else None
    pdrop, seed, off1, off2 = 0.3, 0x123456789ABCDEF0, 1000, 77
    keep = _keep(b, h, sq, sk, seed, off1 + off2, pdrop) if dropout else None
    scale = d ** -0.5
    dk_r, dv_r, dk_f, dv_f, lse, delta = narrow_ref._reference(q, k, v, do, scale, dt, live=live, bias=bias_t,
                                                               keep=keep, p=pdrop)
    kwargs = dict(seqlen_k=sk, scale=scale)
    if window is not None:
        bit = abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p_bit else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT
        kwargs.update(window=window, varlen=abi.varlen_dense(principal_diagonal=bit))
    if bias:
        kwargs['bias'] = bias_t
    if dropout:
        kwargs.update(dropout_p=pdrop,
                      philox_seed=seed,
                      philox_offset1=torch.tensor([off1], dtype=torch.int64, device='cuda'),
                      philox_offset2=off2)
    fn = _build_at(body, head_dim, causal_type, bias, dropout, padded, dtype, mask_at)
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       _dead_lse(lse, lse_mode).reshape(b * h, sq).contiguous(),
       delta.reshape(b * h, sq).contiguous(),
       b,
       sq,
       **kwargs)
    torch.cuda.synchronize()
    assert_written(f'mask_at={mask_at}', dK=dk, dV=dv)
    unseen = None if live is None else ~live.any(0)
    tag = f'{body} d{head_dim} mask_at={mask_at} w{window} p{p_bit} {shape} lse={lse_mode}'
    _ref_check(body, f'dV {tag}', dv, dv_r, dv_f, unseen)
    _ref_check(body, f'dK {tag}', dk, dk_r, dk_f, unseen)


@pytest.mark.parametrize("lse_mode", ("inf", "finite"))
@pytest.mark.parametrize("p_bit", (0, 1))
@pytest.mark.parametrize("window", REF_WINDOWS[1:])
@pytest.mark.parametrize("dtype", ("bf16", "f16"))
@pytest.mark.parametrize("mask_at", MASK_ATS)
@pytest.mark.parametrize("body,head_dim", CASES)
def test_mask_at_reference_varlen(body, head_dim, mask_at, dtype, window, p_bit, lse_mode):
    """A mixed varlen batch (decode, chunked prefill, square, Sq > Sk, a sequence the
    shifted band can miss), each sequence against its own fp64 reference."""
    dt = torch.bfloat16 if dtype == 'bf16' else torch.float16
    h, hk = 4, 2
    lq = [a for a, _ in PD_SEQS]
    lk = [c for _, c in PD_SEQS]
    cu_q = [0] + list(np.cumsum(lq))
    cu_k = [0] + list(np.cumsum(lk))
    torch.manual_seed(head_dim + len(lq))
    q = torch.randn(1, h, sum(lq), head_dim, device='cuda', dtype=dt)
    k = torch.randn(1, hk, sum(lk), head_dim, device='cuda', dtype=dt)
    v = torch.randn_like(k)
    do = torch.randn_like(q)
    dk = torch.full_like(k, float('nan'))
    dv = torch.full_like(v, float('nan'))
    scale = head_dim ** -0.5
    refs = []
    lse = torch.empty(h, sum(lq), device='cuda')
    delta = torch.empty(h, sum(lq), device='cuda')
    for i in range(len(lq)):
        qs, ks = slice(cu_q[i], cu_q[i + 1]), slice(cu_k[i], cu_k[i + 1])
        live = keep_mask(window, p_bit, lq[i], lk[i])
        r = narrow_ref._reference(q[:, :, qs], k[:, :, ks], v[:, :, ks], do[:, :, qs], scale, dt, live=live)
        lse[:, qs] = r[4][0]
        delta[:, qs] = r[5][0]
        refs.append((ks, r, ~live.any(0)))
    bits = abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p_bit else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT
    varlen = abi.varlen_compact(torch.tensor(cu_q, device='cuda', dtype=torch.int32),
                                torch.tensor(cu_k, device='cuda', dtype=torch.int32),
                                max(lq),
                                max(lk),
                                lse_tokens=sum(lq),
                                lse_layout=abi.VARLEN_LSE_LAYOUT_HT)
    varlen['bits'] = abi.principal_diagonal_bits(varlen['bits'], bits)
    fn = _build_at(body, head_dim, 3, False, False, False, dtype, mask_at)
    fn(q,
       k,
       v,
       do,
       dk,
       dv,
       _dead_lse(lse, lse_mode),
       delta,
       1,
       max(lq),
       seqlen_k=max(lk),
       num_seqlens=len(lq),
       varlen=varlen,
       window=window,
       scale=scale)
    torch.cuda.synchronize()
    assert_written(f'mask_at={mask_at}', dK=dk, dV=dv)
    for i, (ks, r, unseen) in enumerate(refs):
        tag = f'{body} d{head_dim} mask_at={mask_at} w{window} p{p_bit} seq{i} lse={lse_mode}'
        _ref_check(body, f'dV {tag}', dv[:, :, ks], r[1], r[3], unseen)
        _ref_check(body, f'dK {tag}', dk[:, :, ks], r[0], r[2], unseen)
