# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forward's softmax hooks, on the S^T accumulator of a `flex_gemm` GEMM.

`head` (scale, bias, mask, running max) and `tail` (exp, row sum, dropout, P
packs) are split so a schedule can overlap each with a different GEMM: head(j+1)
with PV(j), tail(j) with QK(j+1). They read the kernel's run-time values from
`ctx` and the S^T lane map from `ctx.qk`. Run-time branches are `@flyc.jit`.

The QK MFMA shape fixes how a row spreads over lanes, named in EXCHANGE and
REDUCE: on 32x32x8 a row is two lane halves (one peer, 32 away) holding 16
values each, summed serially; on 16x16x16 four lane groups (peers 16 and 32)
holding 4 values each, summed pairwise. The body supplies `ctx.band()` (the
row's key band) and `ctx.lane_key()` (the lane's first key in a tile), and may
supply `ctx.peer_addr(mask)` and `ctx.ph_lane()` to rebuild lane values at their
use instead of keeping them live.

The LSE helpers serve the backward too: its LSE and Delta rows share the
forward's LSE layout. The backward hooks at the end recompute P from LSE and form
dS = P (dP - Delta); how P's exponent rounds is a named option (P_EXP_*), since
the bodies differ there.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw

from kernels.common import buffer_ops

import fmha_utils as fmha
from . import flex_gemm
from .utils import LN2, LOG2E, bias_load4, exp2, mask_band, pack_operand


EXCHANGE = {'32x32x8': (32,), '16x16x16': (16, 32)}     # peer lane distances of a row
REDUCE = {'32x32x8': 'serial', '16x16x16': 'pairwise'}  # order of a lane's row sum and max


def _reduce(vals, op, order):
    if order == 'pairwise':
        while len(vals) > 1:
            vals = [op(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        return vals[0]
    acc = vals[0]
    for v in vals[1:]:
        acc = op(acc, v)
    return acc


def wave_any(pred):
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, raw(pred))) != fx.Int64(0)


def peer(ctx, x, mask=32):
    """`x` of lane ^ `mask`: another part of the row."""
    addr = ctx.peer_addr(mask) if hasattr(ctx, 'peer_addr') else ctx.swap_addr
    xi = fx.Float32(x).bitcast(fx.Int32)
    return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, addr, xi)).bitcast(fx.Float32)


def row_max(ctx, x):
    """Max of `x` over the row's lanes."""
    for mask in EXCHANGE[ctx.qk.mfma]:
        x = fx.maxnumf(x, peer(ctx, x, mask))
    return x


def row_sum(ctx, x):
    """Sum of `x` over the row's lanes."""
    for mask in EXCHANGE[ctx.qk.mfma]:
        x = x + peer(ctx, x, mask)
    return x


def bias_load(ctx, kv_start, exact):
    """This lane's bias for one KV tile, 4 keys per group in S^T order. `exact`
    tiles read 2 bytes at a time: the slab's last element may end mid-dword."""
    if not ctx.traits.bias:
        return None
    soff = fx.Int32(kv_start) * fx.Int32(2)
    out = []
    for g in ctx.qk.group_coords:
        off = ctx.b_byte if exact else ctx.b_lane
        if len(ctx.qk.group_coords) > 1:
            off = off + fx.Int32(g * 2)
        out.append(bias_load4(ctx.b_rsrc,
                              off,
                              soff,
                              ctx.b_shift,
                              exact,
                              flex_gemm.elem_type(ctx.traits.dtype)))
    return out


def score_mod(ctx, s_raw, kv_start, masked, bias_vals):
    """Scale once in fp32 (the max and the exponent share that rounding), add bias, mask."""
    ns = ctx.qk.ns
    acc = ctx.qk.acc
    c_scale_v = fx.full(acc, fx.Float32(ctx.c_scale), fx.Float32)
    scaled = [Vec(sr) * c_scale_v for sr in s_raw]
    s_vals = [scaled[r // acc][r % acc] for r in range(ns)]
    if ctx.traits.bias:
        # Upcast first: log2(e) in the input dtype would differ from the backward's.
        s_vals = [s_vals[r] + fx.Float32(Vec(bias_vals[r // 4])[r % 4].to(fx.Float32)) * fx.Float32(LOG2E)
                  for r in range(ns)]
    if masked:
        band_lo, band_span = ctx.band()
        rel0 = fx.Int32(kv_start) + ctx.lane_key() - band_lo
        s_vals = mask_band(s_vals, rel0, ctx.qk.coords, band_span, ctx.c_neg_inf)
    return s_vals


@flyc.jit
def max_update(ctx, local, m_old):
    """Skips the cross-half exchange when no half-row max passed the threshold;
    exact at threshold 0, since then no max grew."""
    m_new = m_old
    corr = fx.Float32(1.0)
    if wave_any(local > m_old + fx.Float32(ctx.traits.rescale_threshold)):
        m_new = fx.maxnumf(m_old, row_max(ctx, local))
        corr = exp2(m_old - m_new)
    return m_new, corr


@flyc.jit
def rescale(o_accs, corr, width=16):
    """O *= corr, skipped when no row's max moved. `width`: f32 per accumulator."""
    out = list(o_accs)
    if wave_any(corr != fx.Float32(1.0)):
        corr_v = fx.full(width, fx.Float32(corr), fx.Float32)
        out = [Vec(o) * corr_v for o in o_accs]
    return out


def head(ctx, s_raw, kv_start, masked, bias_vals=None, sched=None):
    """Scores of a new tile and the running max: sets ctx.s, ctx.m, ctx.corr.
    `sched` (optional) pins the cluster's order before the max exchange's branch."""
    s_vals = score_mod(ctx, s_raw, kv_start, masked, bias_vals)
    local = _reduce(s_vals, fx.maxnumf, REDUCE[ctx.qk.mfma])
    if sched is not None:
        sched()
    ctx.m, ctx.corr = max_update(ctx, local, ctx.m)
    ctx.s = s_vals


def keep(ctx, kv_start):
    """Dropout keep flags in S^T order: each 4-key group is one Philox call.
    `ctx.ph_lane()`, if the body has one, is the lane's offset from `ctx.ph_row`."""
    out = []
    lane_off = ctx.ph_lane() if hasattr(ctx, 'ph_lane') else None
    ph_tile = ctx.ph_row + fx.Int64(fx.Int32(kv_start) >> fx.Int32(2))
    for g in ctx.qk.group_coords:
        vals = ctx.rng.u32(ctx.ph_seed, ph_tile + (fx.Int64(g // 4) if lane_off is None else lane_off))
        out.extend(fx.Int32(v) > fx.Int32(ctx.idropout_p) for v in vals)
    return out


def tail(ctx, kv_start):
    """P of ctx.s as B packs of the next GEMM; updates ctx.l. l sums P before
    dropout; 1/(1-p) joins 1/l at the end."""
    ns = ctx.qk.ns
    p = [exp2(ctx.s[r] - ctx.m) for r in range(ns)]
    order = REDUCE[ctx.qk.mfma]
    if order == 'pairwise':
        ctx.l = ctx.l * ctx.corr + _reduce(p, lambda x, y: x + y, order)
    else:
        psum = _reduce(p, lambda x, y: x + y, order)
        ctx.l = ctx.l * ctx.corr + psum
    if ctx.traits.dropout:
        flags = keep(ctx, kv_start)
        p = [flags[r].select(p[r], fx.Float32(0.0)) for r in range(ns)]
    elem = flex_gemm.elem_type(ctx.traits.dtype)
    return [pack_operand(p[4 * i:4 * i + 4], elem) for i in range(ns // 4)]


# ---------------------------------------------------------------- LSE rows
# Forward stores LSE; the backward reads LSE and Delta, which share the layout.

def lse_of(m, l):
    """Natural-base LSE of a finished row; +inf for a fully masked row (l = 0)."""
    lse = (m + fx.Float32(fx.math.log2(l))) * fx.Float32(LN2)
    return (l > fx.Float32(0.0)).select(lse, fx.Float32(float('inf')))


def lse_rows(varlen_bits,
             batch,
             head,
             num_head_q,
             max_seqlen_q,
             seqinfo_q0,
             seqinfo_q1,
             num_seqlens,
             row_off):
    """(first row, pitch) of one (batch, head)'s LSE or Delta rows."""
    tokens = fmha.lse_token_pitch(varlen_bits,
                                  0,
                                  max_seqlen_q,
                                  seqinfo_q0,
                                  seqinfo_q1,
                                  num_seqlens)
    return fmha.lse_row_addressing(varlen_bits,
                                   batch,
                                   head,
                                   num_head_q,
                                   tokens,
                                   row_off)


def lse_rsrc_at(ptr, row0, pitch, rows, nullable=True):
    """The rows' buffer resource; a null `ptr` gets an empty range when `nullable`."""
    addr = fx.Int64(fx.ptrtoint(ptr))
    base = addr + fx.Int64(row0) * fx.Int64(4)
    if nullable:
        n_bytes = (addr == fx.Int64(0)).select(fx.Int64(0), fx.Int64(rows * pitch) * fx.Int64(4))
    else:
        n_bytes = fx.Int64(rows * pitch) * fx.Int64(4)
    return buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=n_bytes)


def lse_delta_rsrcs(lse_ptr, delta_ptr, row0, pitch, rows):
    """The backward's LSE and Delta resources: one row layout, so one range."""
    n_bytes = fx.Int64(rows * pitch) * fx.Int64(4)
    row0_b = fx.Int64(row0) * fx.Int64(4)
    return [buffer_ops.create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(ptr)) + row0_b,
                                                        num_records_bytes=n_bytes)
            for ptr in (lse_ptr, delta_ptr)]


def lse_rsrc(ptr,
             rows,
             varlen_bits,
             batch,
             head,
             num_head_q,
             max_seqlen_q,
             seqinfo_q0,
             seqinfo_q1,
             num_seqlens,
             row_off):
    """(rsrc, pitch) of one (batch, head)'s LSE or Delta rows; a null `ptr` gets an empty range."""
    row0, pitch = lse_rows(varlen_bits,
                           batch,
                           head,
                           num_head_q,
                           max_seqlen_q,
                           seqinfo_q0,
                           seqinfo_q1,
                           num_seqlens,
                           row_off)
    return lse_rsrc_at(ptr, row0, pitch, rows), pitch


def lse_load(rsrc, pitch, row):
    """One row's LSE or Delta."""
    return fx.Float32(buffer_ops.buffer_load(rsrc, row * pitch, vec_width=1, dtype=fx.Float32))


def lse_store(rsrc, pitch, lse, row, row_ok, lane_first):
    """One lane of each row stores: `lane_first()` says which (built after the offset)."""
    off = row * pitch
    buffer_ops.buffer_store(lse, rsrc, off, mask=row_ok & lane_first())


# ---------------------------------------------------------------- backward
# On the S^T accumulator of `ctx.qk` (its lane map: ctx.qk.coords), with
# ctx.c_scale, ctx.lse2 (log2-base LSE) and ctx.delta of the lane's row.

P_EXP_TWO_ROUNDINGS = 'two_roundings'   # S * c, then - lse2: the forward's P bit for bit
P_EXP_FMA = 'fma'                       # one fma(S, c, -lse2)


def pair_op(xs, c, op):
    """op(xs, c) on adjacent pairs: v_pk_* math against a 2-wide splat of `c`.
    A 16-wide splat is hoisted out of the loop as 16 registers."""
    c2 = fx.full(2, fx.Float32(c), fx.Float32)
    out = []
    for i in range(len(xs) // 2):
        v2 = op(Vec.from_elements([xs[2 * i], xs[2 * i + 1]], fx.Float32), c2)
        out.extend([v2[0], v2[1]])
    return out


def _p_fma(ctx, s_raw, kv_start, masked, bias_vals):
    """P_EXP_FMA: per value; with bias, pairs of (S*c + bias*log2e) - lse2."""
    ns = ctx.qk.ns
    acc = ctx.qk.acc
    if ctx.traits.bias:
        # Upcast first: log2(e) in the input dtype would differ from the forward's.
        c2 = fx.full(2, fx.Float32(ctx.c_scale), fx.Float32)
        e2 = fx.full(2, fx.Float32(LOG2E), fx.Float32)
        l2 = fx.full(2, fx.Float32(ctx.lse2), fx.Float32)
        s_vals = []
        for i in range(ns // 2):
            sv = Vec.from_elements([Vec(s_raw[(2 * i + e) // acc])[(2 * i + e) % acc] for e in range(2)], fx.Float32)
            bv = Vec.from_elements([fx.Float32(Vec(bias_vals[i // 2])[2 * (i % 2) + e].to(fx.Float32))
                                    for e in range(2)], fx.Float32)
            v2 = (sv * c2 + bv * e2) - l2
            s_vals.extend([v2[0], v2[1]])
    else:
        # One fma: the exponent of S*scale - lse2, rounded once.
        s_vals = [fx.Float32(fmath.fma(Vec(s_raw[r // acc])[r % acc], ctx.c_scale, ctx.neg_lse2)) for r in range(ns)]
    if masked:
        # After the scale (numerical lessons, #7); -inf - lse2 is -inf.
        band_lo, band_span = ctx.band()
        rel0 = fx.Int32(kv_start) + ctx.lane_key() - band_lo
        s_vals = mask_band(s_vals, rel0, ctx.qk.coords, band_span, ctx.c_neg_inf)
    return [exp2(s_vals[r]) for r in range(ns)]


def p_from_lse(ctx, s_raw, kv_start, masked, bias_vals, exponent=P_EXP_TWO_ROUNDINGS):
    """P = exp2(sm_scale*log2e*S + log2e*bias - lse2): one f32 vector per S^T tile
    (P_EXP_TWO_ROUNDINGS, v_pk pairs), or one value per S^T value (P_EXP_FMA, which
    also needs ctx.neg_lse2)."""
    if exponent == P_EXP_FMA:
        return _p_fma(ctx, s_raw, kv_start, masked, bias_vals)
    ns = ctx.qk.ns
    acc = ctx.qk.acc
    # Scale once in fp32, as the forward: P must be the forward's P bit for bit.
    s_vals = pair_op([Vec(s_raw[r // acc])[r % acc] for r in range(ns)], ctx.c_scale, lambda x, c: x * c)
    if ctx.traits.bias:
        l2 = fx.full(2, fx.Float32(LOG2E), fx.Float32)
        out = []
        for i in range(ns // 2):
            b2 = Vec.from_elements([fx.Float32(Vec(bias_vals[i // 2])[2 * (i % 2) + e].to(fx.Float32))
                                    for e in range(2)], fx.Float32)
            v2 = Vec.from_elements([s_vals[2 * i], s_vals[2 * i + 1]], fx.Float32) + b2 * l2
            out.extend([v2[0], v2[1]])
        s_vals = out
    if masked:
        band_lo, band_span = ctx.band()
        rel0 = fx.Int32(kv_start) + ctx.lane_key() - band_lo
        s_vals = mask_band(s_vals, rel0, ctx.qk.coords, band_span, ctx.c_neg_inf)
    x = pair_op(s_vals, ctx.lse2, lambda x, c: x - c)
    return [Vec.from_elements([exp2(x[acc * n + r]) for r in range(acc)], fx.Float32).ir_value()
            for n in range(ns // acc)]


def keep_scale(ctx, kv_start):
    """dP's dropout factor, 1/(1-p) or 0, in S^T order: each 4-key group is one
    Philox call. Floats, not i1: 16 masks would take 32 SGPRs and spill."""
    out = []
    tile = ctx.ph_row + fx.Int64(fx.Int32(kv_start) // fx.Int32(4))
    for g in range(ctx.qk.ns // 4):
        vals = ctx.rng.u32(ctx.ph_seed, tile + fx.Int64(ctx.qk.coords[4 * g] // 4))
        out.extend((fx.Int32(v) > fx.Int32(ctx.idropout_p)).select(fx.Float32(ctx.dropout_scale), fx.Float32(0.0))
                   for v in vals)
    return out


def ds_packs(ctx, p_vals, dp_raw, keep, exponent=P_EXP_TWO_ROUNDINGS):
    """dS = P * (dP - Delta), packed as the next GEMM's B operand: in v_pk pairs
    after P_EXP_TWO_ROUNDINGS, per value after P_EXP_FMA (`p_vals` as p_from_lse
    returned them). Dropout's mask and 1/(1-p) hit dP; P stays undropped
    (numerical lessons, #9)."""
    ns = ctx.qk.ns
    acc = ctx.qk.acc
    dp = [fx.Float32(Vec(dp_raw[r // acc])[r % acc]) for r in range(ns)]
    if ctx.traits.dropout:
        dp = [dp[r] * keep[r] for r in range(ns)]
    if exponent == P_EXP_FMA:
        ds = [p_vals[r] * (dp[r] - ctx.delta) for r in range(ns)]
        elem = flex_gemm.elem_type(ctx.traits.dtype)
        return [pack_operand(ds[4 * i:4 * i + 4], elem) for i in range(ns // 4)]
    dpd = pair_op(dp, ctx.delta, lambda x, c: x - c)
    ds = []
    for i in range(ns // 2):
        pv = Vec.from_elements([Vec(p_vals[i // (acc // 2)])[2 * (i % (acc // 2)) + e] for e in range(2)],
                               fx.Float32)
        v2 = pv * Vec.from_elements([dpd[2 * i], dpd[2 * i + 1]], fx.Float32)
        ds.extend([v2[0], v2[1]])
    elem = flex_gemm.elem_type(ctx.traits.dtype)
    return [pack_operand(ds[4 * i:4 * i + 4], elem) for i in range(ns // 4)]


# ---------------------------------------------------------------- dK/dV P and dS
# A dK/dV S tile (16x16x16, Q rows streamed, K resident) holds one kv column and
# four consecutive q rows per lane, so LSE and Delta are per value and P, dS
# come out as the dV/dK GEMMs' B operand. The band mask sets S to -inf before
# exp, as the Triton reference does (bwd_inner_dk_dv.py) for every body. Where
# dropout's 1/(1-p) meets P for dV is a named option.
P_DROP_AT_STORE = 'store'   # dropout zeroes P; its 1/(1-p) joins dV at the store
P_DROP_PER_ELEM = 'elem'    # P scaled by 1/(1-p) per element


def p_ds_kv(ctx,
            s_accs,
            dp_accs,
            lse_l2,
            delta,
            bias_of,
            mask_of,
            keep_of,
            masked,
            p_drop=P_DROP_AT_STORE):
    """(P packs, dS packs), one each per kv block. `bias_of(b)` gives block b's
    bias, `mask_of(s_vals, b)` its masked scores, `keep_of(b)` its dropout keep
    flags; each is called where the block needs it."""
    assert p_drop == P_DROP_AT_STORE, p_drop
    elem = flex_gemm.elem_type(ctx.traits.dtype)
    c_scale_v = fx.full(4, fx.Float32(ctx.c_scale), fx.Float32)
    lse_v = Vec.from_elements(lse_l2, fx.Float32)
    delta_v = Vec.from_elements(delta, fx.Float32)
    p_packs = []
    ds_packs = []
    for b in range_constexpr(len(s_accs)):
        # Scale once in fp32, as the forward does; the mask lands on the scaled score.
        s = Vec(s_accs[b]) * c_scale_v
        s_vals = [s[r] for r in range(4)]
        if ctx.traits.bias:
            bv = bias_of(b)
            s_vals = [s_vals[r] + bv[r] * fx.Float32(LOG2E) for r in range(4)]
        if masked and ctx.traits.causal:
            s_vals = mask_of(s_vals, b)
        e = Vec.from_elements(s_vals, fx.Float32) - lse_v
        p = [exp2(e[r]) for r in range(4)]
        dp = Vec(dp_accs[b])
        if ctx.traits.dropout:
            keep = keep_of(b)
            dsc = fx.Float32(ctx.dropout_scale)
            p_kept = [keep[r].select(p[r], fx.Float32(0.0)) for r in range(4)]
            dp_vals = [keep[r].select(dp[r] * dsc, fx.Float32(0.0)) for r in range(4)]
            dpd = Vec.from_elements(dp_vals, fx.Float32) - delta_v
        else:
            p_kept = p
            dpd = dp - delta_v
        ds = Vec.from_elements(p, fx.Float32) * dpd
        p_packs.append(pack_operand(p_kept, elem))
        ds_packs.append(pack_operand([ds[r] for r in range(4)], elem))
    return p_packs, ds_packs
