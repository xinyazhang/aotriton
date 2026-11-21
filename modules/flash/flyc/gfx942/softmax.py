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

The LSE helpers at the end serve the backward too: its LSE and Delta rows share
the forward's LSE layout.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
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
    tokens = fmha.lse_token_pitch(varlen_bits,
                                  0,
                                  max_seqlen_q,
                                  seqinfo_q0,
                                  seqinfo_q1,
                                  num_seqlens)
    row0, pitch = fmha.lse_row_addressing(varlen_bits,
                                          batch,
                                          head,
                                          num_head_q,
                                          tokens,
                                          row_off)
    addr = fx.Int64(fx.ptrtoint(ptr))
    base = addr + fx.Int64(row0) * fx.Int64(4)
    n_bytes = (addr == fx.Int64(0)).select(fx.Int64(0), fx.Int64(rows * pitch) * fx.Int64(4))
    return buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=n_bytes), pitch


def lse_store(rsrc, pitch, lse, row, row_ok, lane_first):
    """One lane of each row stores: `lane_first()` says which (built after the offset)."""
    off = row * pitch
    buffer_ops.buffer_store(lse, rsrc, off, mask=row_ok & lane_first())
