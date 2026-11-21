# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forward's softmax hooks, on the S^T accumulator of a `flex_gemm` GEMM.

`head` (scale, bias, mask, running max) and `tail` (exp, row sum, dropout, P
packs) are split so a schedule can overlap each with a different GEMM: head(j+1)
with PV(j), tail(j) with QK(j+1). They read the kernel's run-time values from
`ctx` and the S^T lane map from `ctx.qk`. Run-time branches are `@flyc.jit`.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw

from . import flex_gemm
from .utils import LOG2E, bias_load4, exp2, mask_band, pack_operand


def wave_any(pred):
    return fx.Int64(rocdl.ballot(fx.Int64.ir_type, raw(pred))) != fx.Int64(0)


def peer(ctx, x):
    """`x` of the lane 32 away: the other half of the row."""
    xi = fx.Float32(x).bitcast(fx.Int32)
    return fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, ctx.swap_addr, xi)).bitcast(fx.Float32)


def bias_load(ctx, kv_start, exact):
    """This lane's bias for one KV tile, 4 keys per group in S^T order. `exact`
    tiles read 2 bytes at a time: the slab's last element may end mid-dword."""
    if not ctx.traits.bias:
        return None
    soff = fx.Int32(kv_start) * fx.Int32(2)
    out = []
    for g in ctx.qk.group_coords:
        off = (ctx.b_byte if exact else ctx.b_lane) + fx.Int32(g * 2)
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
    c_scale_v = fx.full(16, fx.Float32(ctx.c_scale), fx.Float32)
    scaled = [Vec(sr) * c_scale_v for sr in s_raw]
    s_vals = [scaled[r // 16][r % 16] for r in range(ns)]
    if ctx.traits.bias:
        # Upcast first: log2(e) in the input dtype would differ from the backward's.
        s_vals = [s_vals[r] + fx.Float32(Vec(bias_vals[r // 4])[r % 4].to(fx.Float32)) * fx.Float32(LOG2E)
                  for r in range(ns)]
    if masked:
        rel0 = fx.Int32(kv_start) + ctx.kv_lane - ctx.band_lo
        s_vals = mask_band(s_vals, rel0, ctx.qk.coords, ctx.band_span, ctx.c_neg_inf)
    return s_vals


@flyc.jit
def max_update(ctx, local, m_old):
    """Skips the cross-half exchange when no half-row max passed the threshold;
    exact at threshold 0, since then no max grew."""
    m_new = m_old
    corr = fx.Float32(1.0)
    if wave_any(local > m_old + fx.Float32(ctx.traits.rescale_threshold)):
        m_new = fx.maxnumf(m_old, fx.maxnumf(local, peer(ctx, local)))
        corr = exp2(m_old - m_new)
    return m_new, corr


@flyc.jit
def rescale(o_accs, corr):
    """O *= corr, skipped when no row's max moved."""
    out = list(o_accs)
    if wave_any(corr != fx.Float32(1.0)):
        corr_v = fx.full(16, fx.Float32(corr), fx.Float32)
        out = [Vec(o) * corr_v for o in o_accs]
    return out


def head(ctx, s_raw, kv_start, masked, bias_vals=None):
    """Scores of a new tile and the running max: sets ctx.s, ctx.m, ctx.corr."""
    s_vals = score_mod(ctx, s_raw, kv_start, masked, bias_vals)
    local = s_vals[0]
    for r in range_constexpr(1, ctx.qk.ns):
        local = fx.maxnumf(local, s_vals[r])
    ctx.m, ctx.corr = max_update(ctx, local, ctx.m)
    ctx.s = s_vals


def keep(ctx, kv_start):
    """Dropout keep flags in S^T order: each 4-key group is one Philox call."""
    out = []
    ph_tile = ctx.ph_row + fx.Int64(fx.Int32(kv_start) >> fx.Int32(2))
    for g in ctx.qk.group_coords:
        vals = ctx.rng.u32(ctx.ph_seed, ph_tile + fx.Int64(g // 4))
        out.extend(fx.Int32(v) > fx.Int32(ctx.idropout_p) for v in vals)
    return out


def tail(ctx, kv_start):
    """P of ctx.s as B packs of the next GEMM; updates ctx.l. l sums P before
    dropout; 1/(1-p) joins 1/l at the end."""
    ns = ctx.qk.ns
    p = [exp2(ctx.s[r] - ctx.m) for r in range(ns)]
    psum = p[0]
    for r in range_constexpr(1, ns):
        psum = psum + p[r]
    ctx.l = ctx.l * ctx.corr + psum
    if ctx.traits.dropout:
        flags = keep(ctx, kv_start)
        p = [flags[r].select(p[r], fx.Float32(0.0)) for r in range(ns)]
    elem = flex_gemm.elem_type(ctx.traits.dtype)
    return [pack_operand(p[4 * i:4 * i + 4], elem) for i in range(ns // 4)]
