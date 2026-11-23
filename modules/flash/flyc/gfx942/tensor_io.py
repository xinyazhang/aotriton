# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tensor output: a `flex_gemm` accumulator to global memory.

The lane map comes from the GEMM's traits, so the forward's O and the backward's
dQ, dK and dV share one store. Called through the module (`tensor_io.store_acc`);
the kernel's `cache_tag` digests this file.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops

from . import flex_gemm


def store_acc(t,
              accs,
              rsrc,
              row,
              row_stride,
              lh,
              col_limit,
              row_ok,
              elem,
              pad_floor,
              scale=None,
              offset_first=True):
    """Store accumulator tiles `accs` of GEMM `t` (acc^T: columns = the GEMM's output rows).

    32x32x8: value group g of tile o holds columns t.group_coords[g] + 4*lh +
    [0, 4) of global row `row`. 16x16x16 (chunked A^T): accumulators 2p and 2p+1
    hold columns 32p + 8*lh + 2i + e, so a lane stores 8 consecutive columns per
    pair; `lh` may be a function, called after the scale vector is built.
    `scale` (f32) multiplies each tile first; columns at or past `col_limit`
    are masked only where a padded build can reach them. `offset_first` builds a
    store's address before its mask (the order only moves instructions).
    """
    if isinstance(t.a, flex_gemm.DerivedA):
        _store_derived(t, accs, rsrc, row, row_stride, lh, col_limit, row_ok, elem, pad_floor, scale)
        return
    if t.mfma == '16x16x16' and t.a.layout == 'chunks':
        scale_v = fx.full(4, fx.Float32(scale), fx.Float32)
        lane_g = lh() if callable(lh) else lh
        for p in range_constexpr(t.reads):
            o0 = Vec(accs[2 * p]) * scale_v
            o1 = Vec(accs[2 * p + 1]) * scale_v
            vals = []
            for i in range_constexpr(4):
                vals += [o0[i], o1[i]]
            packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
            d = fx.Index(32 * p) + fx.Index(lane_g) * 8
            ok = row_ok
            if const_expr(32 * p + 32 > pad_floor):
                ok = ok & (fx.Int32(d) < col_limit)
            buffer_ops.buffer_store(packed.ir_value(), rsrc, row * row_stride + d, mask=ok)
        return
    # Value group i of tile o: columns m*o + 4*lane_groups*i + 4*lh + [0, 4).
    span = 4 * t.lane_groups
    for o in range_constexpr(t.out_tiles):
        acc = Vec(accs[o])
        if const_expr(scale is not None):
            acc = acc * fx.full(t.acc, fx.Float32(scale), fx.Float32)
        for i in range_constexpr(t.acc // 4):
            vals = [acc[4 * i + j] for j in range(4)]
            packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
            col0 = t.m * o + span * i
            d = fx.Index(col0) + lh * 4
            if const_expr(offset_first):
                off = row * row_stride + d
            ok = row_ok
            if const_expr(col0 + span > pad_floor):
                ok = ok & (fx.Int32(d) < col_limit)
            if const_expr(not offset_first):
                off = row * row_stride + d
            buffer_ops.buffer_store(packed.ir_value(), rsrc, off, mask=ok)


def _store_derived(t, accs, rsrc, row, row_stride, lh, col_limit, row_ok, elem, pad_floor, scale):
    """A `DerivedA` GEMM's accumulators, transposed back by the store: element i of
    group (d0, g), MFMA dt, is column d0 + g*(8*(i//4) + 4*lh + i%4) + dt."""
    scale_v = fx.full(16, fx.Float32(scale), fx.Float32)
    accs = [Vec(a) * scale_v for a in accs]
    for d0, g, a0 in t.a.groups:
        for i4 in range_constexpr(4):
            vals = [accs[a0 + dt][4 * i4 + r] for r in range(4) for dt in range(g)]
            for h in range_constexpr(max(1, g // 2)):
                part = vals[8 * h:8 * h + 8] if g >= 2 else vals
                d = fx.Index(d0 + g * 8 * i4 + 8 * h) + lh * (4 * g)
                packed = Vec.from_elements([fx.Float32(x).to(elem) for x in part], elem)
                ok = row_ok
                if const_expr(d0 + g * 8 * i4 + 8 * h + 4 * g + len(part) > pad_floor):
                    ok = ok & (fx.Int32(d) < col_limit)
                buffer_ops.buffer_store(packed.ir_value(), rsrc, row * row_stride + d, mask=ok)
