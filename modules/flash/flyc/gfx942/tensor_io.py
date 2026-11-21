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
              scale=None):
    """Store accumulator tiles `accs` of GEMM `t` (acc^T: columns = the GEMM's output rows).

    32x32x8: value group g of tile o holds columns t.group_coords[g] + 4*lh +
    [0, 4) of global row `row`. 16x16x16 (chunked A^T): accumulators 2p and 2p+1
    hold columns 32p + 8*lh + 2i + e, so a lane stores 8 consecutive columns per
    pair; `lh` may be a function, called after the scale vector is built.
    `scale` (f32) multiplies each tile first; columns at or past `col_limit`
    are masked only where a padded build can reach them.
    """
    if t.mfma == '16x16x16':
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
    for o in range_constexpr(t.out_tiles):
        acc = Vec(accs[o])
        if const_expr(scale is not None):
            acc = acc * fx.full(16, fx.Float32(scale), fx.Float32)
        for i in range_constexpr(4):
            vals = [acc[4 * i + j] for j in range(4)]
            packed = Vec.from_elements([fx.Float32(x).to(elem) for x in vals], elem)
            col0 = t.group_coords[4 * o + i]
            d = fx.Index(col0) + lh * 4
            off = row * row_stride + d
            ok = row_ok
            if const_expr(col0 + 8 > pad_floor):
                ok = ok & (fx.Int32(d) < col_limit)
            buffer_ops.buffer_store(packed.ir_value(), rsrc, off, mask=ok)
