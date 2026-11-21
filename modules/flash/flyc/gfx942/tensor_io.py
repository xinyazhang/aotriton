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

    Value group g of tile o holds columns t.group_coords[g] + 4*lh + [0, 4) of
    global row `row`. `scale` (f32) multiplies each tile first; columns at or
    past `col_limit` are masked only where a padded build can reach them.
    """
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
