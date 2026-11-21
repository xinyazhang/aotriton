# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEMM components of the gfx942 attention kernels, and the `Ctx` a schedule drives them with.

One component, `gemm`, serves every attention GEMM (flyc-gfx942-flex-gemm-design.md,
R2): acc += A.B on 32x32x8 MFMAs, B a register operand passed in (resident Q,
or P/dS from the previous GEMM), the accumulator passed in or fresh. What
differs between the GEMMs is only the traits, chiefly A's source:

- `RowA`: A row-major in LDS, DMA'd; one 16 B read feeds two MFMAs. Forward QK.
- `PermA`: A transposed by v_perm on its way from VGPRs into LDS; one 8 B read
  per MFMA. Forward PV.

`trait_ab` and `trait_tb` are presets of these traits. A GEMM is frozen traits
(Python values only) plus stateless phase functions. Phases take explicit tile offsets and a trace-time buffer index, and
never wait, barrier or set priorities: the schedule places them in its
clusters and owns the waits, using the load counts the traits publish.

Kernels import the phases by name (`from .flex_gemm import x`) so FlyDSL's JIT
cache hashes their source. The presets run at builder scope and are called as
`flex_gemm.trait_ab(...)`; the traits they build reach the cache through `cache_tag`.
"""

from dataclasses import astuple, dataclass, fields

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

from .utils import WAVE, col_mask, lds_ptr_at, mfma_32x32x8, set_alias_scopes


def elem_type(dtype):
    return fx.BFloat16 if dtype == 'bf16' else fx.Float16


def cache_tag(traits):
    """A string naming every field: the JIT cache hashes it, not the traits object."""
    return '_'.join(f'{f.name}={v}' for f, v in zip(fields(traits), astuple(traits)))


# ---------------------------------------------------------------- A sources

@dataclass(frozen=True)
class RowA:
    """A row-major in LDS, rows padded to an odd number of 16 B chunks; DMA'd.
    The GEMM contracts over `cols`."""
    rows: int            # per tile: the accumulator's rows
    cols: int            # the contraction
    pad_floor: int       # columns below this are real in every call
    threads: int
    lds_base: int
    scope: str           # buffer b is alias scope f'{scope}{b}'

    @property
    def row_bytes(self):
        return self.cols * 2

    @property
    def stride(self):
        # 8 lanes reading 8 rows hit 8 bank groups; every read is one base plus an immediate.
        chunks = self.cols // 8
        return (chunks + 1 - chunks % 2) * 16

    @property
    def dma_round_bytes(self):
        return self.threads * 4

    @property
    def n_dma(self):
        """Loads one `gemm_dma` issues per thread."""
        return -(-self.rows * self.stride // self.dma_round_bytes)

    @property
    def tile_bytes(self):
        return self.n_dma * self.dma_round_bytes


@dataclass(frozen=True)
class PermA:
    """A^T in LDS as [cols][rows + 4], written by v_perm from VGPRs. The GEMM
    contracts over `rows`."""
    rows: int            # per tile: the contraction
    cols: int            # the accumulator's rows
    threads: int
    lds_base: int
    scope: str

    @property
    def stride(self):
        """Elements between A^T rows; the 8 B pad spreads 16 lanes over all banks."""
        return self.rows + 4

    @property
    def tile_bytes(self):
        return self.cols * self.stride * 2

    @property
    def units(self):
        """Load units: 4 A rows x 2 columns, one v_perm 4x2 transpose each."""
        return (self.rows // 4) * (self.cols // 2)

    @property
    def units_per_thread(self):
        return -(-self.units // self.threads)

    @property
    def n_load(self):
        """Loads one `gemm_load` issues per thread."""
        return 4 * self.units_per_thread


# ---------------------------------------------------------------- gemm traits

@dataclass(frozen=True)
class GemmTraits:
    """One attention GEMM on 32x32x8: acc[o] += A[o] . B, o over 32-row output tiles."""
    dtype: str
    a: RowA | PermA

    @property
    def out_tiles(self):
        return (self.a.rows if isinstance(self.a, RowA) else self.a.cols) // 32

    @property
    def split(self):
        """MFMAs fed by one A read: a 16 B row-major read covers two k-steps."""
        return 2 if isinstance(self.a, RowA) else 1

    @property
    def reads(self):
        """A reads per output tile."""
        k = self.a.cols if isinstance(self.a, RowA) else self.a.rows
        return k // 8 // self.split

    @property
    def ns(self):
        """Accumulator values per lane."""
        return 16 * self.out_tiles

    @property
    def coords(self):
        """Row of accumulator value r, relative to the lane's base row (lane // 32) * 4."""
        return [32 * (r // 16) + 8 * ((r % 16) // 4) + r % 4 for r in range(self.ns)]

    @property
    def group_coords(self):
        """Row of each 4-value group's first value, relative to the lane's base row."""
        return [32 * (g // 4) + 8 * (g % 4) for g in range(self.ns // 4)]


def trait_ab(dtype, rows, cols, pad_floor, threads, lds_base, scope):
    """acc^T = A.B^T over `cols`: A row-major, DMA'd; B resident."""
    return GemmTraits(dtype, RowA(rows, cols, pad_floor, threads, lds_base, scope))


def trait_tb(dtype, rows, cols, threads, lds_base, scope):
    """acc^T += A^T.B over `rows`: A^T by v_perm; B the previous GEMM's output."""
    return GemmTraits(dtype, PermA(rows, cols, threads, lds_base, scope))


# ---------------------------------------------------------------- B resident

def resident_packs(t, rsrc, row, row_stride, lh, hdim):
    """B packs from global, for a `RowA` GEMM: k-step 2s+e reads columns
    16s + 8*lh + 4e + [0, 4) of `row`; slack columns zeroed (0 * NaN is NaN)."""
    elem = elem_type(t.dtype)
    packs = []
    for s in range_constexpr(t.reads):
        off = row * row_stride + fx.Index(16 * s) + lh * 8
        v8 = buffer_ops.buffer_load(rsrc, off, vec_width=8, dtype=elem)
        if 16 * s + 16 > t.a.pad_floor:
            v8 = Vec(v8).bitcast(fx.Int32) & col_mask(fx.Int32(16 * s) + fx.Int32(lh * 8), hdim)
            v8 = Vec(v8).bitcast(elem).ir_value()
        packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
        packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())
    return packs


def a_col_masks(t, hdim, lh):
    """`RowA` column masks for the reads that may reach slack columns."""
    return {s: col_mask(fx.Int32(16 * s) + fx.Int32(lh * 8), hdim)
            for s in range(t.reads) if 16 * s + 16 > t.a.pad_floor}


# ---------------------------------------------------------------- RowA phases

def gemm_dma_plan(t, wave, lane, row_stride_b, lds_addr):
    """Per-thread (voffsets, M0 values) of `gemm_dma`. Lanes in the padding load row 0 again."""
    a = t.a
    voff = []
    m0 = []
    for i in range_constexpr(a.n_dma):
        line = fx.Index(i * a.dma_round_bytes) + wave * (WAVE * 4)
        p = line + lane * 4
        r = p // a.stride
        c = p % a.stride
        pad = (c >= fx.Index(a.row_bytes)) | (r >= fx.Index(a.rows))
        voff.append(pad.select(fx.Int32(0), fx.Int32(r) * row_stride_b + fx.Int32(c)))
        m0.append(fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type,
                                               fx.Int32(lds_addr + fx.Index(a.lds_base) + line))))
    return voff, m0


def gemm_dma(t, plan, rsrc, scopes, row_start, row_stride_b, buf):
    """Global -> LDS buffer `buf`, rows from `row_start`; the offset rides in soffset."""
    a = t.a
    voff, m0 = plan
    soff = fx.Int32(row_start) * row_stride_b
    al, noal = scopes(f'{a.scope}{buf}')
    for i in range_constexpr(a.n_dma):
        op = rocdl.raw_ptr_buffer_load_lds(rsrc,
                                           lds_ptr_at(fx.Int64(m0[i] + fx.Int32(buf * a.tile_bytes))),
                                           fx.Int32(4),
                                           voff[i],
                                           soff,
                                           fx.Int32(0),
                                           fx.Int32(0))
        set_alias_scopes(op, al, noal)


# ---------------------------------------------------------------- PermA phases

def gemm_load_plan(t, tid, row_stride_b):
    """Per-thread (global voffsets, LDS write offsets). Threads past the last unit redo one."""
    a = t.a
    rows = []
    lds_off = []
    for u in range_constexpr(a.units_per_thread):
        unit = (tid + fx.Index(u * a.threads)) % a.units
        g = unit // (a.cols // 2)
        cp = unit % (a.cols // 2)
        for r in range_constexpr(4):
            rows.append(fx.Int32(g * 4 + fx.Index(r)) * row_stride_b + fx.Int32(cp * 4))
        for e in range_constexpr(2):
            lds_off.append(fx.Index(a.lds_base) + ((cp * 2 + fx.Index(e)) * a.stride + g * 4) * 2)
    return rows, lds_off


def gemm_load(t, plan, rsrc_raw, row_start, row_stride_b):
    """Global -> VGPRs, rows from `row_start`."""
    soff = fx.Int32(row_start) * row_stride_b
    return [fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, rsrc_raw, off, soff, 0)) for off in plan[0]]


def gemm_stage(t, plan, lds, scopes, regs, buf):
    """VGPRs -> A^T in LDS buffer `buf`: each unit's 4x2 block transposed by v_perm."""
    a = t.a
    sel_lo = fx.Int32(0x05040100)
    sel_hi = fx.Int32(0x07060302)
    for u in range_constexpr(a.units_per_thread):
        w = regs[4 * u:4 * u + 4]
        d0 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_lo)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_lo))]
        d1 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_hi)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_hi))]
        for e, dw in ((0, d0), (1, d1)):
            off = plan[1][2 * u + e] + buf * a.tile_bytes
            al, noal = scopes(f'{a.scope}{buf}')
            llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                         lds.view(off, fx.Int32, 8).llvm_ptr,
                         alignment=8,
                         alias_scopes=al,
                         noalias_scopes=noal)


# ---------------------------------------------------------------- read, mma

def gemm_read_plan(t, lm, lh):
    """Per-thread LDS byte offsets of `gemm_read`."""
    a = t.a
    if isinstance(a, RowA):
        return [fx.Index(a.lds_base) + lm * a.stride + (fx.Index(2 * s) + lh) * 16 for s in range(t.reads)]
    return fx.Index(a.lds_base) + (lm * a.stride + lh * 4) * 2


def gemm_read(t, plan, lds, buf):
    """A from LDS buffer `buf`: [out_tiles][reads], 16 B (RowA) or 8 B (PermA) each."""
    a = t.a
    elem = elem_type(t.dtype)
    scope = f'{a.scope}{buf}'
    if isinstance(a, RowA):
        v8 = Vec.make_type(8, elem)
        return [[lds.load(off + buf * a.tile_bytes + o * 32 * a.stride, v8, 16, scope) for off in plan]
                for o in range(t.out_tiles)]
    v4 = Vec.make_type(4, elem)
    base = plan + buf * a.tile_bytes
    return [[lds.load(base + (32 * o * a.stride + 8 * s) * 2, v4, 8, scope) for s in range(t.reads)]
            for o in range(t.out_tiles)]


def gemm_mma(t, a_reads, b_packs, accs=None, a_masks=None):
    """acc[o] += A[o] . B over all k-steps, output tile outer. `accs` None starts
    each tile from zero; `a_masks` (from `a_col_masks`) zero A's slack columns."""
    elem = elem_type(t.dtype)
    if a_masks:
        a_reads = [[Vec(Vec(r).bitcast(fx.Int32) & a_masks[s]).bitcast(elem).ir_value() if s in a_masks else r
                    for s, r in enumerate(row)] for row in a_reads]
    out = [] if accs is None else list(accs)
    for o in range_constexpr(t.out_tiles):
        acc = fx.full(16, 0.0, fx.Float32) if accs is None else out[o]
        for s in range_constexpr(t.reads):
            if t.split == 2:
                a8 = Vec(a_reads[o][s])
                acc = mfma_32x32x8(a8.shuffle(a8, [0, 1, 2, 3]).ir_value(), b_packs[2 * s], acc, elem)
                acc = mfma_32x32x8(a8.shuffle(a8, [4, 5, 6, 7]).ir_value(), b_packs[2 * s + 1], acc, elem)
            else:
                acc = mfma_32x32x8(a_reads[o][s], b_packs[s], acc, elem)
        if accs is None:
            out.append(acc)
        else:
            out[o] = acc
    return out


# ---------------------------------------------------------------- Ctx

class Ctx:
    """One kernel's trace-time state: frozen traits, plus the run-time values its
    phases and hooks pass around. `carried` names the values that cross the tile
    loop's back edge; `carry`/`restore` move them through the loop's iter args."""

    def __init__(self, traits, carried):
        self.traits = traits
        self.carried = carried
        self._sizes = None

    def carry(self):
        out = []
        sizes = []
        for name in self.carried:
            v = getattr(self, name)
            if isinstance(v, list):
                out.extend(v)
                sizes.append(len(v))
            else:
                out.append(v)
                sizes.append(None)
        self._sizes = sizes
        return out

    def restore(self, values):
        values = list(values)
        i = 0
        for name, n in zip(self.carried, self._sizes):
            if n is None:
                setattr(self, name, values[i])
                i += 1
            else:
                setattr(self, name, values[i:i + n])
                i += n
        return self

    def tiles(self, iteration, j_clean, j_clean_end, n_loop):
        """Run `iteration(j, buf, masked)` over [0, n_loop): masked tiles on both
        sides of [j_clean, j_clean_end), the clean ones unrolled by two so `buf` is a
        constant. `j_clean` must be even."""
        self.restore(run_tiles(self, iteration, j_clean, j_clean_end, n_loop))


def _step(ctx, iteration, j, carried, buf, masked):
    ctx.restore(carried)
    iteration(j, buf, masked)
    return ctx.carry()


@flyc.jit
def _masked_step(ctx, iteration, j, carried):
    out = list(carried)
    if j % 2 == fx.Index(0):
        out = _step(ctx, iteration, j, carried, 0, True)
    else:
        out = _step(ctx, iteration, j, carried, 1, True)
    return out


@flyc.jit
def run_tiles(ctx, iteration, j_clean, j_clean_end, n_loop):
    init = ctx.carry()
    results0 = init
    for j, carried in range(0, j_clean, 1, init=init):
        results0 = yield _masked_step(ctx, iteration, j, carried)
    n_pairs = (j_clean_end - j_clean) // 2
    results1 = results0
    for jp, carried in range(0, n_pairs, 1, init=results0):
        mid = _step(ctx, iteration, j_clean + jp * 2, carried, 0, False)
        results1 = yield _step(ctx, iteration, j_clean + jp * 2 + 1, mid, 1, False)
    results2 = results1
    for j, carried in range(j_clean + n_pairs * 2, n_loop, 1, init=results1):
        results2 = yield _masked_step(ctx, iteration, j, carried)
    return results2
