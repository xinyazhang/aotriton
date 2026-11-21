# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEMM components of the gfx942 attention kernels, and the `Ctx` a schedule drives them with.

One component, `gemm`, serves every attention GEMM (flyc-gfx942-flex-gemm-design.md,
R2): acc += A.B on 32x32x8 MFMAs, B a register operand passed in (resident Q,
or P/dS from the previous GEMM), the accumulator passed in or fresh. What
differs between the GEMMs is only the traits, chiefly A's source:

- `RowA`: A row-major in LDS, DMA'd; one 16 B read feeds two MFMAs. Forward QK.
  `dma` picks the copy: 'lines' (the block copies the tile as flat 4 B lines) or
  'rows' (each wave copies whole rows, 256 B per issue).
- `PermA`: A transposed by v_perm on its way from VGPRs into LDS; one 8 B read
  per MFMA. Forward PV. `unit_cols` picks the load unit: 4 rows x 2 columns
  (dword loads) or 4 rows x 8 columns (dwordx4 loads, row groups fastest).

`buffers` is 1 or 2; with one buffer the alias scope is `scope` itself.

On 16x16x16 MFMAs (`mfma`), the xwide layouts:
- `RowA` with `dma='xor'`: wave w copies rows w and w + 8; 16 B chunk c of row r
  sits at c ^ (r % 8). Padded builds read 0 past the head dim and `gemm_fix`
  zeroes the column an odd head dim splits.
- `PermA` with `layout='chunks'`: chunk (p, g, r) at p * 1024 + g * 256 + r * 16
  holds rows 32p + 2r and 32p + 2r + 1 of A^T, keys 4g..4g+3; one 16 B read
  feeds two MFMAs of different accumulators.
Their reads go through `gemm_stream`, which software-pipelines read and MFMA.

`trait_ab` and `trait_tb` are presets of these traits. A GEMM is frozen traits
(Python values only) plus stateless phase functions. Phases take explicit tile offsets and a trace-time buffer index, and
never wait, barrier or set priorities: the schedule places them in its
clusters and owns the waits, using the load counts the traits publish.

Kernels import the `gemm_*` phases by name (`from .flex_gemm import gemm_mma`):
the name reads as running a GEMM. Everything else is called through the module
(`flex_gemm.trait_ab(...)`, `flex_gemm.Ctx(...)`). FlyDSL's JIT cache does not
see module-prefixed calls, so `cache_tag` folds in a digest of the modules' source.
"""

import hashlib
from dataclasses import astuple, dataclass, fields

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as raw
from kernels.common import buffer_ops

from flydsl.expr.typing import T

from .utils import (
    OOB_VOFF,
    WAVE,
    col_mask,
    lds_ptr_at,
    m0_at,
    mfma_16x16x16,
    mfma_32x32x8,
    pipelined,
    set_alias_scopes,
)


def elem_type(dtype):
    return fx.BFloat16 if dtype == 'bf16' else fx.Float16


def cache_tag(traits, *modules):
    """Every traits field, plus a digest of `modules`' source: the JIT cache hashes
    this string, but neither the traits object nor module-prefixed calls."""
    tag = '_'.join(f'{f.name}={v}' for f, v in zip(fields(traits), astuple(traits)))
    digest = hashlib.sha256()
    for m in modules:
        with open(m.__file__, 'rb') as f:
            digest.update(f.read())
    return f'{tag}_src={digest.hexdigest()[:16]}'


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
    dma: str = 'lines'   # 'lines' or 'rows'
    buffers: int = 2

    @property
    def row_bytes(self):
        return self.cols * 2

    @property
    def rows_per_wave(self):
        return self.rows // (self.threads // WAVE)

    @property
    def segs(self):
        """'rows' DMA: 256 B issues per row; the last ends on the row's last byte
        and may overlap the one before (same bytes, same place)."""
        return tuple(range(0, self.row_bytes - 256, 256)) + (self.row_bytes - 256,)

    @property
    def stride(self):
        if self.dma == 'xor':
            return self.row_bytes           # the XOR swizzle spreads the banks
        # 8 lanes reading 8 rows hit 8 bank groups; every read is one base plus an immediate.
        chunks = self.cols // 8
        return (chunks + 1 - chunks % 2) * 16

    @property
    def quarters(self):
        """'xor' DMA: 256 B issues per row."""
        return self.row_bytes // 256

    @property
    def dma_round_bytes(self):
        return self.threads * 4

    @property
    def n_dma(self):
        """Loads one `gemm_dma` issues per thread."""
        if self.dma == 'rows':
            return self.rows_per_wave * len(self.segs)
        if self.dma == 'xor':
            return 2 * self.quarters
        return -(-self.rows * self.stride // self.dma_round_bytes)

    @property
    def tile_bytes(self):
        if self.dma in ('rows', 'xor'):
            return self.rows * self.stride
        return self.n_dma * self.dma_round_bytes

    def scope_of(self, buf):
        return self.scope if self.buffers == 1 else f'{self.scope}{buf}'


@dataclass(frozen=True)
class PermA:
    """A^T in LDS as [cols][rows + 4], written by v_perm from VGPRs. The GEMM
    contracts over `rows`."""
    rows: int            # per tile: the contraction
    cols: int            # the accumulator's rows
    threads: int
    lds_base: int
    scope: str
    buffers: int = 2
    unit_cols: int = 2   # 2: dword loads, columns fastest; 8: dwordx4 loads, row groups fastest
    layout: str = 'rows'     # 'rows': [cols][rows + 4]; 'chunks': (p, g, r) chunks

    @property
    def stride(self):
        """Elements between A^T rows; the 8 B pad spreads 16 lanes over all banks."""
        return self.rows + 4

    @property
    def tile_bytes(self):
        if self.layout == 'chunks':
            return self.rows * self.cols * 2
        return self.cols * self.stride * 2

    @property
    def affine(self):
        """'chunks': a thread's units are whole key groups apart, so one base serves them all."""
        return self.units % self.threads == 0 and self.threads % (self.cols // 2) == 0

    @property
    def units(self):
        """Load units: 4 A rows x `unit_cols` columns, v_perm 4x2 transposes."""
        return (self.rows // 4) * (self.cols // self.unit_cols)

    @property
    def units_per_thread(self):
        return -(-self.units // self.threads)

    @property
    def n_load(self):
        """Loads one `gemm_load` issues per thread."""
        return 4 * self.units_per_thread

    @property
    def n_stage(self):
        """LDS writes one `gemm_stage` issues per thread."""
        if self.layout == 'chunks':
            return self.units_per_thread
        return self.unit_cols * self.units_per_thread

    def scope_of(self, buf):
        return self.scope if self.buffers == 1 else f'{self.scope}{buf}'


# ---------------------------------------------------------------- gemm traits

@dataclass(frozen=True)
class GemmTraits:
    """One attention GEMM: acc[o] += A[o] . B, o over the MFMA's output tiles."""
    dtype: str
    a: RowA | PermA
    mfma: str = '32x32x8'       # or '16x16x16'

    @property
    def m(self):
        """Output rows (and columns) of one MFMA."""
        return 32 if self.mfma == '32x32x8' else 16

    @property
    def acc(self):
        """f32 per lane in one accumulator."""
        return 16 if self.mfma == '32x32x8' else 4

    @property
    def out_tiles(self):
        return (self.a.rows if isinstance(self.a, RowA) else self.a.cols) // self.m

    @property
    def split(self):
        """MFMAs fed by one A read: a 16 B read covers two k-steps (RowA, chunked
        PermA); an 8 B read one."""
        return 2 if isinstance(self.a, RowA) or self.a.layout == 'chunks' else 1

    @property
    def reads(self):
        """A reads per output tile; for chunked PermA, per two output tiles."""
        if self.mfma == '16x16x16':
            # One 16 B read = 8 columns of the lane's 4-key group: two MFMAs.
            return self.a.cols // 32
        k = self.a.cols if isinstance(self.a, RowA) else self.a.rows
        return k // 8 // self.split

    @property
    def ns(self):
        """Accumulator values per lane."""
        return self.acc * self.out_tiles

    @property
    def n_reads(self):
        """LDS reads per tile."""
        if self.mfma == '16x16x16':
            return self.reads
        return self.out_tiles * self.reads

    @property
    def n_mma(self):
        """MFMAs per tile: for a schedule's sched_group_barrier counts."""
        return self.n_reads * self.split

    @property
    def coords(self):
        """Row of accumulator value r, relative to the lane's base row:
        (lane // 32) * 4 on 32x32x8, (lane // 16) * 4 on 16x16x16."""
        if self.mfma == '16x16x16':
            return list(range(self.ns))
        return [32 * (r // 16) + 8 * ((r % 16) // 4) + r % 4 for r in range(self.ns)]

    @property
    def group_coords(self):
        """Row of each 4-value group's first value, relative to the lane's base row."""
        if self.mfma == '16x16x16':
            return [0]
        return [32 * (g // 4) + 8 * (g % 4) for g in range(self.ns // 4)]


def trait_ab(dtype, rows, cols, pad_floor, threads, lds_base, scope, dma='lines', mfma='32x32x8'):
    """acc^T = A.B^T over `cols`: A row-major, DMA'd; B resident."""
    return GemmTraits(dtype,
                      RowA(rows,
                           cols,
                           pad_floor,
                           threads,
                           lds_base,
                           scope,
                           dma=dma),
                      mfma=mfma)


def trait_tb(dtype,
             rows,
             cols,
             threads,
             lds_base,
             scope,
             buffers=2,
             unit_cols=2,
             layout='rows',
             mfma='32x32x8'):
    """acc^T += A^T.B over `rows`: A^T by v_perm; B the previous GEMM's output."""
    return GemmTraits(dtype,
                      PermA(rows,
                            cols,
                            threads,
                            lds_base,
                            scope,
                            buffers=buffers,
                            unit_cols=unit_cols,
                            layout=layout),
                      mfma=mfma)


# ---------------------------------------------------------------- B resident

def resident_packs(t, rsrc, row, row_stride, lh, hdim):
    """B packs from global, for a `RowA` GEMM: k-step 2s+e reads columns
    16s + 8*lh + 4e + [0, 4) of `row`; slack columns zeroed (0 * NaN is NaN)."""
    elem = elem_type(t.dtype)
    packs = []
    if t.mfma == '16x16x16':
        # k-step 2s+e reads columns 32s + 8*lh + 4e + [0, 4); lh is the lane's 4-key group.
        for s in range_constexpr(t.reads):
            c0 = fx.Index(32 * s) + lh * 8
            v8 = buffer_ops.buffer_load(rsrc, row * row_stride + c0, vec_width=8, dtype=elem)
            if 32 * s + 32 > t.a.pad_floor:
                v8 = Vec(Vec(v8).bitcast(fx.Int32) & col_mask(fx.Int32(c0), hdim)).bitcast(elem).ir_value()
            packs.append(Vec(v8).shuffle(Vec(v8), [0, 1, 2, 3]).ir_value())
            packs.append(Vec(v8).shuffle(Vec(v8), [4, 5, 6, 7]).ir_value())
        return packs
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
    """Per-thread addressing of `gemm_dma`. 'lines': (voffsets, M0 values); lanes in
    the padding load row 0 again. 'rows': (lane voffset, the wave's M0, its first row)."""
    a = t.a
    if a.dma == 'xor':
        # Wave w fills rows w and w + 8, so its XOR key is always w: one voffset per lane.
        wave_s = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave)))
        lane_chunk = (fx.Int32(lane // 4) ^ wave_s)
        voff = lane_chunk * fx.Int32(16) + fx.Int32(lane % 4) * fx.Int32(4)
        lane_col = lane_chunk * fx.Int32(8) + fx.Int32(lane % 4) * fx.Int32(2)
        m0 = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(lds_addr + fx.Index(a.lds_base))))
        m0 = m0 + wave_s * fx.Int32(a.row_bytes)
        return wave_s, voff, lane_col, m0
    if a.dma == 'rows':
        wave_s = fx.Index(rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(wave)))
        lane_off = fx.Int32(lane * 4)
        m0_addr = fx.Int32(lds_addr + fx.Index(a.lds_base) + wave_s * (a.rows_per_wave * a.stride))
        m0_base = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, m0_addr))
        row0 = fx.Int32(wave_s * a.rows_per_wave)
        return lane_off, m0_base, row0
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


def gemm_dma(t, plan, rsrc, scopes, row_start, row_stride_b, buf, hdim=None):
    """Global -> LDS buffer `buf`, rows from `row_start`; the offset rides in soffset.
    'xor' needs `hdim` in a padded build: dwords wholly past it read 0."""
    a = t.a
    if a.dma == 'xor':
        wave_s, voff0, lane_col, m0_base = plan
        al, noal = scopes(a.scope_of(buf))
        for k in range_constexpr(2):
            soff = (fx.Int32(row_start) + wave_s + fx.Int32(8 * k)) * row_stride_b
            for q in range_constexpr(a.quarters):
                voff = voff0
                if a.pad_floor < a.cols:
                    voff = (lane_col < fx.Int32(hdim) - fx.Int32(128 * q)).select(voff, fx.Int32(OOB_VOFF))
                # m0_at: hoisted, the sums would pin SGPRs.
                m0 = m0_at(m0_base, buf * a.tile_bytes + 8 * k * a.row_bytes + 256 * q)
                op = rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                   lds_ptr_at(fx.Int64(m0)),
                                                   fx.Int32(4),
                                                   voff,
                                                   soff + fx.Int32(256 * q),
                                                   fx.Int32(0),
                                                   fx.Int32(0))
                set_alias_scopes(op, al, noal)
        return
    if a.dma == 'rows':
        lane_off, m0_base, row0 = plan
        al, noal = scopes(a.scope_of(buf))
        for i in range_constexpr(a.rows_per_wave):
            soff = (fx.Int32(row_start) + row0 + fx.Int32(i)) * row_stride_b
            m0 = fx.Int64(m0_base + fx.Int32(buf * a.tile_bytes + i * a.stride))
            for seg in a.segs:
                op = rocdl.raw_ptr_buffer_load_lds(rsrc,
                                                   lds_ptr_at(m0),
                                                   fx.Int32(4),
                                                   lane_off,
                                                   soff,
                                                   fx.Int32(seg),
                                                   fx.Int32(0))
                set_alias_scopes(op, al, noal)
        return
    voff, m0 = plan
    soff = fx.Int32(row_start) * row_stride_b
    al, noal = scopes(a.scope_of(buf))
    for i in range_constexpr(a.n_dma):
        op = rocdl.raw_ptr_buffer_load_lds(rsrc,
                                           lds_ptr_at(fx.Int64(m0[i] + fx.Int32(buf * a.tile_bytes))),
                                           fx.Int32(4),
                                           voff[i],
                                           soff,
                                           fx.Int32(0),
                                           fx.Int32(0))
        set_alias_scopes(op, al, noal)


def gemm_fix_plan(t, lane, lr, wave, hdim):
    """'xor', padded: an odd `hdim` leaves its slack column in a dword with a real
    one; wave lr % 8, which DMA'd row lr, zeroes that column. (pred, LDS offset)."""
    a = t.a
    h = fx.Int32(hdim)
    pos = (fx.Int32(lr) * fx.Int32(a.row_bytes)
           + (((h >> fx.Int32(3)) ^ fx.Int32(lr % 8)) << fx.Int32(4))
           + (h & fx.Int32(7)) * fx.Int32(2))
    ok = (((h & fx.Int32(1)) == fx.Int32(1)) & (lane < fx.Index(16)) & (lr % 8 == wave))
    return ok, fx.Index(pos) + fx.Index(a.lds_base)


def gemm_fix(t, plan, lds, scopes, buf):
    """Zero the split column of buffer `buf` (after its DMA's vmcnt)."""
    a = t.a
    ok, off = plan
    al, noal = scopes(a.scope_of(buf))

    def ptr():
        # A function, not `lds` itself: the jit body below sees only these.
        return lds.view(off + fx.Index(buf * a.tile_bytes), fx.Int16, 2).llvm_ptr

    @flyc.jit
    def fix():
        if ok:
            llvm.StoreOp(raw(fx.Int16(0)),
                         ptr(),
                         alignment=2,
                         alias_scopes=al,
                         noalias_scopes=noal)
    fix()


# ---------------------------------------------------------------- PermA phases

def gemm_load_plan(t, tid, row_stride_b):
    """Per-thread (global voffsets, LDS write offsets). Threads past the last unit redo one."""
    a = t.a
    if a.layout == 'chunks':
        # Unit = 4 keys x 2 columns -> one 16 B chunk. Affine: one base serves every unit.
        base = []
        wr = []
        for u in range_constexpr(1 if a.affine else a.units_per_thread):
            unit = (tid + fx.Index(u * a.threads)) % a.units
            g = unit // (a.cols // 2)
            cp = unit % (a.cols // 2)
            base.append(fx.Int32(g * 4) * row_stride_b + fx.Int32(cp * 4))
            wr.append(fx.Index(a.lds_base) + (cp // 16) * 1024 + g * 256 + (cp % 16) * 16)
        return base, wr
    if a.unit_cols == 8:
        # Row groups fastest: 16 lanes' A^T writes then cover all 32 banks.
        assert a.units <= a.threads, a.units
        unit = tid % a.units
        g = unit % (a.rows // 4)
        c8 = unit // (a.rows // 4)
        voff = fx.Int32(g * 4) * row_stride_b + fx.Int32(c8 * 16)
        lds_off = fx.Index(a.lds_base) + (c8 * 8 * a.stride + g * 4) * 2
        return voff, lds_off
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
    a = t.a
    if a.layout == 'chunks':
        g_step = a.threads // (a.cols // 2)     # key groups between a thread's units
        out = []
        for u in range_constexpr(a.units_per_thread):
            for kk in range_constexpr(4):
                if a.affine:
                    soff = (fx.Int32(row_start) + fx.Int32(4 * g_step * u + kk)) * row_stride_b
                    base = plan[0][0]
                else:
                    soff = (fx.Int32(row_start) + fx.Int32(kk)) * row_stride_b
                    base = plan[0][u]
                out.append(fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, rsrc_raw, base, soff, 0)))
        return out
    if t.a.unit_cols == 8:
        out = []
        for r in range_constexpr(4):
            soff = (fx.Int32(row_start) + fx.Int32(r)) * row_stride_b
            w4 = Vec(rocdl.raw_ptr_buffer_load(Vec.make_type(4, fx.Int32), rsrc_raw, plan[0], soff, 0))
            out.extend(w4[i] for i in range(4))
        return out
    soff = fx.Int32(row_start) * row_stride_b
    return [fx.Int32(rocdl.raw_ptr_buffer_load(fx.Int32.ir_type, rsrc_raw, off, soff, 0)) for off in plan[0]]


def gemm_stage(t, plan, lds, scopes, regs, buf):
    """VGPRs -> A^T in LDS buffer `buf`: each unit's 4x2 blocks transposed by v_perm."""
    a = t.a
    if a.layout == 'chunks':
        g_step = a.threads // (a.cols // 2)
        sel_lo = fx.Int32(0x05040100)
        sel_hi = fx.Int32(0x07060302)
        al, noal = scopes(a.scope_of(buf))
        for u in range_constexpr(a.units_per_thread):
            w = regs[4 * u:4 * u + 4]
            dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_lo)),
                  fx.Int32(rocdl.perm_b32(w[3], w[2], sel_lo)),
                  fx.Int32(rocdl.perm_b32(w[1], w[0], sel_hi)),
                  fx.Int32(rocdl.perm_b32(w[3], w[2], sel_hi))]
            wr = plan[1][0] + g_step * 256 * u if a.affine else plan[1][u]
            llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                         lds.view(wr + buf * a.tile_bytes, fx.Int32, 16).llvm_ptr,
                         alignment=16,
                         alias_scopes=al,
                         noalias_scopes=noal)
        return
    if a.unit_cols == 8:
        sel = (fx.Int32(0x05040100), fx.Int32(0x07060302))
        al, noal = scopes(a.scope_of(buf))
        for d in range_constexpr(8):
            w = [regs[4 * r + d // 2] for r in range(4)]
            dw = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel[d % 2])), fx.Int32(rocdl.perm_b32(w[3], w[2], sel[d % 2]))]
            llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                         lds.view(plan[1] + (buf * a.tile_bytes + d * a.stride * 2), fx.Int32, 8).llvm_ptr,
                         alignment=8,
                         alias_scopes=al,
                         noalias_scopes=noal)
        return
    sel_lo = fx.Int32(0x05040100)
    sel_hi = fx.Int32(0x07060302)
    for u in range_constexpr(a.units_per_thread):
        w = regs[4 * u:4 * u + 4]
        d0 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_lo)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_lo))]
        d1 = [fx.Int32(rocdl.perm_b32(w[1], w[0], sel_hi)), fx.Int32(rocdl.perm_b32(w[3], w[2], sel_hi))]
        for e, dw in ((0, d0), (1, d1)):
            off = plan[1][2 * u + e] + buf * a.tile_bytes
            al, noal = scopes(a.scope_of(buf))
            llvm.StoreOp(raw(Vec.from_elements(dw, fx.Int32)),
                         lds.view(off, fx.Int32, 8).llvm_ptr,
                         alignment=8,
                         alias_scopes=al,
                         noalias_scopes=noal)


# ---------------------------------------------------------------- read, mma

def _opaque(x):
    """x through a v_mov LLVM cannot see into, so constant offsets fold into the
    ds_read immediate instead of a v_or per read."""
    return fx.Index(fx.Int32(llvm.inline_asm(T.i32, [raw(fx.Int32(x))], 'v_mov_b32 $0, $1', '=v,v')))


def gemm_read_plan(t, lm, lh, lane=None):
    """Per-thread LDS byte offsets of `gemm_read` / `gemm_stream`. On 16x16x16,
    `lm`, `lh` are the lane's row (lane % 16) and 4-key group (lane // 16)."""
    a = t.a
    if isinstance(a, RowA) and a.dma == 'xor':
        # Read t: row lm, chunk 4t + lh, at chunk (4t + lh) ^ (lm % 8); bases alternate by t % 2.
        return [_opaque(fx.Index(a.lds_base) + lm * a.row_bytes + ((fx.Index(4 * e) + lh) ^ (lm % 8)) * 16)
                for e in range(2)]
    if isinstance(a, PermA) and a.layout == 'chunks':
        return _opaque(fx.Index(a.lds_base) + lane * 16)
    if isinstance(a, RowA):
        return [fx.Index(a.lds_base) + lm * a.stride + (fx.Index(2 * s) + lh) * 16 for s in range(t.reads)]
    return fx.Index(a.lds_base) + (lm * a.stride + lh * 4) * 2


def gemm_read(t, plan, lds, buf):
    """A from LDS buffer `buf`: [out_tiles][reads], 16 B (RowA) or 8 B (PermA) each."""
    a = t.a
    elem = elem_type(t.dtype)
    scope = a.scope_of(buf)
    if isinstance(a, RowA):
        v8 = Vec.make_type(8, elem)
        return [[lds.load(off + buf * a.tile_bytes + o * 32 * a.stride, v8, 16, scope) for off in plan]
                for o in range(t.out_tiles)]
    v4 = Vec.make_type(4, elem)
    base = plan if a.buffers == 1 else plan + buf * a.tile_bytes
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


def gemm_read_at(t, plan, lds, buf, i):
    """Read `i` of a streamed GEMM: 16 B from LDS buffer `buf`."""
    a = t.a
    v8 = Vec.make_type(8, elem_type(t.dtype))
    if isinstance(a, RowA):
        return lds.load(plan[i % 2] + (buf * a.tile_bytes + 128 * (i // 2)), v8, 16, a.scope_of(buf))
    return lds.load(plan + (buf * a.tile_bytes + 1024 * i), v8, 16, a.scope_of(buf))


def gemm_mma_at(t, i, a8, b, accs):
    """The two MFMAs of read `i`: RowA steps 2i, 2i+1 into accs[0] (B: packs);
    chunked PermA rows 2i, 2i+1 of A^T into accs[2i], accs[2i+1] (B: one pack)."""
    elem = elem_type(t.dtype)
    # Each half's shuffle right before its MFMA, as the hand-written kernels do.
    if isinstance(t.a, RowA):
        accs[0] = mfma_16x16x16(a8.shuffle(a8, [0, 1, 2, 3]).ir_value(), b[2 * i], accs[0], elem)
        accs[0] = mfma_16x16x16(a8.shuffle(a8, [4, 5, 6, 7]).ir_value(), b[2 * i + 1], accs[0], elem)
    else:
        accs[2 * i] = mfma_16x16x16(a8.shuffle(a8, [0, 1, 2, 3]).ir_value(), b, accs[2 * i], elem)
        accs[2 * i + 1] = mfma_16x16x16(a8.shuffle(a8, [4, 5, 6, 7]).ir_value(), b, accs[2 * i + 1], elem)


def gemm_stream(t, plan, lds, buf, b, accs=None, ahead=2):
    """A streamed GEMM on 16x16x16: each read issued `ahead` reads before its MFMAs,
    every step pinned (LLVM alone keeps one buffer and waits on every read).
    `accs` None starts from zero. The one phase that orders its own ops."""
    accs = [fx.full(4, 0.0, fx.Float32)] if accs is None else list(accs)

    def use(i, a8):
        gemm_mma_at(t, i, a8, b, accs)

    pipelined(t.reads, ahead, lambda i: Vec(gemm_read_at(t, plan, lds, buf, i)), use)
    return accs


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
            if isinstance(v, list) and v and isinstance(v[0], list):
                # [out_tiles][reads] from gemm_read
                for row in v:
                    out.extend(row)
                sizes.append(tuple(len(row) for row in v))
            elif isinstance(v, list):
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
            elif isinstance(n, tuple):
                rows = []
                for k in n:
                    rows.append(values[i:i + k])
                    i += k
                setattr(self, name, rows)
            else:
                setattr(self, name, values[i:i + n])
                i += n
        return self

    def tiles(self, iteration, j_clean, j_clean_end, n_loop):
        """Run `iteration(j, buf, masked)` over [0, n_loop): masked tiles on both
        sides of [j_clean, j_clean_end), the clean ones unrolled by two so `buf` is a
        constant. `j_clean` must be even."""
        self.restore(run_tiles(self, iteration, j_clean, j_clean_end, n_loop))

    def tile_pairs(self, iteration, j_clean, j_clean_end, n_iter):
        """As `tiles`, but the masked tiles run in pairs too (`n_iter` even)."""
        self.restore(run_tile_pairs(self, iteration, j_clean, j_clean_end, n_iter))


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


@flyc.jit
def run_tile_pairs(ctx, iteration, j_clean, j_clean_end, n_iter):
    """Every loop runs (even, odd) pairs, so `buf` is always a constant; `n_iter`
    is even. Masked pairs bracket the clean ones."""
    init = ctx.carry()
    results0 = init
    for jp, carried in range(0, j_clean // 2, 1, init=init):
        mid = _step(ctx, iteration, jp * 2, carried, 0, True)
        results0 = yield _step(ctx, iteration, jp * 2 + 1, mid, 1, True)
    n_pairs = (j_clean_end - j_clean) // 2
    results1 = results0
    for jp, carried in range(0, n_pairs, 1, init=results0):
        mid = _step(ctx, iteration, j_clean + jp * 2, carried, 0, False)
        results1 = yield _step(ctx, iteration, j_clean + jp * 2 + 1, mid, 1, False)
    j_tail = j_clean + n_pairs * 2
    results2 = results1
    for jp, carried in range(0, (n_iter - j_tail) // 2, 1, init=results1):
        mid = _step(ctx, iteration, j_tail + jp * 2, carried, 0, True)
        results2 = yield _step(ctx, iteration, j_tail + jp * 2 + 1, mid, 1, True)
    return results2
