# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash attn_fwd FlyDSL backend (gfx1201).

DEMO / DESIGN SKETCH — not wired into flash_entry.py yet. The `ati.flyc.*`
decorator namespace this file uses does not exist; it is proposed here so the
authoring surface can be reviewed against a real kernel before it is built. See
`modules/flash/flyc/PLAN.md` Part 6 for the implementation plan.

A FlyDSL backend is a THIRD kind, and it takes one half from each of the two ATI
already has:

    triton (@ati.source)   compiled during the build, ATI owns the perf space,
                           1:1 argument names, functional axes ARE kernel params
    aiter  (@ati.affine.*) prebuilt .co, no perf space, no axes of its own —
                           inherits the operator's and filters them
    flyc   (@ati.flyc.*)   compiled during the build (from triton), inherits and
                           filters the operator's axes (from aiter), and — unique
                           to it — dispatches an hsaco whose kernarg list is NOT
                           the operator's

Three consequences, each handled below:

1. **The tuning table is programmatic.** `fmha_tuning_gfx1201.resolve_knobs()` is
   the sole producer of a schedule, so there is no `@ati.tune.schema`, no
   `@ati.tune.configs`, and no perf axes — ATI enumerates no perf variants here.

   That is NOT the same as one hsaco per functional. Today the count is one, but
   only because the shipped schedule targets long sequences and short ones fall to
   the Triton backend; a seqlen-dependent FlyDSL tuner will emit several. Nothing
   downstream may assume the count — hsacos are packed into per-functional .aks2
   archives exactly as Triton's are, which is also what keeps Triton's autotune
   code generator reusable for flyc later. See PLAN.md 6.2.

2. **No functional axes are declared here.** They belong to the operator (owned by
   the default triton backend); this backend inherits them and narrows with
   `@ati.disable`, exactly as `aiter_fwd.py` does. `flyc_attn_fwd` then reads
   `choices.NAME` (a `ChoiceView`) — the OPERATOR's choices, parsed from
   `--signature` text by the driver — and maps them to builder knobs. This is
   also why the axes cannot
   be re-declared as `@ati.scalar`: they are not arguments of the flyc kernel at
   all, they are build-time Python values.

3. **The kernarg list is declared, because nothing else can supply it.** aiter hands
   the params struct to a C++ cookie and never names a kernarg; triton gets its list
   from the kernel signature. flyc dispatches the hsaco directly, and the hsaco's
   layout comes from `flash_attn_func_aiw_kernel`, so the wiring below is the
   description's real payload.
"""

from dataclasses import dataclass, asdict

import aotriton.template_instantiation as ati
from ._common import flash_disabled, check_value


@dataclass
class FlycFwdHints:
    """Tuning inputs the builder may read that are NOT functional axes.

    Registered with `@ati.flyc.hints` — namespaced with the rest of the FlyDSL-specific
    surface rather than added to `ati.tune.*`, which is the SHARED tuning vocabulary
    (schema / configs / binning / fallback all feed the LUT and the tuning DB; this
    feeds one builder and nothing else). `ati.affine.*` is the same pattern for aiter.
    See PLAN.md 6.9.1.

    FlyDSL would call these part of the **problem**, opposite its **schedule**
    (`fmha_tuning_gfx1201.py:410`: *"a caller states a problem, the tuning policy
    answers with a schedule"*) — worth knowing if you arrive from that file. Strictly
    the RUNTIME half of it: dtype / head_dim / causal are part of FlyDSL's problem too,
    but in ATI they are functional axes and arrive on `f`. `f` and `hints` together are
    what `FmhaInputMetadata` keeps in one dataclass.

    Defaults are what Phase 1 always passes, and they must reproduce the schedule the
    kernel ships today: `resolve_knobs` reads none of these, so every build is
    hint-independent until FlyDSL's tuner grows a seqlen dependence.
    """
    seqlen_q: int = 0        # 0 = unknown/any; a real value once the tuner uses it
    seqlen_k: int = 0
    num_heads: int = 0
    batch: int = 0


# The compiled tile ladder, from fmha_tuning_gfx1201._BLOCK_DMODEL_LADDER. A literal,
# not an import: the generator parses this module and has no flydsl. `flyc_build`
# (build time) does import flydsl, and `resolve_knobs` re-validates the tile, so a
# drift here fails loudly at build rather than silently emitting the wrong kernel.
FLYC_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512})


def _flyc_fwd_disabled(f):
    """Everything this backend cannot serve, in one predicate.

    Arch lives here rather than in a `@ati.flyc.arch([...])` declaration: it is one
    more exclusion among several, and splitting it across two mechanisms means two
    places to look when a functional unexpectedly has no flyc kernel. (`aiter_fwd.py`
    uses @ati.affine.arch; that predates `f.arch` being available to predicates.)
    """
    if f.arch != 'gfx1201':
        return True
    if flash_disabled(f):
        return True
    # f16/bf16 WMMA only.
    if '*fp32' in check_value(f, ['Q']):
        return True
    # Off-ladder head dims are rejected outright by `resolve_knobs`, not rounded:
    # rounding is the *interface*'s job, because it also has to arrange the runtime
    # extent and the padded-head contract that make the rounding safe.
    if check_value(f, ['BLOCK_DMODEL']) not in FLYC_HEAD_DIMS:
        return True
    return False


@ati.start
@ati.disable(when=_flyc_fwd_disabled)
# `cite` fills the GAPS: any argument below that this description does not fully
# claim (dtype variables, strideless operands) is cloned from the triton kernel by
# apparel name, so the two backends cannot drift on a shared operand's type.
@ati.cite('op_attn_fwd.triton.attn_fwd')
#
# --- the kernarg ABI, in `flash_attn_func_aiw_kernel` order ------------------
#
# The kernel, NOT the Python launcher: the launcher is host code that the C++ shim
# replaces (it computes the grid and marshals arguments), while the @flyc.kernel
# def is what fixes the kernarg layout. The two lists differ — the launcher passes
# its `batch_size` into the kernel's `num_seqlens` slot, and carries a `stream` the
# kernel does not have.
#
# The order is frozen and load-bearing: fmha_common_gfx1201.py:137 records that
# switching these pointers to fx.Tensor would grow the kernarg segment from 268 to
# 428 bytes and shift the offset of every argument after the first.
#
# TODO(fx.Tensor): these declarations are also what the AOT compiler will read to
# synthesise operand descriptors with the right dtype and rank, which is what would
# let a future launcher take `fx.Tensor` instead of `fx.Pointer`. Not supported yet
# and deliberately so -- the tensor kernarg ABI is not pinned down (each fx.Tensor
# adds a 40-byte by-value memref descriptor interleaved after its pointer, and what
# those bytes contain is unverified). See jit2aot.md, "fx.Tensor and the hsaco ABI".
# Raw pointers + explicit strides stay the position while FlyDSL's ABI settles.
#
# `rank=4` with only THREE strides is the declaration that the last dimension is
# unit-stride and has no argument. FlyDSL requires stride(3) == 1 and reads no
# D-axis stride at all, so unlike the triton kernel (where `stride_qk` exists as a
# parameter and `contiguous=-1` names it) there is nothing here to point at.
# Deriving it from the shortfall — trailing `rank - len(strides)` dims are
# implicitly unit, constexpr 1, not passed — needs no new keyword and makes the
# rank annotation carry its own justification. `contiguous=-1` would be actively
# WRONG here: it indexes the matched stride list, which has three entries, so it
# would mark `stride_q_seq` as the unit stride.
@ati.tensor('Q',    'T_io', rank=4, strides='stride_q_*', wires_to='Q')
@ati.tensor('K',    'T_io', rank=4, strides='stride_k_*', wires_to='K')
@ati.tensor('V',    'T_io', rank=4, strides='stride_v_*', wires_to='V')
@ati.tensor('O',    'T_io', rank=4, strides='stride_o_*', wires_to='Out')
# L is always compact: the kernel derives both pitches from LSE_LAYOUT, num_head_q
# and the token count, so it has no stride arguments by design.
@ati.tensor('L',    '*fp32:16', rank=2, wires_to='L')
@ati.tensor('Bias', 'T_io', rank=4, strides='stride_b?', wires_to='B')
# varlen seqinfo: FlyDSL splits each side into a (base, stride) pair rather than
# AOTriton's cu_seqlens/seq_strides naming; the operands are the same tensors.
@ati.tensor('seqinfo_q0', '*i32:16', rank=1, wires_to='cu_seqlens_q')
@ati.tensor('seqinfo_q1', '*i32:16', rank=1, wires_to='seq_strides_q')
@ati.tensor('seqinfo_k0', '*i32:16', rank=1, wires_to='cu_seqlens_k')
@ati.tensor('seqinfo_k1', '*i32:16', rank=1, wires_to='seq_strides_k')
#
# --- the two arguments that are not a rename ---------------------------------
#
# See the block comment below the ABI for why these are context helpers rather
# than renames or inline expressions.
#
# `varlen_bits` is FlyDSL's layout descriptor; AOTriton has no such operand, and
# this is also where the MODE half of `Num_seqlens` lands.
@ati.scalar('varlen_bits', 'i32', wires_to=ati.context_helper('flyc_varlen_bits'))
# `batch_size` and `num_seqlens` are a PAIR, and neither is a rename.
#
# FlyDSL's contract (fmha_abi_gfx1201.varlen_args docstring): `batch_size` is
# q.size(0) always, whatever the layout; `num_seqlens` is how many sequences are
# packed into a 1HTD tensor, and 0 when nothing is packed. Dense is (B, 0);
# packed with N sequences is (1, N). The kernel then branches on the pair:
#     nseq_idx = (num_seqlens != 0).select(num_seqlens, batch_size)
#
# AOTriton spells the same information differently: a `Batch` operand plus a
# SIGNED three-way `Num_seqlens` (>0 packed count, 0 dense, <0 BHSD-padded
# varlen). So:
#
#   batch_size   <- params.Q->size(0), NOT params.Batch. Under packed varlen Q
#                   is 1HTD, so q.size(0) is 1 while Batch is not.
#   num_seqlens  <- max(Num_seqlens, 0). A negative Num_seqlens is padded, not
#                   packed, so FlyDSL wants 0 and the layout rides in
#                   varlen_bits. VERIFY against flyc_varlen_bits: the two
#                   helpers must agree on how the padded case is encoded.
#
# Getting this pair wrong fails SILENTLY -- FlyDSL's own docstring: "it launches
# N programs over a tensor whose batch axis is 1, and every one of them
# addresses a plausible row." Assert in the shim, do not rely on the helper.
@ati.scalar('batch_size', 'i32', wires_to=ati.context_helper('flyc_batch_size'))
@ati.scalar('num_seqlens', 'i32', wires_to=ati.context_helper('flyc_num_seqlens'))
#
# --- plain renames ------------------------------------------------------------
@ati.scalar('max_seqlen_q', 'i32', wires_to='Max_seqlen_q')
@ati.scalar('max_seqlen_k', 'i32', wires_to='Max_seqlen_k')
@ati.scalar('window_left',  'i32', wires_to='Window_left')
@ati.scalar('window_right', 'i32', wires_to='Window_right')
# PRNG: 1:1 with the triton kernel as of FlyDSL 971dce48 ("the philox seed becomes a
# pointer, and the forward reports what it drew") + 53334317 ("the philox offset
# splits into a pointer and an immediate"). These were the two transient mismatches
# in the previous revision of this file; both are gone, and neither needs a helper.
# Declared one per line, in kernel order, NOT grouped. Grouping the four *u64
# pointers into one @ati.tensor([...]) reads better but silently reorders:
# philox_offset2 is a scalar sitting BETWEEN them in the kernel signature, so the
# grouped form pushed it to the end and the declared kernarg order was wrong.
@ati.tensor('philox_seed_ptr', 'T_u64', rank=0)
@ati.tensor('philox_offset1', 'T_u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
@ati.tensor('philox_seed_output', 'T_u64', rank=0)
@ati.tensor('philox_offset_output', 'T_u64', rank=0)
#
# --- the two dropout arguments that are not a rename --------------------------
@ati.scalar('idropout_p',    'i32',  wires_to=ati.context_helper('flyc_idropout_p'))
@ati.scalar('dropout_scale', 'fp32', wires_to=ati.context_helper('flyc_dropout_scale'))
#
# --- plain renames, continued -------------------------------------------------
@ati.scalar('num_head_q',   'i32',  wires_to='Num_head_q')
@ati.scalar('num_head_k',   'i32',  wires_to='Num_head_k')
@ati.scalar('hdim_qk',      'i32',  wires_to='Hdim_qk')
@ati.scalar('hdim_vo',      'i32',  wires_to='Hdim_vo')
@ati.scalar('sm_scale_arg', 'fp32', wires_to='Sm_scale')
#
# --- why `ati.context_helper` and not an inline expression --------------------
#
# Four arguments above are not renames of an operator operand. They need host-side
# code, and the place that code already goes is a member function on the generated
# CONTEXT class, hand-implemented in modules/flash/csrc/:
#
#     wires_to=ati.context_helper('flyc_num_seqlens')
#       -> declares  int32_t flyc_num_seqlens() const;  on FlycAttnFwdContext
#       -> author implements it in modules/flash/csrc/flyc_attn_fwd.cc
#
# This is not a new concept. `dim3 grid_calculator() const;` is declared in the
# generated context struct (codegen/template/shim.h:73) and implemented by hand as
# `AttnFwdContext::grid_calculator()` in modules/flash/csrc/attn_fwd.cc — same class,
# same namespace (AOTRITON_NS::v3::flash), same split. `context_helper` only lets a
# description declare MORE of them instead of the set being hardwired.
#
# NO ARGUMENTS, because the context already carries everything: `params` (the whole
# operator params struct) and the selected perf fields are members. grid_calculator
# demonstrates the range — it reads params->Num_seqlens, params->Q->size(1),
# params->Batch, this->BLOCK_M and this->PERSISTENT_TYPE, and takes nothing.
#
# The return type is not declared twice: it comes from the @ati.scalar type on the
# same line, so `'i32'` fixes the signature as `int32_t`.
#
# Rejected alternative — `wires_to=ati.expr('<C++ expression>')`, i.e. a C++ string
# in the Python description:
#   * a typo is caught only when the generated .cc compiles, and the error points at
#     generated code; a missing context helper is a LINK error naming the symbol
#   * a string cannot be unit-tested, stepped through, or given a comment explaining
#     the three-way Num_seqlens encoding at the point it is decoded
#   * anything past one expression (abs(), the varlen bit packing) does not fit
#   * two mechanisms for host-side code — grid_calculator in csrc, expressions in the
#     description — means two places to look
# The cost is that trivial cases (dropout_scale is one divide) also become functions.
# Uniformity is worth it: one mechanism, one file, one place a reviewer looks.
#
# THE LAUNCH GRID lands in the same place: `grid_calculator()` is already the hook,
# and FlycAttnFwdContext implements it as
# `(num_head_q, cdiv(max_seqlen_q, BLOCK_M), batch)` with block `(BLOCK_SIZE, 1, 1)`.
# Open question: BLOCK_M/BLOCK_SIZE are FlyDSL KNOBS, and flyc declares no perf axes,
# so they are not perf fields on the context. They have to reach it some other way —
# the JSON sidecar folded into the compiled-in metadata is the obvious candidate.
@ati.flyc.hints(FlycFwdHints)
@ati.flyc.kernel('../flyc/flash_attn_func_gfx1201_aiw.py', functionals_of='op_attn_fwd')
def flyc_attn_fwd(choices, hints):
    """Build one hsaco for the functional described by `choices`, optimized
    for `hints`.

    Executed by `aotriton.flyc_compile` at build time, in a venv that has flydsl —
    never by the generator, which only reads the decorators above. Returns
    `(built, sidecar)`: `built` is whatever the builder returns (the driver drives
    it to a code object), and `sidecar` is a JSON-serialisable dict of whatever
    this description wants recorded alongside the hsaco. Here that is
    `asdict(knobs)` — `resolve_knobs` is the only place `block_m` (and everything
    else `flyc_compile`'s Task 3d output needs bar `block_size`, which the driver
    recovers itself from the compiled IR's `known_block_size`) is known, and it
    would otherwise go out of scope on return: `built` is the `_launch` closure,
    which exposes only `compile` and the `varlen_*` helpers, not `knobs`.

    TWO objects, because they are two kinds of fact (PLAN.md 6.9):

      choices  what the kernel must SUPPORT: a `ChoiceView` (`ir/choices.py`)
               over the compile-time identity. Two call sites hand this
               function two different backings, and the function reads neither
               one directly -- only the interface: the generator has a linked
               `ir.Functional` and passes the real thing, `f.choices`
               (`FunctionalChoiceView`); the driver (`flyc_compile.py`) has
               only `--signature` text in a separate process with no linked
               IR, and passes a `MappingChoiceView` over the parsed dict (see
               `jit2aot.md` "Correction 2" for why a `Functional` cannot be
               rebuilt from that text). `choices.NAME` reads a choice variable
               by attribute; `choices.arg('Q')` reads a real argument name
               that is not one (`T_io` is the variable governing `Q`).
      hints    what the kernel should be OPTIMIZED FOR. Declared by
               `@ati.flyc.hints` above. Not axes, and deliberately so —
               `seqlen_q` is a tune BINNING dimension
               (`@ati.tune.binning(Max_seqlen_q=...)`), and promoting it to an
               axis would multiply the functional space and the godel
               numbering for every backend in order to serve one.

    Deliberate asymmetry with `_flyc_fwd_disabled` above, which takes a real
    `ir.Functional` and reads `f.arch`: disable predicates run
    GENERATOR-side, where the linked IR exists; this function runs DRIVER-side,
    where only `--signature` text exists. `choices` is not a `Functional` and
    must not grow into one -- if a build body ever needs arch, it arrives as an
    explicit third parameter from `--target`, not smuggled into `choices`.

    Today `resolve_knobs` reads no field of `hints` — FlyDSL's tuner is currently
    seqlen-independent, so the count stays at one per functional and every field sits
    at its default. The parameter exists so that stops being an API change — and the
    packaging is already N-capable, so neither is the artifact layout.

    `resolve_knobs`, NOT `plan`: `plan()` is the JIT entry point, which takes a
    *real* head_dim and rounds it up the ladder, deriving `padded_head` on the way.
    AOT already knows the tile — it IS `BLOCK_DMODEL` — and `PADDED_HEAD` is its own
    functional axis, so `plan()` would silently re-derive an axis the operator has
    already fixed. The builder's own keyword front end draws the same distinction.
    """
    # ONLY the flydsl-free tuning module at call time. The FlyDSL-bearing import
    # lives inside `build()` below, so the code generator -- which calls this
    # function but never the callable -- never imports flydsl.
    from fmha_tuning_gfx1201 import FmhaInputMetadata, FmhaKnobs, resolve_knobs

    tile = choices.BLOCK_DMODEL
    meta = FmhaInputMetadata(
        # `num_heads` reaches the emitted kernel ONLY through STRIDE_TOKEN, which is
        # read exclusively under STRIDES_CONSTEXPR — a dense-only diagnostic arm the
        # AOT path never selects (see the knob below). Pinning it to 1 keeps it out
        # of the functional space. Asserted, not assumed: this is a property of
        # today's kernel, not a contract it owes us.
        num_heads=1,
        head_dim=tile,
        # FlyDSL's causal_type IS AOTriton's CAUSAL_TYPE (0 none / 1 top-left /
        # 2 bottom-right / 3 window), and the kernel only ever emits {0, 3} — the
        # same pair the operator's CAUSAL_TYPE axis offers. 1:1, no mapping.
        causal=choices.CAUSAL_TYPE != 0,
        causal_type=choices.CAUSAL_TYPE,
        dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
        bias=bool(choices.BIAS_TYPE),
        dropout=bool(choices.ENABLE_DROPOUT),
    )
    # Supply FmhaKnobs to resolve_knobs to make sure knobs.block_dmodel align with choices.BLOCK_DMODEL
    knobs = resolve_knobs(meta, FmhaKnobs(
        block_dmodel=tile,
        padded_head=choices.PADDED_HEAD,
        # AOT cannot bake strides: one binary must serve every layout. This is also
        # what makes `num_heads` above irrelevant to the emitted code.
        strides_constexpr=False,
    ))
    assert not knobs.strides_constexpr, \
        'num_heads=1 is only safe while STRIDE_TOKEN stays behind strides_constexpr'

    def build():
        """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
        so ONLY `aotriton.flyc_compile` (run by ninja) may call this."""
        from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module_primary
        return build_flash_attn_func_aiw_module_primary(meta, knobs)

    return build, asdict(knobs)
