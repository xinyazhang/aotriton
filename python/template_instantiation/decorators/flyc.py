# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `ati.flyc.*` decorator surface (FlyDSL-compiled kernels).

A flyc kernel is a THIRD kind of backend, between triton (`@ati.source`, compiled
during the build, ATI owns the perf space) and affine (`@ati.affine.*`, prebuilt
`.co`, no perf space): it is compiled during the build (from a FlyDSL description),
inherits and filters the operator's functional axes, and dispatches an hsaco whose
kernarg list is NOT the operator's — so the kernarg ABI has to be declared, which
nothing else can supply. Its description uses the stacked-@ form:

    @ati.start
    @ati.disable(when=_flyc_fwd_disabled)
    @ati.cite('op_attn_fwd.triton.attn_fwd')      # fills argument-type GAPS
    @ati.tensor('Q', 'T_io', rank=4, strides='stride_q_*', wires_to='Q')
    ...
    @ati.scalar('varlen_bits', 'i32', wires_to=ati.context_helper('flyc_varlen_bits'))
    @ati.flyc.hints(FlycFwdHints)                 # optimization-input dataclass
    @ati.flyc.kernel()                            # innermost marker, no path (item D)
    def flyc_attn_fwd(arch, choices, hints):
        ...
        build.flyc_source = 'flash_attn_func_gfx1201_aiw.py'   # relative to modules/flash/flyc/
        build.flyc_kernel_name = 'flash_attn_func_aiw_kernel'  # the @flyc.kernel def's own name
        return build, sidecar

These produce passive spec records; specs/flyc.py collects them into a FlycDecl.
Phase 1: no build, no codegen — the description exists so `aotriton.flyc_compile`
(python/flyc_compile.py) can drive it out of process. See
`modules/flash/flyc/PLAN.md` Part 6 / `PLAN-PHASE1.md` Task 4.
"""

from __future__ import annotations

from ..specs.base import StackedSpec


# --- spec records (callable -> accumulate onto the placeholder def) ----------


class FlycKernelSpec(StackedSpec):
    """@ati.flyc.kernel: the innermost marker that makes the def a
    flyc-kernel description (the flyc analogue of @ati.affine.aiter_asm /
    @ati.source).

    Carries NO path (item D). The vendored kernel FILE the description's
    builder actually drives can vary by `arch` -- Phase C's whole point -- so
    it cannot be known here, at decoration time, before any arch is chosen.
    Instead the description's builder function sets it, once it has resolved
    `arch`, as two attributes on the `build` closure it returns:
    `build.flyc_source` (the vendored file, relative to
    `modules/flash/flyc/`) and `build.flyc_kernel_name` (the `@flyc.kernel`
    def's own name inside that file). `codegen/flytune.py` and
    `flyc_compile.py` read those two strings off `build` -- never anything
    eagerly stashed here.

    The operator whose functionals this kernel inherits is NOT declared here.
    It used to be, as `functionals_of=`, when a flyc kernel was reachable no
    other way. It is now an `@ati.backend` of that operator, so the operator
    declares the relationship and `ir/ops/infer.py` binds it — the same
    inference every triton kernel already relies on, and for the reason stated
    there: which operator a kernel serves is the operator's fact, not the
    kernel's."""

    __slots__ = ()

    def __repr__(self):
        return 'FlycKernelSpec()'


class FlycHintsSpec(StackedSpec):
    """@ati.flyc.hints(Dataclass): the optimization-input dataclass a flyc builder
    may read (seqlen_q, seqlen_k, ... — NOT functional axes; see
    modules/flash/aot/flyc_attn_fwd.py's FlycFwdHints for the rationale). Phase 1
    only uses this to construct the defaults object the driver passes to the
    builder; no codegen.

    Lives here, not in decorators/tune.py: `ati.tune.*` is the SHARED tuning
    vocabulary feeding the LUT and the tuning DB; this feeds one description's
    builder and nothing else. `ati.affine.*` is the precedent for a
    backend-specific namespace (PLAN.md 6.9.1)."""

    __slots__ = ('hints_cls',)

    def __init__(self, hints_cls):
        self.hints_cls = hints_cls

    def __repr__(self):
        return f'FlycHintsSpec({self.hints_cls!r})'


# --- public decorator namespace (ati.flyc.*) --------------------------------


def kernel():
    """@ati.flyc.kernel(): innermost marker, no path (item D). See
    FlycKernelSpec for where the vendored file/def name actually come from
    now: the `build` closure the description's builder function returns,
    resolved lazily once `arch` is known."""
    return FlycKernelSpec()


def hints(hints_cls):
    """@ati.flyc.hints(Dataclass): register the builder's optimization-input
    dataclass. See FlycHintsSpec."""
    return FlycHintsSpec(hints_cls)
