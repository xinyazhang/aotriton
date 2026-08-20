# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The FlycDecl passive record + its collector (pipeline Stage 2).

The stacked-@ flyc finalizer (specs/finalize.py) partitions an @ati.flyc.* stack
into one FlycDecl, attached to the def as `fn.__ati_node__`. NO build, and — unlike
a triton kernel — NOT routed through describe(): describe() validates that specs
claim every parameter of a known (AST-parsed) signature exactly once, and a flyc
description has no such signature (the hsaco kernarg ABI is declared by the
@ati.tensor/@ati.scalar stack itself, not introspected from a def). Collected
passively instead, like AffineDecl: the disable predicate, the cite, the module
path, the placeholder function, the registered hints dataclass, and the
tensor/scalar specs as an inert list — nothing here is built or validated.
`aotriton.flyc_compile` (Phase 1's only consumer) reads `module_path` and `hints()`.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .node import AtiNode
from ..ast_params import find_one_function, has_decorator_attr, collect_params

if TYPE_CHECKING:
    from ..decorators.disable import DisableSpec
    from ..decorators.cite import CiteSpec


@dataclass(slots=True, kw_only=True)
class FlycDecl(AtiNode):
    """Passive record of an @ati.flyc stack (a flyc kernel's "object file"): the
    marker + @ati.flyc.* metadata + the inert tensor/scalar kernarg-ABI specs +
    optional @ati.cite/@ati.disable. NO build in Phase 1. Attached to the def as
    `fn.__ati_node__`.

    `cite`/`disable` stay SINGULAR. A kernel may carry at most one of each --
    that is true of a triton kernel too, and `collect_flyc_decl` enforces it
    below. The builder pipeline (`resolve_cites`, `build_kernel`) reads the
    list-valued `cites`/`disables` off a Triton `KernelSpec`, but that is the
    BUILDER's contract, not a fact about a description; the singular-to-list
    adaptation happens once, in the linker's `_flyc_kernel_spec`, rather than
    being pushed back into the declaration where it would misstate what a user
    can write.

    `overrides`/`dtype_vars`/`tune`/`params` are genuinely plural (or genuinely
    optional) and exist for the builder pipeline.

    `kernel`/`params` are populated from an AST walk of `module_path` (the
    vendored kernel file) that finds the unique `@flyc.kernel`-decorated
    function -- the same operation `@ati.source` performs for a Triton kernel,
    reusing the same `ast_params`/`introspect` machinery rather than keeping a
    second AST walker (`ir/flyc/kdesc.py`'s `_real_param_order` did the same
    walk a second time and has since been deleted -- `iter_launch_arguments`
    reads `self._built.arguments`, sourced from these `params` via
    `build_kernel`). `dtype_vars`/`tune` remain inert placeholders: nothing
    populates or consumes them yet."""

    name: str                          # the placeholder def's __name__
    module_path: Path                  # resolved vendored-kernel-file path
    desc_path: Path                    # the DESCRIPTION module's own file (Task 5c's DESC column)
    functionals_of: str                # operator NAME this kernel's functionals come from (Task 5a)
    hints_cls: type | None             # the @ati.flyc.hints dataclass, or None
    fn: object                         # the placeholder def itself (the builder)
    kernel: object = None              # KernelStub for the AST-located @flyc.kernel def
    tensors: list = field(default_factory=list)     # inert list[TensorSpec]
    scalars: list = field(default_factory=list)     # inert list[ScalarSpec]
    cite: 'CiteSpec | None' = None
    disable: 'DisableSpec | None' = None
    overrides: list = field(default_factory=list)   # list[Override], unused until a later step
    dtype_vars: list = field(default_factory=list)  # list[ChoiceVar], unused until a later step
    tune: object = None                             # TuneSpec | None, unused until a later step
    params: list = field(default_factory=list)      # list[ParamSpec], real kernel signature order

    def hints(self):
        """A default-constructed instance of the registered hints dataclass, or
        None if @ati.flyc.hints was never applied."""
        if self.hints_cls is None:
            return None
        return self.hints_cls()

    @property
    def param_names(self):
        """Mirrors KernelSpec.param_names (specs/kernel.py) -- build_kernel's
        `_build_axes` reads this off either spec type duck-typed."""
        return [p.name for p in self.params]


def _flyc_kernel_stub(module_path, name):
    """AST-parse `module_path` (the vendored kernel file) for the unique
    `@flyc.kernel`-decorated function and wrap it as a `KernelStub` -- the same
    non-importing stand-in `@ati.source` builds for a Triton kernel
    (decorators/source.py). Unlike a Triton kernel (looked up by NAME,
    top-level only), the flyc kernel def is looked up by DECORATOR and may sit
    in a nested scope, so this walks every scope (`walk=True`) and requires the
    match to be unique."""
    from ..decorators.source import KernelStub

    path = Path(module_path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    what = f'flyc kernel {name!r}: {path}'
    fn = find_one_function(tree, lambda n: has_decorator_attr(n, 'kernel'),
                           walk=True, what=what)
    params = collect_params(fn, what=what)
    return KernelStub(fn.name, params, str(path))


def collect_flyc_decl(placeholder, specs):
    """Partition an @ati.flyc stack into a passive FlycDecl (no build, no
    describe() validation).

    Accepts the same spec kinds as specs/finalize.py's `_partition` (the Triton
    stacked-@ collector). `overrides`/`dtype_vars` accumulate as lists because
    a stack may legitimately carry several; `cite`/`disable` stay singular and
    are asserted at most once, because a stack may not."""
    from ..decorators.flyc import FlycKernelSpec, FlycHintsSpec
    from ..decorators import TensorSpec, ScalarSpec, DisableSpec, CiteSpec, ChoiceVar
    from ..ir import Override

    marker = None
    hints_cls = None
    tensors = []
    scalars = []
    overrides = []
    dtype_vars = []
    cite = None
    disable = None
    for s in specs:
        if isinstance(s, FlycKernelSpec):
            assert marker is None, 'multiple @ati.flyc.kernel markers in one stack'
            marker = s
        elif isinstance(s, FlycHintsSpec):
            assert hints_cls is None, 'duplicate @ati.flyc.hints on one kernel'
            hints_cls = s.hints_cls
        elif isinstance(s, TensorSpec):
            tensors.append(s)
        elif isinstance(s, ScalarSpec):
            scalars.append(s)
        elif isinstance(s, ChoiceVar):
            dtype_vars.append(s)
        elif isinstance(s, CiteSpec):
            assert cite is None, 'multiple @ati.cite on one @ati.flyc stack'
            cite = s
        elif isinstance(s, Override):
            overrides.append(s)
        elif isinstance(s, DisableSpec):
            assert disable is None, 'multiple @ati.disable on one @ati.flyc stack'
            disable = s
        else:
            raise AssertionError(
                f'unexpected spec {s!r} in an @ati.flyc stack; flyc kernels accept '
                f'@ati.flyc.*, @ati.tensor/@ati.scalar, @ati.cite, @ati.disable, '
                f'@ati.derives and @ati.type_var/@ati.scalar_var only')
    assert marker is not None, '@ati.start flyc path without an @ati.flyc.kernel marker'
    assert marker.functionals_of is not None, (
        f'@ati.flyc.kernel on {placeholder.__name__!r} is missing functionals_of=; '
        f'Phase 1 has no other route to a functional space (flyc declares none of '
        f'its own and is not an operator backend yet -- PLAN-PHASE1.md Task 5a)')
    # The DESCRIPTION module's own file (e.g. modules/flash/aot/flyc_attn_fwd.py),
    # NOT marker.module_path (the vendored kernel file under modules/flash/flyc/).
    # This is what codegen/root.py's Fly.compile DESC column feeds to
    # `aotriton.flyc_compile <desc_path> --kernel_name ...` (Task 5c).
    desc_path = Path(inspect.getfile(placeholder)).resolve()
    from ..introspect import kernel_params
    stub = _flyc_kernel_stub(marker.module_path, placeholder.__name__)
    return FlycDecl(name=placeholder.__name__, module_path=marker.module_path,
                    desc_path=desc_path, functionals_of=marker.functionals_of,
                    hints_cls=hints_cls, fn=placeholder, kernel=stub,
                    params=kernel_params(stub), tensors=tensors,
                    scalars=scalars, overrides=overrides, dtype_vars=dtype_vars,
                    cite=cite, disable=disable)
