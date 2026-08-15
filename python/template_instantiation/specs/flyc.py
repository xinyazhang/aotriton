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

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .node import AtiNode

if TYPE_CHECKING:
    from ..decorators.disable import DisableSpec
    from ..decorators.cite import CiteSpec


@dataclass(slots=True, kw_only=True)
class FlycDecl(AtiNode):
    """Passive record of an @ati.flyc stack (a flyc kernel's "object file"): the
    marker + @ati.flyc.* metadata + the inert tensor/scalar kernarg-ABI specs +
    optional @ati.cite/@ati.disable. NO build in Phase 1. Attached to the def as
    `fn.__ati_node__`."""

    name: str                          # the placeholder def's __name__
    module_path: Path                  # resolved vendored-kernel-file path
    desc_path: Path                    # the DESCRIPTION module's own file (Task 5c's DESC column)
    functionals_of: str                # operator NAME this kernel's functionals come from (Task 5a)
    hints_cls: type | None             # the @ati.flyc.hints dataclass, or None
    fn: object                         # the placeholder def itself (the builder)
    tensors: list = field(default_factory=list)     # inert list[TensorSpec]
    scalars: list = field(default_factory=list)     # inert list[ScalarSpec]
    cite: 'CiteSpec | None' = None
    disable: 'DisableSpec | None' = None

    def hints(self):
        """A default-constructed instance of the registered hints dataclass, or
        None if @ati.flyc.hints was never applied."""
        if self.hints_cls is None:
            return None
        return self.hints_cls()


def collect_flyc_decl(placeholder, specs):
    """Partition an @ati.flyc stack into a passive FlycDecl (no build, no
    describe() validation)."""
    from ..decorators.flyc import FlycKernelSpec, FlycHintsSpec
    from ..decorators import TensorSpec, ScalarSpec, DisableSpec, CiteSpec

    marker = None
    hints_cls = None
    tensors = []
    scalars = []
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
        elif isinstance(s, CiteSpec):
            assert cite is None, 'multiple @ati.cite on one @ati.flyc stack'
            cite = s
        elif isinstance(s, DisableSpec):
            disable = s
        else:
            raise AssertionError(
                f'unexpected spec {s!r} in an @ati.flyc stack; flyc kernels accept '
                f'@ati.flyc.*, @ati.tensor/@ati.scalar, @ati.cite and @ati.disable only')
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
    return FlycDecl(name=placeholder.__name__, module_path=marker.module_path,
                    desc_path=desc_path, functionals_of=marker.functionals_of,
                    hints_cls=hints_cls, fn=placeholder, tensors=tensors,
                    scalars=scalars, cite=cite, disable=disable)
