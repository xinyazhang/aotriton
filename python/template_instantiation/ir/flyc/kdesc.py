# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelDescription — the codegen-facing IR for a flyc (FlyDSL-compiled) kernel.

Mirrors `ir/affine/kdesc.py`'s shape: a minimal `Interface` implementer that owns
no functional space of its own (PLAN.md 6.3) — a flyc kernel inherits and filters
the operator its `functionals_of=` names. Built by the linker
(`codegen/linker.py:_build_flycs`, Task 5a) from a `FlycDecl`, which resolves that
name against the already-built operators and passes the Operator object in as
`functionals_source`.

`_axes_overrides` / `axes_multi` delegate to `functionals_source` so
`Interface.gen_functionals` enumerates exactly the axes the operator does, but
every yielded `Functional` still carries `meta_object=self` (this flyc kdesc, not
the operator) — that is what gives filepack paths / hsaco entry names / the
`Fly.compile` KERNEL_NAME column this kernel's own NAME/FAMILY rather than the
operator's (PLAN-PHASE1.md Task 7a: "Give flyc its own zip ... keyed by the
description's name").
"""

from ..interface import Interface


class KernelDescription(Interface):
    """A flyc kernel built from the @ati.flyc.* stacked form. ATI-native, like
    AffineKernel: subclasses the ATI Interface base (identity surface) directly,
    no functional space of its own (inherits the operator's via
    `functionals_source`), no perf space (`ir/flyc/ksignature.py` leaves perf/copt
    sections empty — the FlyDSL tuning model is unsettled, PLAN-PHASE1.md 0c)."""

    CODEGEN_MODULE = 'flyc'
    TUNE_NAME = None
    FILE_PFX = 'flyc'
    ENUM_PREFIX = 'kFlyc_'
    is_tunable = False

    def __init__(self, *, name, family, module_path, disable=None,
                functionals_source=None):
        self.NAME = name
        self.FAMILY = family
        self.MODULE_PATH = module_path
        self._disable = disable                        # DisableSpec | None
        # The Operator this kernel's `functionals_of=` names, resolved by the
        # linker (Task 5a). None only transiently, between __new__ and the
        # linker's assignment -- every kdesc actually handed to the generator has
        # this set (collect_flyc_decl asserts functionals_of is present).
        self._functionals_source = functionals_source
        self.desc_path = None                          # set by the linker (Task 5c's DESC column)

    @property
    def perf_cfields(self):
        return []

    def _axes_overrides(self):
        """(axes, overrides) to enumerate over -- the referenced operator's own
        (PLAN.md 6.3). `Interface.gen_functionals` uses these but still stamps
        `meta_object=self` on every yielded Functional (see module docstring)."""
        assert self._functionals_source is not None, (
            f'flyc kernel {self.NAME!r} has no functionals_source; the linker must '
            f'resolve functionals_of= before gen_functionals is usable '
            f'(PLAN-PHASE1.md Task 5a)')
        return self._functionals_source._axes_overrides()

    @property
    def axes_multi(self):
        """Delegated to functionals_source: `Functional.compact_choices` /
        `.unified_signature` (ir/functional.py) read `meta_object.axes_multi`, and
        meta_object is THIS kdesc once `_axes_overrides` is wired in (see above)."""
        assert self._functionals_source is not None, (
            f'flyc kernel {self.NAME!r} has no functionals_source '
            f'(PLAN-PHASE1.md Task 5a)')
        return self._functionals_source.axes_multi

    def list_functional_params(self):
        return []

    @property
    def func_cfields(self):
        # The kernarg ABI is not the operator's params struct (PLAN.md's third
        # consequence) — Phase 1 declares no struct contribution. Phase 2's
        # wires_to consumption is what would populate this.
        return []

    def is_functional_disabled(self, functional):
        if self._disable is None:
            return False
        return self._disable.when(functional)
