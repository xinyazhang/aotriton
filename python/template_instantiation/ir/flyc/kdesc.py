# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelDescription — the codegen-facing IR for a flyc (FlyDSL-compiled) kernel.

Mirrors `ir/affine/kdesc.py`'s shape: a minimal `Interface` implementer with no
functional space of its own in Phase 1 — a flyc kernel inherits and filters the
operator's axes (PLAN.md 6.3) rather than owning any, so `gen_functionals` yields
nothing here, same as AffineKernel. Phase 1 does not build this from a FlycDecl at
all (there is currently no discovery path to one — see PLAN-PHASE1.md Task 5a:
flyc is deliberately not registered as an operator backend yet, so `codegen/linker.py`
has nothing to call this from). This class exists so the `ir/flyc/` package shape
Task 0c reserved is filled in, and so Task 5/Phase 2 has a ready adapter to build
from a FlycDecl once the parser can reach one.
"""

from ..interface import Interface


class KernelDescription(Interface):
    """A flyc kernel built from the @ati.flyc.* stacked form. ATI-native, like
    AffineKernel: subclasses the ATI Interface base (identity surface) directly,
    no functional space of its own (inherits the operator's), no perf space
    (`ir/flyc/ksignature.py` leaves perf/copt sections empty — the FlyDSL tuning
    model is unsettled, PLAN-PHASE1.md 0c)."""

    CODEGEN_MODULE = 'flyc'
    TUNE_NAME = None
    FILE_PFX = 'flyc'
    ENUM_PREFIX = 'kFlyc_'
    is_tunable = False

    def __init__(self, *, name, family, module_path, disable=None):
        self.NAME = name
        self.FAMILY = family
        self.MODULE_PATH = module_path
        self._disable = disable      # DisableSpec | None

    @property
    def perf_cfields(self):
        return []

    def gen_functionals(self, build_for_target_arch):
        # A flyc kernel inherits the operator's functional axes; it declares none
        # of its own (PLAN.md 6.3).
        yield from ()

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
