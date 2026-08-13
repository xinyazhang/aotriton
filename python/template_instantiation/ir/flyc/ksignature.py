# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelSignature — the per-compiled-instance signature of a flyc kernel.

A thin, STANDALONE class calling `ir/lib/naming.py` directly — it does NOT reuse or
subclass `ir/triton/ksignature.py`'s `KernelSignature`. That class carries Triton
vocabulary (`num_warps` / `num_stages` / `waves_per_eu`, `COMPILER_OPTIONS`, the
gfx1250 double-warps workaround) that is specific to Triton's autotune model and
does not apply here.

flyc enumerates no perf variants in Phase 1 (PLAN-PHASE1.md 0c/5a): the FlyDSL
tuning model is unsettled, and the programmatic `resolve_knobs` builder is a
candidate that looks nothing like a psel/copt grid. Rather than guess, both
sections are permanently empty for now — this keeps the hsaco entry-name /
archive shape identical to Triton's (so later phases can reuse Triton's autotune
code generator) without committing to what a flyc perf/copt vocabulary would be.
"""

from functools import cached_property

from ..lib import naming as lib_naming


class KernelSignature:
    """The perf + compiler-option signature of one compiled flyc kernel instance
    (one Functional). `perf_section` / `copt_section` are always empty strings —
    a deliberate deferral (see module docstring), not a claim that flyc has no
    perf; promote either to a real vocabulary here, not by borrowing Triton's,
    when the FlyDSL tuning model settles."""

    def __init__(self, f: 'Functional'):
        self._functional = f

    @property
    def perf_section(self) -> str:
        return ''

    @property
    def copt_section(self) -> str:
        return ''

    @cached_property
    def hsaco_entry_name(self) -> str:
        return lib_naming.entry_name(self._functional,
                                     perf=self.perf_section,
                                     copt=self.copt_section)

    def blake2b_hash(self, package_path):
        return lib_naming.blake2b_hash(package_path, self.hsaco_entry_name)
