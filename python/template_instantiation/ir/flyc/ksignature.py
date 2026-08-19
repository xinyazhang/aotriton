# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelSignature — the per-compiled-instance signature of a flyc kernel.

A thin, STANDALONE class calling `ir/lib/naming.py` directly — it does NOT reuse or
subclass `ir/triton/ksignature.py`'s `KernelSignature`. That class carries Triton
vocabulary (`num_warps` / `num_stages` / `waves_per_eu`, `COMPILER_OPTIONS`, the
gfx1250 double-warps workaround) that is specific to Triton's autotune model and
does not apply here.

flyc's perf vocabulary is Design B (PLAN-PHASE2.md Task 2): no C struct, no
psel/copt grid choice among candidate images -- every functional resolves to
exactly one hsaco, whose distinguishing knob set (the builder's `sidecar`, all 23
`FmhaKnobs` fields) is carried verbatim in the `#P` section as a schemaless
';'-separated 'k=v' string, parsed at runtime by `class Schemaless`
(`include/aotriton/_internal/schemaless.h`) rather than a generated struct/accessor.
`copt_section` stays permanently empty -- flyc has no compiler-option grid.
"""

from functools import cached_property

from ..lib import naming as lib_naming


def _schemaless_value(v) -> str:
    """Render one FmhaKnobs field as the Schemaless grammar's `value` production
    (PLAN-PHASE2.md Task 2): `0 | -1 | True | False | None | transposed | auto`.
    This is `str(v)`, not `repr(v)` -- for every type FmhaKnobs fields actually
    take (int, bool, None, str) the two agree except for `str`, where `repr`
    would add quotes the measured grammar does not have (`v_lds_layout=transposed`,
    never `v_lds_layout='transposed'`)."""
    return str(v)


class KernelSignature:
    """The perf + compiler-option signature of one compiled flyc kernel instance
    (one Functional). `perf_section` renders the builder's `sidecar` dict (all 23
    `FmhaKnobs` fields, via `asdict(knobs)`) as `k=v;k=v`; `copt_section` is always
    empty (see module docstring)."""

    def __init__(self, f: 'Functional', *, sidecar: dict | None = None):
        self._functional = f
        # The builder's (built, sidecar) return, kept verbatim (PLAN-PHASE2.md
        # Task 2's trap: NEVER read back from the on-disk <hsaco>.json, which is
        # produced later, at true build time, by a different process).
        self._sidecar = dict(sidecar) if sidecar else {}

    @property
    def perf_section(self) -> str:
        return ';'.join(f'{k}={_schemaless_value(v)}' for k, v in self._sidecar.items())

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
