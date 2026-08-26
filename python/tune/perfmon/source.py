# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`dispatch.driver.EntrySource` for perfmon (dispatch-perfmon-exec.md D10).

The perfmon analogue of `dispatch_tasks.py`'s `TuneEntrySource` (D07), but
genuinely different where the two workloads are genuinely different
(dispatch-perfmon.md §5.2 / `dispatch/driver.py`'s module docstring):

* Entries come from `PerfDescription.prime_entries`/`coverage_entries` --
  curated sets, never a per-field cross product.
* Batches are ONE PRESET AT A TIME, and that order matters here where it
  does not for tuning: `dispatch-perfmon.md` §5.7 -- grouping the insert by
  preset saves the profiled runner a ~390ms context switch per preset
  change, so `PerfEntrySource.batches()` yielding presets in a stable order
  is load-bearing, not cosmetic.
* `task_config` carries a `"preset"` key tuning's never has (§7); `queue_row`
  always writes `class='perf_measure'`, `subclass=''` -- `schema.sql`'s CHECK
  constraint accepts nothing else for that class, and that rejection is the
  point (D10 spec).
"""

from __future__ import annotations

import sys
from dataclasses import asdict


class PerfEntrySource:
    """One (preset, arch, entry) -> task_config -> queue_row pipeline for
    perf_measure dispatch.

    `presets` is resolved by the caller (`dispatch_tasks.py`'s `dispatch_
    perf_tasks`) from `--preset` if given, else `perfmon.presets.
    available_presets(workdir)` (D06) -- `--preset` SELECTS a subset, it
    does NOT validate that a preset is real or servable
    (dispatch-perfmon.md §3.4); this class just iterates whatever list it is
    handed.
    """

    def __init__(self, module_name: str, module_instance, args, *, presets: list[str]):
        self.module_name = module_name
        self.module_instance = module_instance
        self.args = args
        self.presets = presets
        self.ENTRY_CLASS = module_instance.ENTRY_CLASS

        # `PerfDescription.max_seqlen` (D04's `validate_hw_feature`): an
        # optional CLI-supplied ceiling, combined with a per-arch measured
        # one if any. `None` (this attribute's own default) means "no CLI
        # override" -- passed through as-is, NOT coerced to a sentinel like
        # `entries()` below has to for entry GENERATION, since `validate_
        # hw_feature` already treats `None` as "no limit from this source".
        module_instance.max_seqlen = args.max_seqlen

    def batches(self):
        yield from self.presets

    def entries(self, batch, arch):
        """`prime_entries`/`coverage_entries` require a concrete int bound
        (entries above it are excluded at generation time, never shrunk --
        `pdesc.py`'s docstrings). When `--max_seqlen` was not given
        (`args.max_seqlen is None`), generate everything and let `validate_
        hw_feature`'s own ceiling (which DOES understand `None` as
        "unbounded") do the filtering instead -- `sys.maxsize` here is
        just "no generation-time bound", not a real seqlen anyone will ever
        request."""
        max_seqlen = self.args.max_seqlen if self.args.max_seqlen is not None else sys.maxsize
        if self.args.entry_set == 'coverage':
            yield from self.module_instance.coverage_entries(arch, max_seqlen)
        else:
            yield from self.module_instance.prime_entries(arch, max_seqlen)

    def validate_hw_feature(self, arch, entry):
        return self.module_instance.validate_hw_feature(arch, entry)

    def task_config(self, batch, arch, entry) -> dict:
        """dispatch-perfmon.md §7: exactly these four keys -- no
        `tuning_level` (perfmon has exactly one level)."""
        return {
            "arch": arch,
            "module": self.module_name,
            "preset": batch,
            "entry": asdict(entry),
        }

    def queue_row(self, task_config: dict) -> dict:
        return {
            'arch': task_config['arch'],
            'module': task_config['module'],
            'class': 'perf_measure',
            # schema.sql's CHECK constraint requires the empty string here
            # (class='perf_measure' AND subclass='') -- never anything else.
            'subclass': '',
            'task_config': task_config,
            'priority': 5,
        }
