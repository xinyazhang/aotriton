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
from dataclasses import asdict, fields


def norm_value(value) -> str:
    """Text form of an entry field, for `--<field>` matching.

    Normalising both sides to text is what lets one comparison serve fields
    whose types are unions (`hdim: int | tuple[int, int]`, `causal: bool |
    tuple[int, int]`) without the CLI having to know which arm it got:

      * `bool` -> int, so `--causal 1` matches `causal=True`;
      * integral `float` -> int, so `--dropout_p 0` matches `0.0`;
      * `tuple` -> comma-joined, so `--hdim 64,128` matches `(64, 128)`.
    """
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, tuple):
        return ','.join(norm_value(v) for v in value)
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def norm_text(text: str) -> str:
    """`norm_value` for a string off the command line, so `0.0` and `0`
    agree with each other and with the entry."""
    if ',' in text:
        return ','.join(norm_text(part) for part in text.split(','))
    try:
        f = float(text)
    except ValueError:
        return text
    return str(int(f)) if f == int(f) else str(f)


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

    #: Every axis of the entry space. `seqlen_qk` is one axis of (q, k)
    #: pairs, not two -- see PerfDescription.entry_set_axes.
    AXIS_NAMES = ('dtype', 'hdim', 'seqlen_qk', 'causal', 'dropout_p', 'bias_type')

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

        # --- axes: the ground truth for what gets dispatched ---------
        #
        # `--entry_set` SUPPLIES VALUES to the axes; any axis named on the
        # command line replaces what it supplied. A set is a named bundle of
        # defaults, not a fixed list of entries, so a combination in no set
        # is still reachable for a one-off debug run.
        #
        # `--entry_set` has NO DEFAULT, deliberately. Defaulting it to
        # `prime` would make a bare invocation quietly queue the whole prime
        # set across every preset -- thousands of rows from a command that
        # named nothing. Omitting it is legal, but then every axis has to be
        # given by hand, and the error below says which are missing.
        max_seqlen = args.max_seqlen if args.max_seqlen is not None else sys.maxsize
        if args.entry_set is None:
            self.axes = {}
        else:
            self.axes = module_instance.entry_set_axes(
                args.entry_set, arch=None, max_seqlen=max_seqlen)

        self.overridden = []
        for name in self.AXIS_NAMES:
            given = getattr(args, name, None)
            if given is None:
                continue
            self.axes[name] = list(given)   # already typed by argparse
            self.overridden.append(name)

        missing = [n for n in self.AXIS_NAMES if not self.axes.get(n)]
        if missing:
            raise ValueError(
                "no values for " + ', '.join(f'--{m}' for m in missing)
                + ". Give them explicitly, or pass --entry_set to supply "
                  "every axis at once.")

        # A CLI-supplied seqlen ceiling has to be reapplied to an axis the
        # operator set by hand, since only the entry set consulted it.
        if args.max_seqlen is not None and 'seqlen_qk' in self.overridden:
            self.axes['seqlen_qk'] = [
                (q, k) for q, k in self.axes['seqlen_qk']
                if max(q, k) <= args.max_seqlen]
            if not self.axes['seqlen_qk']:
                raise ValueError(
                    f"--max_seqlen {args.max_seqlen} excludes every pair "
                    "given to --seqlen_qk.")

    def batches(self):
        yield from self.presets

    def entries(self, batch, arch):
        """The cross product of the resolved axes (see __init__)."""
        yield from self.module_instance.entries_from_axes(self.axes)

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
