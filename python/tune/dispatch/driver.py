# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared dispatch driver: generate, dedup, confirm and insert
(dispatch-perfmon-exec.md D07; survey in dispatch-perfmon.md §5).

Extracted verbatim (in behavior, not necessarily in loop order -- see
`dispatch()`'s docstring) from `dispatch_tasks.py`'s old, tuning-only
`dispatch_tasks()` body (lines 207-331 before this commit): the dedup, the
arch loop, the `validate_hw_feature` skip with its once-per-`(arch, reason)`
printing, the progress lines, the confirmation prompt, `--dry_run`,
`ensure_partition`, and `dispatch_bulk`. About 20 of dispatch_tasks()'s ~125
lines were workload-specific (§5.1); those stay behind, split out per
workload into an `EntrySource` implementation.

Entry generation itself (§5.2) is deliberately NOT part of this shared
driver: tuning's cross product of per-field choices and perfmon's curated
prime/coverage sets are different in kind, and forcing either into the
other's shape would be a lie. That is the one real seam, `EntrySource`.

For tuning specifically, this is a pure refactor: the same entries, same
task_configs, and the same SET of queue rows come out as before. Insertion
ORDER may differ (this driver loops batch-outermost, then arch, then
entries within a batch -- the old tuning loop was entry-outermost, arch
innermost), but per dispatch-perfmon.md §5.7, insert order only matters for
perfmon (`fetch_tasks` orders `priority DESC, id ASC`, so grouping by preset
avoids the profiled runner's 390ms-per-preset-change cost); tuning has no
such ordering dependency.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import fields
from typing import Protocol, runtime_checkable


@runtime_checkable
class EntrySource(Protocol):
    """One (batch, arch, entry) -> task_config -> queue_row pipeline.

    `TuneEntrySource` (dispatch_tasks.py) and `PerfEntrySource` (D10,
    perfmon/source.py) are the two implementations."""

    #: The queued entry dataclass. Used only to build the dedup key
    #: (`(arch,) + entry field values`, dispatch-perfmon.md §5.4) -- never
    #: to generate entries; that is `entries()`'s job, per-implementation.
    ENTRY_CLASS: type

    def batches(self) -> Iterator[str | None]:
        """Insert-order groups. Perfmon yields one preset at a time (order
        is worth 3.5x, dispatch-perfmon.md §5.7); tuning yields a single
        `None`."""
        ...

    def entries(self, batch: str | None, arch: str) -> Iterator:
        """Entries for one (batch, arch). §5.2: cross product for tuning,
        curated set for perfmon -- genuinely different in kind, so this is
        the whole seam."""
        ...

    def validate_hw_feature(self, arch: str, entry) -> tuple[bool, str]:
        """(supported, reason); same contract as `TuningDescription`/
        `PerfDescription`.validate_hw_feature."""
        ...

    def task_config(self, batch: str | None, arch: str, entry) -> dict:
        """The dict that becomes `task_queue.task_config` (plus `arch`/
        `module` used to build the queue row)."""
        ...

    def queue_row(self, task_config: dict) -> dict:
        """The `dispatch_bulk` input dict: arch, module, class, subclass,
        task_config, priority."""
        ...


def make_hashable_key(entry_field_names: tuple[str, ...], task_config: dict):
    """`(arch,) + entry field values + (preset,)` -- the dedup key shared by
    both workloads (dispatch-perfmon.md §5.4, D08). Derived from whichever
    `ENTRY_CLASS` the source declares, so this one function works for both
    `FlashEntry` (tuning) and `FlashEntry` (perfmon, D03) alike -- they
    happen to share a name today, but nothing here assumes that.

    `task_config.get('preset')` is always appended, even though tuning's
    task_config never has a `'preset'` key: `.get()` then reliably returns
    `None` on both sides of the comparison (this function computes the key
    for both freshly-generated task_configs here and for completed ones
    fetched by `TaskQueue.completed_task_configs`), so it is a no-op for
    tuning and, for perfmon, folds the preset into the entry's identity for
    free -- 'free' because `preset` lives inside `task_config`, not as a
    separate parameter this function would otherwise need.

    Public (no leading underscore): `dispatch_tasks.py`'s
    `get_completed_tasks` must hash completed rows with the exact same
    scheme this module uses for freshly-generated ones, or `--skip_completed`
    would silently stop deduplicating; importing this one definition is how
    that is guaranteed rather than merely hoped for."""
    arch = task_config['arch']
    entry_dict = task_config['entry']
    field_values = tuple(entry_dict[fname] for fname in entry_field_names)
    return (arch,) + field_values + (task_config.get('preset'),)


def dispatch(*, source: EntrySource, arch_list: list[str], conn_params: dict,
             args, completed_configs: set | None = None) -> int:
    """Generate, dedup, confirm and insert. Workload-neutral.

    `args` must provide `.skip_completed`, `.verbose`, `.dry_run`, `.yes`
    (read individually) and support `vars(args)` (the confirmation prompt
    prints every key/value, unchanged from the pre-D07 prompt).

    `completed_configs`, when `args.skip_completed` is set, is the caller's
    already-fetched, already-class-scoped completed-task set -- fetching it
    is still workload-specific (hits `task_queue` filtered by `(module,
    class, subclass)`), and D08 is what makes that fetch itself shared and
    class-aware; D07 only extracts what comes after it. `None` (the
    default) is only valid when `args.skip_completed` is falsy.

    Returns the number of rows actually dispatched (0 if cancelled, dry-run,
    or nothing to do).
    """
    from ..pq.dispatcher import TaskDispatcher

    completed = completed_configs if completed_configs is not None else set()
    entry_field_names = tuple(f.name for f in fields(source.ENTRY_CLASS))

    tty_output = sys.stdin.isatty()
    printed_hw_reasons = set()
    tasks_to_dispatch = []
    skipped_count = 0

    for batch in source.batches():
        for arch in arch_list:
            for entry in source.entries(batch, arch):
                supported, reason = source.validate_hw_feature(arch, entry)
                if not supported:
                    if tty_output:
                        key = (arch, reason)
                        if key not in printed_hw_reasons:
                            printed_hw_reasons.add(key)
                            print(f"Skipping {arch} configurations: {reason}")
                    continue

                task_config = source.task_config(batch, arch, entry)

                if args.skip_completed:
                    config_key = make_hashable_key(entry_field_names, task_config)
                    if config_key in completed:
                        skipped_count += 1
                        if args.verbose:
                            print(f"Skipping completed task: {task_config['entry']}")
                        continue

                tasks_to_dispatch.append(source.queue_row(task_config))

                if args.verbose:
                    print(f"Prepared task for {task_config['arch']}: {task_config['entry']}")

    print(f"Prepared {len(tasks_to_dispatch)} tasks for dispatch")
    if args.skip_completed and skipped_count > 0:
        print(f"Skipped {skipped_count} already-completed tasks")

    if not args.yes and sys.stdin.isatty():
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        try:
            response = input(f"Proceed with dispatch? [y/N]: ")
            if response.lower() not in ('y', 'yes'):
                print("Dispatch cancelled")
                return 0
        except (KeyboardInterrupt, EOFError):
            print("\nDispatch cancelled")
            return 0

    if args.dry_run:
        print("Dry run mode - tasks not dispatched")
        return 0

    dispatcher = TaskDispatcher(conn_params)

    for arch in arch_list:
        try:
            dispatcher.ensure_partition(arch)
        except Exception as e:
            print(f"Warning: Failed to ensure partition for {arch}: {e}", file=sys.stderr)

    dispatched = dispatcher.dispatch_bulk(tasks_to_dispatch, batch_size=1000)
    print(f"Dispatched {dispatched} tasks to PostgreSQL queue")
    return dispatched
