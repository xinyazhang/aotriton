# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`PerfDescription`: the per-family description perfmon dispatch drives,
modelled on `aotriton.tune.tdesc.TuningDescription` (T02, perfmon-exec0.md).

Same import discipline as `TuningDescription`: torch-free AT MODULE SCOPE and
AT CONSTRUCTION TIME. `perfmon dispatch` (like `dispatch_tasks.py`)
instantiates every registered family's `PerfDesc()` up front to build
argparse subparsers from `list_ifaces()` / entry-space metadata, so a
top-level torch/pyaotriton import here would break dispatch on any machine
that lacks them (most build/dispatch hosts do). Concrete subclasses (e.g.
`modules/flash/perfmon/__init__.py`'s `PerfDesc`) must only import
torch/pyaotriton lazily, inside methods that actually need a GPU -- and
`prime_entries`/`coverage_entries`/`list_ifaces`/`functional_pon`/
`shape_pon`/`tflops` never do, by design: they are pure entry-space /
arithmetic operations, answerable without a GPU.

perfmon-rev0.md §7 defines the on-disk PON split this ABC exposes:

  * `functional_pon(entry, iface)` -- the FEATURE half: `iface`, `dtype`,
    `causal`, `dropout_p`, `bias_type`, `gqa`, `varlen`, `storage_flip`.
    Hashed with SHA-256 to a filename (store.py, T26).
  * `shape_pon(entry)` -- the SHAPE half: `hdim`, `seqlen_q`, `seqlen_k`,
    `BATCH`, `N_HEADS`. Assigned a small stable-within-tag integer id
    (store.py, T26).

Both are rendered with `aotriton.utils.pon.render_pon` (T01) so the wire
format is byte-identical to what `FlashEntry.as_text()` already emits
(rev0 §7) -- no new serialization format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class PerfDescription(ABC):
    """Per-family perfmon description: entry-space generators, PON
    rendering, and TFLOPS derivation. Analogous in spirit to
    `TuningDescription`, but scoped to perfmon's much narrower needs (no
    accuracy checking, no impl resolution -- backend selection is handled by
    the existing `ImplSelector`/op-level machinery, reused unchanged, D3)."""

    #: The family's QUEUED entry dataclass (reused from the family's `tune`
    #: block, never redefined here) -- what task_config['entry'] holds and
    #: what prime_entries()/coverage_entries() yield, e.g. flash's
    #: `FlashEntry`. Deliberately narrower than INPUT_METADATA: fields the
    #: GPU worker resolves (N_HEADS, BATCH, ...) are not dispatch-time
    #: choices, so they are not part of this class.
    ENTRY_CLASS: type

    #: What the GPU worker RESOLVES a queued `ENTRY_CLASS` instance to
    #: (D05's `resolve_entry()`), e.g. flash's `FlashInputMetadata`. Carries
    #: every `ENTRY_CLASS` field plus the ones only the worker can pick
    #: (N_HEADS, BATCH, storage_flip, ...), since those depend on the
    #: worker's own VRAM. `shape_pon()`/`functional_pon()` take an instance
    #: of THIS class, not of `ENTRY_CLASS` -- see their docstrings.
    INPUT_METADATA: type

    def validate_hw_feature(self, arch: str, entry) -> tuple[bool, str]:
        """(supported, reason). Called per (arch, entry); False skips the
        entry. `reason` is printed once per (arch, reason), never per entry.

        Same contract as `aotriton.tune.tdesc.TuningDescription.
        validate_hw_feature` (tdesc.py:180). Default: everything is
        supported. Subclasses override to reject hardware-unsupported
        configurations (e.g. a per-arch sequence-length ceiling)."""
        return True, ''

    @abstractmethod
    def prime_entries(self, arch: str, max_seqlen: int) -> Iterator:
        """Yield `ENTRY_CLASS` instances for the "prime" set (rev0 §6.1):
        the small, always-published entry space (major hdims × causal ×
        {sq==sk seqlens <= max_seqlen} × dtype), every other field at its
        feature-off default. `max_seqlen` excludes entries that do not fit
        this variant's VRAM -- excluded, never shrunk (BATCH/N_HEADS are
        fixed per entry, rev0 §6.1)."""

    @abstractmethod
    def coverage_entries(self, arch: str, max_seqlen: int) -> Iterator:
        """Yield `ENTRY_CLASS` instances for the "coverage" set (rev0 §6.2):
        the full functional cross-product, seqlen space reduced to the
        L-shaped `{(s,s)} ∪ {(s,max)} ∪ {(max,s)}` pairs."""

    @abstractmethod
    def entry_set_names(self) -> dict[str, str]:
        """Named entry set -> a one-line description of what it is FOR.

        Both halves are used: the keys are `--entry_set`'s legal values, and
        the descriptions go in its help, because the names alone do not tell
        an operator which to reach for."""

    @abstractmethod
    def entry_set_axes(self, name: str, arch: str, max_seqlen: int) -> dict:
        """Axis values a named entry set supplies, keyed by ENTRY_CLASS field
        name -- except that `(seqlen_q, seqlen_k)` appear as ONE `seqlen_qk`
        axis of pairs.

        The AXES are the ground truth for what gets dispatched; `--entry_set`
        is a shortcut that supplies their values, and any axis the operator
        names on the command line replaces what the set supplied. So a set is
        a named bundle of defaults, not a fixed list of entries.

        `seqlen_qk` is one axis and not two because no set crosses q with k:
        the prime set walks the diagonal and coverage walks the L-shape
        (`3n-2` pairs, not `n^2`). Two independent axes could not express
        either without also generating pairs the sets deliberately exclude.
        """

    @abstractmethod
    def entries_from_axes(self, axes: dict):
        """Cross product of `axes` -> ENTRY_CLASS instances, in a stable
        order (rev0 §5.7 makes perfmon's insert order load-bearing)."""

    @abstractmethod
    def list_ifaces(self) -> list[str]:
        """Bare interface names this family measures end-to-end (op level
        only, D3), e.g. `['attn_fwd', 'attn_bwd']` for flash. Unlike tuning's
        `list_impls()`, these are never DSL-prefixed -- perfmon has exactly
        one level."""

    @abstractmethod
    def functional_pon(self, entry, iface: str) -> str:
        """The FEATURE half of `entry`'s PON, rendered with `render_pon`
        (rev0 §7): `iface`, `dtype`, `causal`, `dropout_p`, `bias_type`,
        `gqa`, `varlen`, `storage_flip`. Hashed to a filename by `store.py`
        (T26); two entries differing only in shape must render identical
        `functional_pon` text.

        Takes the RESOLVED metadata (an `INPUT_METADATA` instance), not a
        queued `ENTRY_CLASS` entry. `N_HEADS`/`BATCH`/`storage_flip` are
        chosen on the GPU worker (D05), so this is a report-time call."""

    @abstractmethod
    def shape_pon(self, entry) -> str:
        """The SHAPE half of `entry`'s PON, rendered with `render_pon`
        (rev0 §7): `hdim`, `seqlen_q`, `seqlen_k`, `BATCH`, `N_HEADS`.
        Assigned a small integer id, stable within one `docs/perf/<tag>/`
        directory, by `store.py` (T26).

        Takes the RESOLVED metadata (an `INPUT_METADATA` instance), not a
        queued `ENTRY_CLASS` entry. `N_HEADS`/`BATCH` are chosen on the GPU
        worker (D05), so this is a report-time call."""

    @abstractmethod
    def tflops(self, entry, iface: str, seconds: float) -> float:
        """Achieved Matrix Core TFLOPS for one measurement of `entry` on
        `iface`, given the wall time of one iteration in seconds. Ports
        `modules/<family>/visperf/static/<family>.js`'s formula (rev0 §6,
        §8) -- see e.g. `modules/flash/perfmon/tflops.py`."""

    @abstractmethod
    def resolve_entry(self, entry, vram_total_gb: float | None):
        """`ENTRY_CLASS` (queued) -> `INPUT_METADATA` (resolved), against
        this GPU's own VRAM (D05's per-family memory model, e.g. flash's
        `modules/flash/perfmon/resolve.py`). `vram_total_gb` is the GPU
        worker's own runner self-report (D05a); `None` means unknown, and an
        implementation must not silently shrink the workload in that case.

        Added here (dispatch-perfmon-exec.md D12) so `localq/handlers/
        perf_measure.py` -- which, like `tune_kernel.py`, must stay
        family-neutral -- can call `desc.resolve_entry(...)` instead of
        importing a specific family's `resolve` module directly. D05 itself
        only committed the flash implementation; this abstract method is the
        hook that was missing to reach it generically."""
