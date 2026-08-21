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

    #: The family's entry dataclass (reused from the family's `tune` block,
    #: never redefined here -- e.g. flash's `FlashInputMetadata`).
    ENTRY_CLASS: type

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
        `functional_pon` text."""

    @abstractmethod
    def shape_pon(self, entry) -> str:
        """The SHAPE half of `entry`'s PON, rendered with `render_pon`
        (rev0 §7): `hdim`, `seqlen_q`, `seqlen_k`, `BATCH`, `N_HEADS`.
        Assigned a small integer id, stable within one `docs/perf/<tag>/`
        directory, by `store.py` (T26)."""

    @abstractmethod
    def tflops(self, entry, iface: str, seconds: float) -> float:
        """Achieved Matrix Core TFLOPS for one measurement of `entry` on
        `iface`, given the wall time of one iteration in seconds. Ports
        `modules/<family>/visperf/static/<family>.js`'s formula (rev0 §6,
        §8) -- see e.g. `modules/flash/perfmon/tflops.py`."""
