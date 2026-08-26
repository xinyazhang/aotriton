# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Prime and coverage entry-space generators for flash perfmon
(perfmon-rev0.md §6, perfmon-exec0.md T03).

Reuses `FlashEntry`/`FlashInputMetadata` from `modules/flash/tune/entry.py`
(via `aotriton.tune.registry.load_flash_entry_module` -- the existing
by-path accessor for that module; `modules/flash` is a plain directory, not
a package, so a bare `from ..tune.entry import ...` cannot reach it from
this sibling `perfmon/` package block). Neither is redefined here.

D03 (dispatch-perfmon-exec.md) split perfmon's entry into two roles that
used to be conflated:

* `FlashEntry` -- the 7 base fields (`dtype`/`hdim`/`seqlen_q`/`seqlen_k`/
  `causal`/`dropout_p`/`bias_type`) -- is what gets QUEUED. `prime_entries`/
  `coverage_entries` below yield this class.
* `FlashInputMetadata` -- `FlashEntry` plus `N_HEADS`/`BATCH`/`storage_flip`
  -- is what the GPU worker RESOLVES a queued entry to (D05's
  `resolve_entry()`). Those three fields are worker-chosen (VRAM-dependent),
  never dispatch-time choices, so they cannot be part of what is queued.

Torch-free at module scope, matching every other perfmon description module
(pdesc.py's docstring) -- `load_flash_entry_module()` only reaches
`modules/flash/tune/entry.py`, which is itself torch-free at import time.

KNOWN GAPS relative to perfmon-rev0.md's coverage-set text (§6.2), reported
here rather than silently guessed:

* rev0 §6.2 lists a coverage functional cross-product of
  "dtype × dropout × bias_type × GQA × SWA/windowed × varlen". Neither a
  sliding-window ("SWA") flag nor a varlen flag exists anywhere on
  `FlashEntry`/`FlashInputMetadata` in this codebase today (grepped
  `modules/flash/tune/*.py`: `varlen_type` is a call-site constant `0` in
  `calls.py`, never a tunable entry field). This module therefore does not
  vary those two axes -- there is nothing to vary.
* GQA and `storage_flip` were previously varied here too (`N_HEADS` as
  `int` vs. `tuple[int, int]`, following the `dataclasses.replace(im,
  N_HEADS=(10, 2))` convention in `modules/flash/tune/desc.py`), but D03
  moved `N_HEADS`/`storage_flip` off the QUEUED entry entirely -- they live
  only on `FlashInputMetadata`, which is resolved worker-side (D05), never
  chosen at dispatch time. Coverage therefore no longer varies GQA or
  `storage_flip`: there is no longer a queued field for either to set.
* rev0 §6.2's prose does not list `hdim` or `causal` among the coverage
  functional axes (only the prime set, §6.1, crosses those two). Taken
  literally: coverage entries hold `hdim` and `causal` fixed. This module
  fixes them at `COVERAGE_HDIM` / `COVERAGE_CAUSAL` below, both called out
  explicitly so the choice is visible rather than buried in a loop.
* T04's Verify step for coverage only checks the seqlen L-shape pair count
  (`3n-2`), which is invariant to every other axis, so none of the above
  choices are load-bearing for the test that exists at time of writing (T13).
"""

from __future__ import annotations

from aotriton.tune.registry import load_flash_entry_module

_flash_entry_module = load_flash_entry_module()
FlashEntry = _flash_entry_module.FlashEntry
FlashInputMetadata = _flash_entry_module.FlashInputMetadata

# --- rev0 §6.1 prime set -----------------------------------------------------

PRIME_HDIMS = (64, 128, 192, 256, 384, 512)
PRIME_CAUSAL = (False, True)
PRIME_SEQLENS = (128, 1024, 4096, 16384)
PRIME_DTYPES = ('bfloat16', 'float16')


def seqlens_for(max_seqlen: int) -> list[int]:
    """Prime/coverage seqlens not exceeding this variant's `max_seqlen`
    (rev0 §6.1: entries above it are excluded upstream, never dispatched)."""
    return [s for s in PRIME_SEQLENS if s <= max_seqlen]


def prime_entries(max_seqlen: int):
    """rev0 §6.1: hdim × causal × {sq==sk seqlens <= max_seqlen} × dtype,
    every other field at its feature-off default. Independent of `iface` --
    the caller crosses this generator's output with `list_ifaces()`.

    Yields `FlashEntry` (the QUEUED shape, D03): `N_HEADS`/`BATCH`/
    `storage_flip` are not dispatch-time choices -- the GPU worker resolves
    them (D05's `resolve_entry()`), against its own VRAM."""
    seqlens = seqlens_for(max_seqlen)
    for hdim in PRIME_HDIMS:
        for causal in PRIME_CAUSAL:
            for seqlen in seqlens:
                for dtype in PRIME_DTYPES:
                    yield FlashEntry(
                        dtype=dtype,
                        hdim=hdim,
                        seqlen_q=seqlen,
                        seqlen_k=seqlen,
                        causal=causal,
                        dropout_p=0.0,
                        bias_type=0,
                    )


# --- rev0 §6.2 coverage set ---------------------------------------------------

COVERAGE_DTYPES = ('bfloat16', 'float16')
COVERAGE_DROPOUT_P = (0.0, 0.5)
COVERAGE_BIAS_TYPE = (0, 1)

# See the module docstring's "KNOWN GAPS" note: rev0 §6.2 does not cross
# these two axes for coverage, unlike the prime set. Fixed at a
# representative value rather than silently omitted from the entry.
COVERAGE_HDIM = 128
COVERAGE_CAUSAL = False


def l_shape_seqlen_pairs(seqlens: list[int]) -> list[tuple[int, int]]:
    """The wishlist's L-shape (rev0 §6.2): the diagonal plus every (s, max)
    and (max, s) pair. `3n-2` pairs for `n` distinct seqlens (n>=1)."""
    if not seqlens:
        return []
    s_max = max(seqlens)
    pairs = {(s, s) for s in seqlens}
    pairs |= {(s, s_max) for s in seqlens}
    pairs |= {(s_max, s) for s in seqlens}
    return sorted(pairs)


def coverage_entries(max_seqlen: int):
    """rev0 §6.2: full functional cross-product over the axes that actually
    exist on `FlashEntry` today (dtype × dropout_p × bias_type -- see the
    module docstring's "KNOWN GAPS" for GQA/storage_flip/SWA/varlen, none of
    which are queued-entry fields), crossed with the L-shaped seqlen pair
    set.

    Yields `FlashEntry` (the QUEUED shape, D03) -- see `prime_entries()`."""
    seqlens = seqlens_for(max_seqlen)
    pairs = l_shape_seqlen_pairs(seqlens)
    for seqlen_q, seqlen_k in pairs:
        for dtype in COVERAGE_DTYPES:
            for dropout_p in COVERAGE_DROPOUT_P:
                for bias_type in COVERAGE_BIAS_TYPE:
                    yield FlashEntry(
                        dtype=dtype,
                        hdim=COVERAGE_HDIM,
                        seqlen_q=seqlen_q,
                        seqlen_k=seqlen_k,
                        causal=COVERAGE_CAUSAL,
                        dropout_p=dropout_p,
                        bias_type=bias_type,
                    )
