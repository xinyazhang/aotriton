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

# Tuning's own seqlen ladder (modules/flash/tune/desc.py's
# get_entry_choices()), plus 16384.
#
# Perfmon exists to monitor the performance of TUNING TABLE ENTRIES, so its
# candidates have to be the shapes the tuning table actually holds -- a
# seqlen nobody tuned has no table entry whose performance could be tracked.
# An earlier, invented ladder (128, 1024, 4096, 16384) sampled four points
# that mostly are tuned but skipped everything below 128 and every step
# between, so most of the table went unmonitored.
#
# 16384 is the deliberate exception: it is past the top of tuning's range,
# kept because rev0 §6.1 wants a point beyond what is tuned, and it is the
# one candidate `max_seqlen` most often excludes on a small-VRAM part.
TUNING_SEQLENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PRIME_SEQLENS = TUNING_SEQLENS + (16384,)
PRIME_DTYPES = ('bfloat16', 'float16')


def seqlens_for(max_seqlen: int) -> list[int]:
    """Prime/coverage seqlens not exceeding this variant's `max_seqlen`
    (rev0 §6.1: entries above it are excluded upstream, never dispatched)."""
    return [s for s in PRIME_SEQLENS if s <= max_seqlen]


def prime_entries(max_seqlen: int):
    """rev0 §6.1, as the cross product of `prime_axes()`. Kept as a named
    generator because tests and any caller that wants the set without going
    through the CLI still ask for it by name."""
    yield from entries_from_axes(prime_axes(max_seqlen))


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
    """rev0 §6.2, as the cross product of `coverage_axes()`."""
    yield from entries_from_axes(coverage_axes(max_seqlen))


# --- entry-set axes: the ground truth is the AXES, not the generators -------
#
# `--dtype`, `--hdim`, `--seqlen_qk`, ... are what define the entry space.
# `--entry_set` is a shortcut that SUPPLIES VALUES to them, and any axis the
# operator names explicitly replaces what the set supplied. That is the
# inverse of the first implementation, where the curated generators were
# authoritative and the flags could only subset their output -- which made a
# combination outside the set unreachable even for a one-off debug run.
#
# `seqlen_qk` is ONE axis of (seqlen_q, seqlen_k) pairs, not two independent
# axes, because neither set crosses them: prime walks the diagonal and
# coverage walks the L-shape (`3n-2` pairs, not `n^2`). Two independent axes
# could not express either without also generating pairs the sets
# deliberately exclude.

def prime_axes(max_seqlen: int) -> dict:
    """rev0 §6.1's axis values. Every field of FlashEntry appears, so the
    cross product of this dict IS the prime set."""
    return {
        'dtype': list(PRIME_DTYPES),
        'hdim': list(PRIME_HDIMS),
        'seqlen_qk': [(s, s) for s in seqlens_for(max_seqlen)],
        'causal': list(PRIME_CAUSAL),
        'dropout_p': [0.0],
        'bias_type': [0],
    }


def coverage_axes(max_seqlen: int) -> dict:
    """rev0 §6.2's axis values -- the functional cross product over the axes
    FlashEntry actually has, crossed with the L-shaped seqlen pairs."""
    return {
        'dtype': list(COVERAGE_DTYPES),
        'hdim': [COVERAGE_HDIM],
        'seqlen_qk': l_shape_seqlen_pairs(seqlens_for(max_seqlen)),
        'causal': [COVERAGE_CAUSAL],
        'dropout_p': list(COVERAGE_DROPOUT_P),
        'bias_type': list(COVERAGE_BIAS_TYPE),
    }


ENTRY_SETS = {'prime': prime_axes, 'coverage': coverage_axes}

#: One line per set, for `--entry_set`'s help. Says what the set is FOR, not
#: just what it contains -- the names alone do not tell an operator which to
#: reach for.
ENTRY_SET_HELP = {
    'prime': "the shapes the tuning table holds -- every tuned seqlen plus "
             "16384, crossed with hdim x causal x dtype, no dropout or bias",
    'coverage': "functional breadth -- dropout and bias on and off, over the "
                "L-shaped seqlen pairs (3n-2, not n^2), at one hdim",
}


def entries_from_axes(axes: dict):
    """Cross product of `axes` -> FlashEntry, one per combination.

    Field order is fixed here rather than taken from the dict so the
    emitted order is stable across runs and across Python versions --
    dispatch inserts in this order, and rev0 §5.7 makes insert order
    load-bearing for perfmon.
    """
    import itertools
    for dtype, hdim, (sq, sk), causal, dropout_p, bias_type in itertools.product(
            axes['dtype'], axes['hdim'], axes['seqlen_qk'],
            axes['causal'], axes['dropout_p'], axes['bias_type']):
        yield FlashEntry(
            dtype=dtype,
            hdim=hdim,
            seqlen_q=sq,
            seqlen_k=sk,
            causal=causal,
            dropout_p=dropout_p,
            bias_type=bias_type,
        )
