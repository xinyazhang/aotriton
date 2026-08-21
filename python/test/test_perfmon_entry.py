# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""perfmon-exec0.md T04: Stage A regression coverage for the perfmon entry
space (modules/flash/perfmon/entry.py, python/tune/perfmon/pdesc.py).

None of this needs a GPU, torch, dacite or a database connection -- see
test_tune_infra.py's module docstring for the same rationale applied to the
tuning infrastructure this mirrors.
"""

from dataclasses import fields
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / 'modules'


def _perfdesc():
    from aotriton.tune.registry import load_family_perfmon
    return load_family_perfmon('flash', modules_dir=_MODULES_DIR).PerfDesc()


# --- prime set size (rev0 §6.1) ----------------------------------------------

def test_prime_set_size_at_max_seqlen_16384():
    pd = _perfdesc()
    entries = list(pd.prime_entries('gfx950', 16384))
    # 6 hdims x 2 causal x 4 seqlens x 2 dtypes
    assert len(entries) == 6 * 2 * 4 * 2 == 96


def test_prime_set_size_at_max_seqlen_4096():
    pd = _perfdesc()
    entries = list(pd.prime_entries('gfx950', 4096))
    # 6 hdims x 2 causal x 3 seqlens (128/1024/4096; 16384 excluded) x 2 dtypes
    assert len(entries) == 6 * 2 * 3 * 2 == 72


def test_prime_set_excludes_seqlens_above_max_seqlen():
    pd = _perfdesc()
    entries = list(pd.prime_entries('gfx950', 4096))
    assert all(e.seqlen_q <= 4096 and e.seqlen_k <= 4096 for e in entries)


# --- coverage seqlen L-shape (rev0 §6.2) -------------------------------------

def _flash_perfmon_entry_module():
    """`modules/flash` is a plain directory, not an importable package (F6/D8
    -- see modules/flash/tune/__init__.py's docstring), so entry.py must be
    reached through the supported by-path loader, not `import modules...`."""
    from aotriton.tune.registry import load_family_perfmon
    from importlib import import_module
    mod = load_family_perfmon('flash', modules_dir=_MODULES_DIR)
    return import_module('.entry', package=mod.__name__)


@pytest.mark.parametrize('max_seqlen,n', [(128, 1), (1024, 2), (4096, 3), (16384, 4)])
def test_coverage_seqlen_pairs_count_3n_minus_2(max_seqlen, n):
    """T04: "coverage seqlen pairs number 3n-2 for n distinct seqlens"."""
    pd = _perfdesc()
    entries = list(pd.coverage_entries('gfx950', max_seqlen))
    pairs = {(e.seqlen_q, e.seqlen_k) for e in entries}
    assert len(pairs) == 3 * n - 2


@pytest.mark.parametrize('max_seqlen,n', [(128, 1), (1024, 2), (4096, 3), (16384, 4)])
def test_l_shape_seqlen_pairs_helper_agrees(max_seqlen, n):
    """Same property, exercised directly against the generator's own
    l_shape_seqlen_pairs()/seqlens_for() helpers."""
    entry_mod = _flash_perfmon_entry_module()
    seqlens = entry_mod.seqlens_for(max_seqlen)
    pairs = entry_mod.l_shape_seqlen_pairs(seqlens)
    assert len(seqlens) == n
    assert len(pairs) == 3 * n - 2


# --- PON round-trip (rev0 §7) -------------------------------------------------

def test_pon_round_trip_covers_every_flash_entry_field_plus_iface_batch_n_heads():
    from aotriton.tune.registry import load_flash_entry_module
    from aotriton.utils.pon import parse_pon

    FlashEntry = load_flash_entry_module(modules_dir=_MODULES_DIR).FlashEntry
    flash_entry_fields = {f.name for f in fields(FlashEntry)}

    pd = _perfdesc()
    entries = list(pd.prime_entries('gfx950', 16384))
    for entry in entries[:5]:
        for iface in pd.list_ifaces():
            wire = pd.functional_pon(entry, iface) + ';' + pd.shape_pon(entry)
            d = parse_pon(wire)
            assert flash_entry_fields <= d.keys()
            assert {'iface', 'BATCH', 'N_HEADS'} <= d.keys()
            assert d['iface'] == iface
            assert d['BATCH'] == entry.BATCH
            assert d['N_HEADS'] == entry.N_HEADS


def test_no_two_distinct_entries_share_a_pon_pair():
    pd = _perfdesc()
    entries = list(pd.prime_entries('gfx950', 16384))
    seen = set()
    for entry in entries:
        for iface in pd.list_ifaces():
            key = (pd.functional_pon(entry, iface), pd.shape_pon(entry))
            assert key not in seen, f'duplicate PON pair for {entry!r}, iface={iface!r}'
            seen.add(key)
    assert len(seen) == len(entries) * len(pd.list_ifaces())
