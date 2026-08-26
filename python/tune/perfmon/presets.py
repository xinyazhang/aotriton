# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Preset enumeration for perfmon dispatch (dispatch-perfmon-exec.md D06).

A "preset" is one `(rocm, aotriton_tag)` pair this fleet is configured to
measure, rendered as `f'rocm{rocm}+aotriton{tag}'` -- the same string
`launch_runner.sh` and D10's `PerfEntrySource` key off of.

sqlite3 only, stdlib -- no new dependency.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _read_config(workdir: Path, key: str) -> str:
    """One value from `<workdir>/workers.db`'s `config` table. Raises,
    naming both the key and the file, if the key is absent -- never a bare
    `KeyError` (a caller catching `sqlite3`/`KeyError` specifically would
    miss this)."""
    db_path = workdir / 'workers.db'
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            'SELECT value FROM config WHERE key = ?', (key,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"config key {key!r} not found in {db_path}")
    return row[0]


def available_presets(workdir: Path) -> list[str]:
    """Presets this fleet is CONFIGURED TO MEASURE, from workers.db.

    Enumeration, not validation. Every perfmon worker is expected to serve
    every preset -- if that needs a nested container, that is the worker's
    problem (launch_runner.sh is the layer for it). So this must NOT check
    whether anything can run them: there is no availability to test.
    """
    rocm = _read_config(workdir, 'perfmon::default_rocm')
    tags = json.loads(_read_config(workdir, 'perfmon::tags'))
    return [f'rocm{rocm}+aotriton{tag}' for tag in tags]


def parse_preset(preset: str) -> tuple[str, str]:
    """`'rocm7.14.0+aotriton0.13b'` -> `('7.14.0', '0.13b')`. Used only for
    display and for `launch_runner.sh`'s directory (keyed on the tag half).
    Do not route on the halves elsewhere -- dispatch-perfmon.md §3.2:
    multi-ROCm is explicitly out of scope."""
    if not preset.startswith('rocm') or '+aotriton' not in preset:
        raise ValueError(f"malformed preset: {preset!r}")
    rocm_part, _, tag = preset.partition('+aotriton')
    rocm = rocm_part[len('rocm'):]
    return rocm, tag
