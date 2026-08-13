# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A safe, separator-configurable replacement for `v3python/tune/utils.py`'s
`parse_python`.

`parse_python` (`v3python/tune/utils.py:43-50`) splits a line on `;`, then on
`=` with `maxsplit=1`, then calls `eval(v)` on the value. Two problems: the
`eval` can execute arbitrary code, and the separator is hard-coded to `;` so
nothing that needs a different one (e.g. flyc's `--signature`/`--hints`
strings, which use a space) can reuse it.

`parse_kv` keeps the shape and fixes both: `ast.literal_eval` accepts exactly
the forms build inputs use -- ints, floats, quoted strings, tuples, lists,
`True`/`False`/`None` -- and raises on anything else, including bare
identifiers. `sep` defaults to `';'` so this is a drop-in for every existing
`parse_python` caller (`FlashEntry.parse_text`, `FlashInputMetadata.parse_text`);
callers that use a different wire separator (`flyc_compile` passes `sep=' '`)
get one parser instead of two spellings of one.
"""

import ast


def parse_kv(line: str, sep: str = ';') -> dict:
    """Parse `"k1=v1<sep>k2=v2..."` into `{k1: v1, k2: v2, ...}`.

    Each value is decoded with `ast.literal_eval`, so it must be a Python
    literal (int, float, quoted string, tuple, list, `True`/`False`/`None`);
    anything else -- notably a bare identifier or a call -- raises rather
    than executing.
    """
    d = {}
    for assignment in filter(None, (a.strip() for a in line.split(sep))):
        k, v = assignment.split('=', maxsplit=1)
        d[k.strip()] = ast.literal_eval(v.strip())
    return d
