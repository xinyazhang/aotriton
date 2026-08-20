# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PON (Plain / Python Object Notation): a safe, separator-configurable
`k=v;k=v` wire format, and the one home for both reading and writing it.

`aotriton.tune.utils.parse_python` -- the format's original reader, since
retired -- split a line on `;`, then on `=` with `maxsplit=1`, then called
`eval(v)` on the value. Two problems: the `eval` could execute arbitrary
code, and the separator was hard-coded to `;` so nothing that needed a
different one (e.g. flyc's `--signature`/`--hints` strings, which use a
space) could reuse it.

`parse_pon` keeps the shape and fixes both: `ast.literal_eval` accepts exactly
the forms build inputs use -- ints, floats, quoted strings, tuples, lists,
`True`/`False`/`None` -- and raises on anything else, including bare
identifiers. `sep` defaults to `';'` so this was a drop-in for every
`parse_python` caller (`FlashEntry.parse_text`, `FlashInputMetadata.parse_text`,
now switched over); callers that use a different wire separator
(`flyc_compile` passes `sep=' '`) get one parser instead of two spellings of
one.

`render_pon` is the matching writer: `repr()` per value, not `str()`, so every
value it emits is exactly what `ast.literal_eval` (hence `parse_pon`) would
read back -- the format is round-trippable in both directions, and there is
only one dialect. For `str` values specifically, `render_pon` asserts that
`repr(v) == "'" + v + "'"`: single-quoted, nothing escaped. If a value ever
stops satisfying that (a quote, backslash, or non-printable character in it),
the assertion fails loudly, naming the key, at the point the wire text is
built -- not silently emitting something the C++ reader (`class Pon`,
`include/aotriton/_internal/pon.h`) cannot parse with its plain two-character
quote-strip.
"""

import ast


def parse_pon(line: str, sep: str = ';') -> dict:
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


def render_pon(d: dict, sep: str = ';') -> str:
    """Render `{k1: v1, k2: v2, ...}` as `"k1=v1<sep>k2=v2..."`.

    Every value is rendered with `repr()`, so `parse_pon(render_pon(d)) == d`
    for any `d` this function accepts. A `str` value must round-trip through a
    plain single-quoted `repr()` -- `repr(v) == "'" + v + "'"` -- or this
    raises naming the offending key; that precondition is what lets the C++
    reader strip exactly one pair of surrounding single quotes instead of
    implementing a full Python-repr unescaper.
    """
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            assert repr(v) == "'" + v + "'", (
                f"render_pon: value of {k!r} ({v!r}) does not round-trip "
                f"through a plain single-quoted repr() (it contains a quote, "
                f"backslash, or non-printable character) -- PON cannot "
                f"represent it")
        parts.append(f"{k}={v!r}")
    return sep.join(parts)
