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

`render_pon` is the matching writer: `repr()`-compatible per value, not
`str()`, so every value it emits is exactly what `ast.literal_eval` (hence
`parse_pon`) would read back -- the format is round-trippable in both
directions, and there is only one dialect. For `str` values specifically,
`render_pon` requires `repr(v) == "'" + v + "'"`: single-quoted, nothing
escaped. If a value ever stops satisfying that (a quote, backslash, or
non-printable character in it), it raises loudly, naming the key, at the point
the wire text is built -- not silently emitting something the C++ reader
(`class Pon`, `include/aotriton/_internal/pon.h`) cannot parse with its plain
two-character quote-strip.

**A PON string contains NO SPACES.** This is a hard property of the format, not
a cosmetic one, and it is why `render_pon` cannot simply call `repr()` on a
container: `repr((1, 2))` is `'(1, 2)'`, with a space after the comma.

The reason is the exaid/testrun wire protocol. `ExaidProxy.write`
(`python/tune/exaid.py`) joins its arguments with `' '`, and the worker's
`first()` (`python/tune/testrun.py`) splits the line back apart on `' '`. A
single embedded space therefore does not corrupt the value -- it silently
re-tokenizes the whole command, so a later positional argument is read from the
middle of an earlier one. `render_pon` renders containers with a bare `,`
separator and rejects any value whose rendering would contain a space,
including a `str` with a space inside it.
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


def _render_value(k, v) -> str:
    """One PON value: `repr()`-equivalent, but with no spaces anywhere.

    Recurses through tuples/lists rather than calling `repr()` on them, because
    `repr` separates container elements with `', '` and a space breaks the wire
    protocol (see the module docstring). The single-element tuple keeps its
    trailing comma -- `(1,)`, not `(1)`, which would read back as a plain int.
    """
    if isinstance(v, str):
        if repr(v) != "'" + v + "'":
            raise ValueError(
                f"render_pon: value of {k!r} ({v!r}) does not round-trip "
                f"through a plain single-quoted repr() (it contains a quote, "
                f"backslash, or non-printable character) -- PON cannot "
                f"represent it")
        if ' ' in v:
            raise ValueError(
                f"render_pon: value of {k!r} ({v!r}) contains a space. A PON "
                f"string is passed as one token on the exaid/testrun wire, "
                f"which splits on ' ', so an embedded space silently "
                f"re-tokenizes the whole command")
        return repr(v)
    if isinstance(v, tuple):
        inner = ','.join(_render_value(k, x) for x in v)
        return f'({inner},)' if len(v) == 1 else f'({inner})'
    if isinstance(v, list):
        return '[' + ','.join(_render_value(k, x) for x in v) + ']'
    out = repr(v)
    if ' ' in out:
        raise ValueError(
            f"render_pon: value of {k!r} ({v!r}) renders as {out!r}, which "
            f"contains a space; PON values must be space-free (see the module "
            f"docstring on the exaid/testrun wire protocol)")
    return out


def render_pon(d: dict, sep: str = ';') -> str:
    """Render `{k1: v1, k2: v2, ...}` as `"k1=v1<sep>k2=v2..."`.

    `parse_pon(render_pon(d)) == d` for any `d` this function accepts, and the
    result never contains a space. A `str` value must round-trip through a
    plain single-quoted `repr()` -- `repr(v) == "'" + v + "'"` -- or this
    raises `ValueError` naming the offending key; that precondition is what
    lets the C++ reader strip exactly one pair of surrounding single quotes
    instead of implementing a full Python-repr unescaper.
    """
    text = sep.join(f'{k}={_render_value(k, v)}' for k, v in d.items())
    # Belt and braces: _render_value rejects spaces per value, but a caller
    # passing sep=' ' would reintroduce them at the joins, and that is legal
    # (flyc's --signature is space-separated and is NOT a single wire token).
    # Only assert the property the per-value checks are responsible for.
    assert sep == ' ' or ' ' not in text, (
        f'render_pon: emitted a space with sep={sep!r}: {text!r}')
    return text
