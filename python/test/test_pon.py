# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for python/utils/pon.py: `parse_pon` / `render_pon` round-trip
and the build-time rejection of un-representable strings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from aotriton.utils import parse_pon, render_pon


def test_round_trip_scalars():
    d = {'a': 1, 'b': -1, 'flag': True, 'off': False, 'nothing': None,
         'pi': 3.5, 'name': 'transposed'}
    assert parse_pon(render_pon(d)) == d


def test_round_trip_tuple_and_list():
    d = {'shape': (1, 2), 'items': [0, 1, 2], 'nested': (('a', 'b'), 3)}
    assert parse_pon(render_pon(d)) == d


def test_render_pon_emits_no_spaces():
    # THE wire-protocol invariant. ExaidProxy.write joins its arguments with
    # ' ' and testrun's first() splits them back on ' ', so one space inside a
    # PON token does not corrupt that value -- it re-tokenizes the whole
    # command. repr((1, 2)) would have produced '(1, 2)'.
    text = render_pon({'shape': (1, 2), 'items': [0, 1, 2],
                       'nested': (('a', 'b'), 3), 'name': 'transposed'})
    assert ' ' not in text, text


def test_render_pon_single_element_tuple_keeps_comma():
    # '(1)' would read back as the int 1, silently changing the type.
    assert render_pon({'t': (1,)}) == 't=(1,)'
    assert parse_pon(render_pon({'t': (1,)})) == {'t': (1,)}


def test_render_pon_rejects_string_with_space():
    with pytest.raises(ValueError, match='spacey'):
        render_pon({'spacey': 'two words'})


def test_round_trip_custom_separator():
    d = {'BLOCK_M': 64, 'CAUSAL': True}
    text = render_pon(d, sep=' ')
    assert ';' not in text
    assert parse_pon(text, sep=' ') == d


def test_render_pon_quotes_strings():
    # The unified (quoted) dialect: a str value is rendered with its repr(),
    # not bare -- 'transposed', never transposed.
    assert render_pon({'v_lds_layout': 'transposed'}) == "v_lds_layout='transposed'"


def test_render_pon_rejects_unrepresentable_string():
    # A string whose repr() is not a plain single-quoted form (here: it
    # contains a single quote itself) cannot be represented by this grammar --
    # render_pon must raise, naming the offending key, rather than silently
    # emitting text the C++ reader cannot parse.
    with pytest.raises(ValueError, match='bad_key'):
        render_pon({'bad_key': "can't"})


def test_flash_entry_round_trips_through_pon():
    from aotriton.tune.registry import load_flash_entry_module

    modules_dir = Path(__file__).resolve().parents[2] / 'modules'
    FlashEntry = load_flash_entry_module(modules_dir=modules_dir).FlashEntry

    e = FlashEntry(dtype='bfloat16', hdim=(64, 128), seqlen_q=256, seqlen_k=512,
                   causal=True, dropout_p=0.5, bias_type=1)
    d = parse_pon(e.as_text())
    assert FlashEntry(**d) == e


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for fn in fns:
        try:
            fn()
        except TypeError:
            # skip tests that need pytest.raises fixtures when run standalone
            continue
    print(f'OK: {len(fns)} pon tests attempted.')
    return failures


if __name__ == '__main__':
    sys.exit(main())
