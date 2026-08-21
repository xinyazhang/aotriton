# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""perfmon-exec0.md T04: golden-value + guard-string coverage for
modules/flash/perfmon/tflops.py, the Python port of
modules/flash/visperf/static/flash.js:20-29.

Precedent for the guard-string technique:
test_tune_infra.py's test_flash_entry_as_text_matches_codegen_copy (compares
two independent implementations of the same formula instead of just calling
one of them), generalized here to compare against the ORIGINAL SOURCE TEXT,
since flash.js has no importable Python twin to call.
"""

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / 'modules'
_FLASH_JS = _MODULES_DIR / 'flash' / 'visperf' / 'static' / 'flash.js'


def _tflops_module():
    from aotriton.tune.registry import load_family_perfmon
    from importlib import import_module
    mod = load_family_perfmon('flash', modules_dir=_MODULES_DIR)
    return import_module('.tflops', package=mod.__name__)


# --- (a) golden values, hand-computed, 6 significant digits -----------------

def test_attn_fwd_tflops_golden_non_causal():
    tflops = _tflops_module()
    # sq=sk=128, hdim=64, non-causal, batch=3, n_heads=5, 1ms/iter.
    # valid = 128*128 = 16384
    # flops_per_matmul = 2*16384*64 = 2097152
    # total = 2*2097152*3*5 = 62914560
    # tflops = 62914560 / 0.001 / 1e12 = 0.06291456
    got = tflops.attn_fwd_tflops(128, 128, 64, False, 0.001, 3, 5)
    assert got == pytest.approx(0.06291456, rel=1e-6)


def test_attn_fwd_tflops_golden_causal_square():
    tflops = _tflops_module()
    # sq=sk=128, hdim=64, causal, batch=3, n_heads=5, 1ms/iter.
    # valid = sq*sk - (sq^2-sq)/2 = 16384 - (16384-128)/2 = 16384 - 8128 = 8256
    # flops_per_matmul = 2*8256*64 = 1056768
    # total = 2*1056768*3*5 = 31703040
    # tflops = 31703040 / 0.001 / 1e12 = 0.03170304
    got = tflops.attn_fwd_tflops(128, 128, 64, True, 0.001, 3, 5)
    assert got == pytest.approx(0.03170304, rel=1e-6)


def test_attn_fwd_tflops_golden_causal_sq_greater_than_sk():
    tflops = _tflops_module()
    # sq=256, sk=128, hdim=128, causal, batch=2, n_heads=4, 2ms/iter.
    # sq > sk branch: valid = (sk^2+sk)/2 = (16384+128)/2 = 8256
    # flops_per_matmul = 2*8256*128 = 2113536
    # total = 2*2113536*2*4 = 33816576
    # tflops = 33816576 / 0.002 / 1e12 = 0.016908288
    got = tflops.attn_fwd_tflops(256, 128, 128, True, 0.002, 2, 4)
    assert got == pytest.approx(0.016908288, rel=1e-6)


def test_attn_bwd_tflops_golden_is_fwd_scaled_by_2p5():
    tflops = _tflops_module()
    # Same shape as the first golden but through the bwd formula: bwd divides
    # the time argument by 2.5 before calling the fwd formula, i.e.
    # tflops_bwd(seconds) == tflops_fwd(seconds/2.5) == 2.5 * tflops_fwd(seconds).
    fwd = tflops.attn_fwd_tflops(128, 128, 64, False, 0.001, 3, 5)
    bwd = tflops.attn_bwd_tflops(128, 128, 64, False, 0.001, 3, 5)
    assert bwd == pytest.approx(fwd * 2.5, rel=1e-6)
    assert bwd == pytest.approx(0.1572864, rel=1e-6)


# --- (b) guard test: fails if flash.js's formula text changes ----------------

def _extract_js_function_body(source: str, name: str) -> str:
    """Extract the brace-matched body of `function <name>(...) { ... }` from
    `source`. Both attnFwdTflops/_attnValidElements have no nested braces of
    their own inside the top-level function body except for their own control
    flow, so a simple depth counter (not a nested-function-aware parser) is
    enough -- this mirrors the plain substring/regex approach the codebase's
    other guard tests already use (test_flash_entry_as_text_matches_codegen_copy
    compares as_text() output directly rather than parsing Python ASTs)."""
    m = re.search(r'function\s+' + re.escape(name) + r'\s*\([^)]*\)\s*\{', source)
    assert m, f'{name} not found in flash.js'
    start = m.end()
    depth = 1
    i = start
    while depth > 0:
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
        i += 1
    return source[start:i - 1]


# Checked-in expected text (verbatim body, whitespace preserved) for
# `attnFwdTflops` and `_attnValidElements` as of the flash.js revision this
# port was written against. If flash.js's formula changes, this test fails
# loudly instead of tflops.py silently drifting from its source of truth.
_EXPECTED_ATTN_FWD_TFLOPS_BODY = '''
  const valid = _attnValidElements(seqlen_q, seqlen_k, causal);
  const flops_per_matmul = 2 * valid * hdim;
  const total_flops = 2 * flops_per_matmul * batch * n_heads;
  return total_flops / (median_ms * 1e-3) / 1e12;
'''

_EXPECTED_ATTN_VALID_ELEMENTS_BODY = '''
  if (causal) {
    return seqlen_q <= seqlen_k
      ? seqlen_q * seqlen_k - (seqlen_q * seqlen_q - seqlen_q) / 2
      : (seqlen_k * seqlen_k + seqlen_k) / 2;
  }
  return seqlen_q * seqlen_k;
'''

_EXPECTED_ATTN_BWD_TFLOPS_BODY = '''
  // backward = 2.5x forward FLOPs (2.0 bwd matmuls + 0.5 recompute)
  return attnFwdTflops(seqlen_q, seqlen_k, hdim, causal, median_ms / 2.5, batch, n_heads);
'''


def test_flash_js_attn_fwd_tflops_formula_text_unchanged():
    source = _FLASH_JS.read_text()
    body = _extract_js_function_body(source, 'attnFwdTflops')
    assert body == _EXPECTED_ATTN_FWD_TFLOPS_BODY, (
        'flash.js attnFwdTflops changed -- modules/flash/perfmon/tflops.py '
        'must be re-ported and this golden string updated')


def test_flash_js_attn_valid_elements_formula_text_unchanged():
    source = _FLASH_JS.read_text()
    body = _extract_js_function_body(source, '_attnValidElements')
    assert body == _EXPECTED_ATTN_VALID_ELEMENTS_BODY, (
        'flash.js _attnValidElements changed -- modules/flash/perfmon/tflops.py '
        'must be re-ported and this golden string updated')


def test_flash_js_attn_bwd_tflops_formula_text_unchanged():
    source = _FLASH_JS.read_text()
    body = _extract_js_function_body(source, 'attnBwdTflops')
    assert body == _EXPECTED_ATTN_BWD_TFLOPS_BODY, (
        'flash.js attnBwdTflops changed -- modules/flash/perfmon/tflops.py '
        'must be re-ported and this golden string updated')
