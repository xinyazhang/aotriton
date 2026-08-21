# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash attention TFLOPS formulas -- direct port of
`modules/flash/visperf/static/flash.js:20-29` (perfmon-exec0.md T03).

Counts only the two matmuls (QK^T and PV); excludes softmax, LSE, dropout --
identical scope to the JS source (itself sourced from ROCm/triton
perf-kernels/flash-attention.py @ 0ec280cf, bench_flash_attention()).

DEVIATION FROM perfmon-exec0.md's T03 PSEUDOCODE, REPORTED PER THAT TASK'S
OWN INSTRUCTION ("Read flash.js and match it line for line, including the
`valid` computation for the causal case. Any intentional divergence must be
reported, not made silently."):

T03's inline pseudocode approximates the causal case as
`valid = seqlen_q * seqlen_k * 0.5`. The actual `flash.js` (`_attnValidElements`,
lines 11-18) does NOT use that approximation -- it computes the exact count of
non-masked (row, col) pairs under a causal mask:

    causal, seqlen_q <= seqlen_k:  seqlen_q*seqlen_k - (seqlen_q^2 - seqlen_q)/2
    causal, seqlen_q >  seqlen_k:  (seqlen_k^2 + seqlen_k) / 2
    non-causal:                    seqlen_q * seqlen_k

These agree only in the large-seqlen limit (e.g. sq=sk=128: exact 8256 vs
`*0.5` approximation 8192, a ~0.4% difference -- larger for smaller seqlens).
This module ports the ACTUAL flash.js formula verbatim, per T03's explicit
authority ("that file ... is the authority"), not the plan's illustrative
snippet.
"""

from __future__ import annotations


def _attn_valid_elements(seqlen_q: int, seqlen_k: int, causal: bool) -> float:
    """Port of flash.js `_attnValidElements` (lines 11-18), verbatim."""
    if causal:
        if seqlen_q <= seqlen_k:
            return seqlen_q * seqlen_k - (seqlen_q * seqlen_q - seqlen_q) / 2
        return (seqlen_k * seqlen_k + seqlen_k) / 2
    return seqlen_q * seqlen_k


def attn_fwd_tflops(seqlen_q: int, seqlen_k: int, hdim: int, causal: bool,
                     seconds: float, batch: int, n_heads: int) -> float:
    """Port of flash.js `attnFwdTflops` (lines 20-25). `seconds` is the
    per-iteration wall time in SECONDS (flash.js's `median_ms` is
    milliseconds; the `* 1e-3` there is folded into this signature instead)."""
    valid = _attn_valid_elements(seqlen_q, seqlen_k, causal)
    flops_per_matmul = 2 * valid * hdim
    total_flops = 2 * flops_per_matmul * batch * n_heads
    return total_flops / seconds / 1e12


def attn_bwd_tflops(seqlen_q: int, seqlen_k: int, hdim: int, causal: bool,
                     seconds: float, batch: int, n_heads: int) -> float:
    """Port of flash.js `attnBwdTflops` (lines 27-29): backward = 2.5x
    forward FLOPs (2.0 bwd matmuls + 0.5 recompute), expressed by scaling the
    forward formula's time argument down by 2.5 (matching flash.js calling
    attnFwdTflops with `median_ms / 2.5`)."""
    return attn_fwd_tflops(seqlen_q, seqlen_k, hdim, causal, seconds / 2.5,
                            batch, n_heads)
