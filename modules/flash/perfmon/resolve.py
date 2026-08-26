# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Worker-side entry resolution for flash perfmon: `FlashEntry` (queued, D03)
-> `FlashInputMetadata` (resolved, what the GPU worker actually measures),
with perfmon's own byte-exact memory model (dispatch-perfmon-exec.md D05).

Purely additive: this module does NOT touch `modules/flash/tune/desc.py`.
Perfmon does not reuse tuning's `_clamp_memory_usage` -- that formula is an
empirical fudge because the tuning path allocates reference tensors (for
accuracy comparison) whose count/size depend on the test cases being
compared, and it runs under torch, whose caching allocator hides memory the
formula cannot see. Perfmon has neither: the runner is C++, does zero
accuracy checking, and allocates no reference tensors at all -- every device
buffer it makes is visible in
`modules/flash/perfmon/runner/head/adapter.cc`. So the model here is an
exact byte count instead of a curve fit. It also means this module must stay
free of any torch dependency: tuning's `_clamp_memory_usage` imports torch at
the end of its body, and the perfmon image installs no torch on purpose
(`create_perfmon_dockerfile.sh:31`).

Byte accounting (read off `adapter.cc`; `round_to_8x` from
`runner/lib/common.cc:15`):

    dt        = 2 for float16/bfloat16, 4 for float32
    HQ        = N_HEADS (query heads); HK = HQ // gqa_ratio (kv heads)
    q_elems   = B * HQ * SQ * D            # anything shaped like Q
    k_elems   = B * HK * SK * D            # anything shaped like K/V
    lse_elems = B * HQ * SQ                # L and D are rank-2 (B, HQ*SQ), fp32
    bias_elems= B * HQ * SQ * round_to_8x(SK)

`attn_fwd` (`build_fwd`): Q, K, V, Out, L, optional bias, 4-byte counter.

    fwd = dt * (2*q_elems + 2*k_elems)     # Q, Out | K, V
        + 4  * lse_elems                   # L, fp32
        + (dt * bias_elems if bias_type else 0)

`attn_bwd` (`build_bwd`): Q, K, V, Out, DO, DK, DV, DQ, L, D, optional B+DB,
and fp32 `DQ_ACC`.

    bwd = dt * (4*q_elems + 4*k_elems)     # Q, Out, DO, DQ | K, V, DK, DV
        + 4  * (2 * lse_elems)             # L and D, both fp32
        + (dt * 2 * bias_elems if bias_type else 0)   # B and DB
        + 4  * q_elems                     # DQ_ACC -- fp32, shaped like Q

`DQ_ACC` is fp32 regardless of the entry's dtype (twice what Q costs on a
bf16 entry), and is counted ALWAYS here even though `build_bwd` only
allocates it for backend 2 (gfx942/gfx950): one queue row is one entry, and
the DAG fans out over (iface, backend) on the same worker, so resolving once
per entry against the worst case keeps every measurement of it comparable
(see dispatch-perfmon-exec.md D05).

Budget: `usable = vram_total_gb * 2**30 * PERFMON_VRAM_FRACTION -
L2_FLUSH_BYTES` (the flush buffer is core's, allocated once per process, not
per-entry -- subtracted, not scaled). A quarter of VRAM is left unused
deliberately: the model is an exact count of what the adapter allocates, not
of what the process occupies (HIP context, allocator fragmentation, the
kernel's own workspace, loaded kernel images are all outside it). Do not
raise `PERFMON_VRAM_FRACTION` without measured evidence from a real run.
"""

from __future__ import annotations

import dataclasses

from aotriton.tune.registry import load_flash_entry_module

_flash_entry_module = load_flash_entry_module()
FlashEntry = _flash_entry_module.FlashEntry
FlashInputMetadata = _flash_entry_module.FlashInputMetadata

#: Use at most 75% of VRAM; leave 25% unused (see module docstring).
PERFMON_VRAM_FRACTION = 0.75

#: perfmon/core/timing.h:58, kL2FlushBytes -- allocated once per process,
#: not part of the per-entry context, so it is subtracted from the budget
#: rather than scaled with it.
L2_FLUSH_BYTES = 1 << 30


def round_to_8x(n: int) -> int:
    """Matches `runner/lib/common.cc:15` exactly: the bias tensor is
    allocated at this padded width and only *viewed* as `SK`, so the padded
    size is what occupies VRAM."""
    return 8 * ((n + 7) // 8)


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype == 'float32' else 2


def _heads(n_heads) -> tuple[int, int]:
    """(HQ, HK): query/kv head counts. Plain int N_HEADS means HQ == HK; a
    GQA `(q, kv)` tuple gives them independently."""
    if isinstance(n_heads, tuple):
        return n_heads[0], n_heads[1]
    return n_heads, n_heads


def _q_elems(im) -> int:
    hq, _ = _heads(im.N_HEADS)
    return im.BATCH * hq * im.seqlen_q * im.hdim


def _k_elems(im) -> int:
    _, hk = _heads(im.N_HEADS)
    return im.BATCH * hk * im.seqlen_k * im.hdim


def _lse_elems(im) -> int:
    hq, _ = _heads(im.N_HEADS)
    return im.BATCH * hq * im.seqlen_q


def _bias_elems(im) -> int:
    hq, _ = _heads(im.N_HEADS)
    return im.BATCH * hq * im.seqlen_q * round_to_8x(im.seqlen_k)


def fwd_bytes(im) -> int:
    """Exact byte count of every device buffer `build_fwd` (adapter.cc)
    allocates for one `attn_fwd` measurement of resolved metadata `im`."""
    dt = _dtype_bytes(im.dtype)
    total = dt * (2 * _q_elems(im) + 2 * _k_elems(im)) + 4 * _lse_elems(im)
    if im.bias_type:
        total += dt * _bias_elems(im)
    return total


def bwd_bytes(im, *, count_dq_acc: bool = True) -> int:
    """Exact byte count of every device buffer `build_bwd` (adapter.cc)
    allocates for one `attn_bwd` measurement of resolved metadata `im`.

    `count_dq_acc` defaults to True and should stay True everywhere except
    the D05 Verify script, which uses False only to isolate DQ_ACC's own
    contribution -- see the module docstring for why DQ_ACC is always
    counted in real use even though only backend 2 allocates it."""
    dt = _dtype_bytes(im.dtype)
    q = _q_elems(im)
    total = dt * (4 * q + 4 * _k_elems(im)) + 4 * (2 * _lse_elems(im))
    if im.bias_type:
        total += dt * 2 * _bias_elems(im)
    if count_dq_acc:
        total += 4 * q
    return total


def usable_bytes(vram_total_gb: float) -> float:
    """Bytes resolve_entry() may fill for one entry's worst-case (bwd)
    measurement, on a GPU reporting `vram_total_gb` GiB total VRAM."""
    return vram_total_gb * (1 << 30) * PERFMON_VRAM_FRACTION - L2_FLUSH_BYTES


def _n_heads_at_floor(n_heads) -> bool:
    if isinstance(n_heads, tuple):
        return n_heads[1] <= 1
    return n_heads <= 1


def _halve_n_heads(n_heads):
    """Halve, never step through magic numbers: n, n//2, n//4, ..., 1.
    GQA keeps its ratio: halve kv and set q = kv * (q_orig // kv_orig), so
    q % kv == 0 still holds -- the adapter's kv_heads_of() raises on
    non-divisibility otherwise."""
    if isinstance(n_heads, tuple):
        q, kv = n_heads
        ratio = q // kv
        kv = max(1, kv // 2)
        return (kv * ratio, kv)
    return max(1, n_heads // 2)


def resolve_entry(entry, vram_total_gb: float | None):
    """FlashEntry -> FlashInputMetadata for one worker's GPU.

    Deterministic given (entry, vram_total_gb): the same node must resolve
    the same entry identically on every run, or repeat measurements are not
    comparable. Nothing here reads a clock, a PRNG or the environment.
    """
    base = {f.name: getattr(entry, f.name) for f in dataclasses.fields(FlashEntry)}
    im = FlashInputMetadata(**base)

    if vram_total_gb is None:
        # Unknown VRAM must not silently shrink the workload.
        return im

    budget = usable_bytes(vram_total_gb)
    n_heads = im.N_HEADS
    batch = im.BATCH

    while True:
        candidate = dataclasses.replace(im, N_HEADS=n_heads, BATCH=batch)
        if bwd_bytes(candidate) <= budget:
            return candidate

        # Heads first, then batch -- the same precedence tuning uses.
        if not _n_heads_at_floor(n_heads):
            n_heads = _halve_n_heads(n_heads)
            continue
        if batch > 1:
            batch = max(1, batch // 2)
            continue

        # Floored on both axes and it still does not fit: this entry is not
        # measurable on this GPU. A loud failure at prepare/resolve time is
        # correct -- the runtime counterpart of D04's dispatch-time
        # exclusion; silently measuring something else would be worse.
        raise RuntimeError(
            f"resolve_entry: {entry!r} does not fit in {budget:.0f} usable "
            f"VRAM bytes (vram_total_gb={vram_total_gb}) even at the floor "
            f"(N_HEADS={n_heads!r}, BATCH=1); not measurable on this GPU.")
