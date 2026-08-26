# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash-specific perfmon block (perfmon-exec0.md T03).

Loaded by `aotriton.tune.registry.load_family_perfmon('flash', ...)`, by
path, under the synthetic name `_aotriton_modules_flash_perfmon` -- same
rationale as `modules/flash/tune/__init__.py` / `modules/flash/visperf/
__init__.py` (D8 in perfmon-rev0.md): `modules/flash` must stay a plain
directory, not a package.

Exports `PerfDesc`, a concrete `aotriton.tune.perfmon.pdesc.PerfDescription`.
Torch-free at import time and at construction (`PerfDesc()` takes no
arguments and does no GPU/torch/pyaotriton work) -- see pdesc.py's module
docstring for why this matters to `perfmon dispatch`.
"""

from __future__ import annotations

from aotriton.tune.perfmon.pdesc import PerfDescription
from aotriton.utils.pon import render_pon

from . import entry as _entry
from . import tflops as _tflops


#: Largest seqlen each arch can hold. Absent -> no limit.
#: Populated from measurement, not guesswork -- an arch missing here is
#: unconstrained, which is the safe direction (it fails loudly at run time
#: rather than silently skipping coverage).
_MAX_SEQLEN_BY_ARCH: dict[str, int] = {}


def _n_heads_for_tflops(entry) -> int:
    """FLOPs scale with the number of QUERY heads; for GQA (`N_HEADS` a
    `(num_q_heads, num_kv_heads)` tuple) that is element 0 -- the same
    convention `modules/flash/tune/desc.py`'s `_clamp_memory_usage` already
    uses (`n_heads = im.N_HEADS[0] if is_gqa else im.N_HEADS`)."""
    n_heads = entry.N_HEADS
    return n_heads[0] if isinstance(n_heads, tuple) else n_heads


class PerfDesc(PerfDescription):
    ENTRY_CLASS    = _entry.FlashEntry
    INPUT_METADATA = _entry.FlashInputMetadata

    #: Bare op-level interface names perfmon measures end-to-end (D3) --
    #: matches `modules/flash/tune/level_op.py:list_impls`.
    IFACES = ('attn_fwd', 'attn_bwd')

    #: Optional per-run ceiling, set by the caller (e.g. dispatch's
    #: --max_seqlen), combined with _MAX_SEQLEN_BY_ARCH in
    #: validate_hw_feature(). None (the default) means no caller-supplied
    #: limit.
    max_seqlen: int | None = None

    def validate_hw_feature(self, arch: str, entry) -> tuple[bool, str]:
        """Enforce the per-GPU sequence-length ceiling (rev0 §6: 16k is too
        large for some GPUs). The effective ceiling is the smaller of
        _MAX_SEQLEN_BY_ARCH[arch] (measured, currently empty -- see that
        table's docstring) and self.max_seqlen (an optional caller-supplied
        limit); either or both may be absent, in which case there is no
        ceiling from that source."""
        limits = [v for v in (self.max_seqlen, _MAX_SEQLEN_BY_ARCH.get(arch))
                  if v is not None]
        if not limits:
            return True, ''
        limit = min(limits)
        if entry.seqlen_q > limit or entry.seqlen_k > limit:
            return False, f"seqlen > {limit} exceeds this arch's ceiling"
        return True, ''

    def prime_entries(self, arch: str, max_seqlen: int):
        yield from _entry.prime_entries(max_seqlen)

    def coverage_entries(self, arch: str, max_seqlen: int):
        yield from _entry.coverage_entries(max_seqlen)

    def list_ifaces(self) -> list[str]:
        return list(self.IFACES)

    def functional_pon(self, entry, iface: str) -> str:
        """Takes the RESOLVED metadata, not a queued entry. N_HEADS/BATCH are
        chosen on the GPU worker (D05), so this is a report-time call."""
        d = {
            'iface': iface,
            'dtype': entry.dtype,
            'causal': bool(entry.causal),
            'dropout_p': entry.dropout_p,
            'bias_type': entry.bias_type,
            'gqa': not isinstance(entry.N_HEADS, int),
            # No varlen field exists on FlashEntry/FlashInputMetadata yet --
            # see entry.py's "KNOWN GAPS" docstring note. Emitted as a
            # constant so the on-disk PON shape is forward-compatible with a
            # future varlen axis without a format change.
            'varlen': False,
            'storage_flip': bool(entry.storage_flip),
        }
        return render_pon(d)

    def shape_pon(self, entry) -> str:
        """Takes the RESOLVED metadata, not a queued entry. N_HEADS/BATCH are
        chosen on the GPU worker (D05), so this is a report-time call."""
        d = {
            'hdim': entry.hdim,
            'seqlen_q': entry.seqlen_q,
            'seqlen_k': entry.seqlen_k,
            'BATCH': entry.BATCH,
            'N_HEADS': entry.N_HEADS,
        }
        return render_pon(d)

    def tflops(self, entry, iface: str, seconds: float) -> float:
        fn = _tflops.attn_bwd_tflops if iface == 'attn_bwd' else _tflops.attn_fwd_tflops
        return fn(entry.seqlen_q, entry.seqlen_k, entry.hdim, bool(entry.causal),
                   seconds, entry.BATCH, _n_heads_for_tflops(entry))
