# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd dQ/dB FlyDSL backend (gfx1201).

The dQ/dB half of the flyc backward, and the sibling of `flyc_bwd_dkdv.py` --
see that file for the shared rationale (no perf axes, no functional axes of its
own, declared kernarg ABI, and why the backend is a metro with a Triton
`bwd_preprocess` in front).

Two things are specific to this kernel:

* **It writes dB.** dB is `dS`, which the kernel already forms for its last
  GEMM. dK/dV has the same quantity but walks Q tiles for a fixed KV block, so
  its eight elements are eight q *rows* at one kv column and the store would go
  down a dB column; this kernel writes along a row instead. That is why dB is
  the dQ kernel's output in both DSLs, and it is not gated: `bias=True` always
  emits it (FlyDSL 077626cd dropped the `return_dbias` knob), matching
  `bwd_kernel_dq.py`'s unconditional DB operand.
* **`block_m` and `block_n` are both real knobs** here, so nothing has to be
  re-derived for the grid the way `flyc_bwd_dkdv.py` re-derives `block_n`.
"""

from dataclasses import asdict

import aotriton.template_instantiation as ati
from ._flyc_common import FlycBwdHints, flyc_bwd_disabled


# The compiled tile ladder, from fmha_tuning_bwd_dq_gfx1201._BLOCK_DMODEL_LADDER.
# A literal, not an import, for the reason flyc_bwd_dkdv.py records. Narrower
# than dK/dV's -- no 320 or 448 -- which is exactly why the ladder is a per-kernel
# constant rather than something shared in _flyc_common.py.
FLYC_BWD_DQ_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224,
                                   256, 384, 512})


def _flyc_bwd_dq_disabled(f):
    return flyc_bwd_disabled(f, head_dims=FLYC_BWD_DQ_HEAD_DIMS)


@ati.start
@ati.disable(when=_flyc_bwd_dq_disabled,
             I_understand_this_overrides_cited_disable=True)
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dq')
#
# --- the kernarg ABI, in `bwd_dq_kernel` order -------------------------------
#
# The kernel, NOT the Python launcher: the two differ by `stream` and by
# `batch_size` (see flyc_bwd_dkdv.py). Same convention as the forward and dK/dV
# since FlyDSL 1b58fb93 -- inputs, outputs, LSE/Delta, seqinfo, scalars, strides
# -- with (DQ, DB) where dK/dV has (DK, DV).
@ati.tensor('Q',     'T_io', rank=4, strides='stride_q_*',  wires_to='Q')
@ati.tensor('K',     'T_io', rank=4, strides='stride_k_*',  wires_to='K')
@ati.tensor('V',     'T_io', rank=4, strides='stride_v_*',  wires_to='V')
@ati.tensor('B',     'T_io', rank=4, strides='stride_b_*',  wires_to='B')
@ati.tensor('DO',    'T_io', rank=4, strides='stride_do_*', wires_to='DO')
@ati.tensor('DQ',    'T_io', rank=4, strides='stride_dq_*', wires_to='DQ')
# DB has its OWN stride triple, not B's. FlyDSL made that split deliberately
# (fmha_abi_gfx1201.bias_args: "B and DB are separate tensors and may be laid
# out differently ... deriving one from the other happens to work whenever both
# are contiguous, and writes to the wrong addresses the moment either is a
# view"). `stride_b_*` does not match `stride_db_*`, so the two globs stay
# disjoint on their own.
@ati.tensor('DB',    'T_io', rank=4, strides='stride_db_*', wires_to='DB')
@ati.tensor('LSE',   '*fp32:16', rank=2, wires_to='L')
@ati.tensor('Delta', '*fp32:16', rank=2, wires_to='D')
@ati.tensor('seqinfo_q0', '*i32:16', rank=1, wires_to='cu_seqlens_q')
@ati.tensor('seqinfo_q1', '*i32:16', rank=1, wires_to='seq_strides_q')
@ati.tensor('seqinfo_k0', '*i32:16', rank=1, wires_to='cu_seqlens_k')
@ati.tensor('seqinfo_k1', '*i32:16', rank=1, wires_to='seq_strides_k')
@ati.scalar('varlen_bits', 'i32', wires_to=ati.context_helper('flyc_varlen_bits'))
@ati.scalar('num_seqlens', 'i32', wires_to=ati.context_helper('flyc_num_seqlens'))
@ati.scalar('max_seqlen_q', 'i32', wires_to='max_seqlen_q')
@ati.scalar('max_seqlen_k', 'i32', wires_to='max_seqlen_k')
@ati.scalar('window_left',  'i32', wires_to='Window_left')
@ati.scalar('window_right', 'i32', wires_to='Window_right')
@ati.tensor('philox_seed_ptr', '*u64', rank=0)
@ati.tensor('philox_offset1', '*u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
@ati.scalar('idropout_p',    'i32',  wires_to=ati.context_helper('flyc_idropout_p'))
@ati.scalar('dropout_scale', 'fp32', wires_to=ati.context_helper('flyc_dropout_scale'))
@ati.scalar('num_head_q', 'i32',  wires_to='num_head_q')
@ati.scalar('num_head_k', 'i32',  wires_to='num_head_k')
@ati.scalar('hdim_qk',    'i32',  wires_to='hdim_qk')
@ati.scalar('hdim_vo',    'i32',  wires_to='hdim_vo')
@ati.scalar('sm_scale',   'fp32', wires_to='sm_scale')
@ati.flyc.hints(FlycBwdHints)
@ati.flyc.kernel()
def flyc_bwd_dq(arch, choices, hints):
    """Build one dQ/dB hsaco for the functional described by `choices`.

    Build-time only (`aotriton.flyc_compile`); the generator calls this for its
    knobs and never calls the returned `build`. See `flyc_attn_fwd.py` for the
    `choices` / `hints` contract.
    """
    from fmha_tuning_bwd_dq_gfx1201 import (
        BwdDqInputMetadata, BwdDqKnobs, resolve_knobs,
    )

    tile = choices.BLOCK_DMODEL
    meta = BwdDqInputMetadata(
        # Unread by resolve_knobs, whose policy keys on head_dim and causal.
        num_heads=1,
        head_dim=tile,
        causal=choices.CAUSAL_TYPE != 0,
        causal_type=choices.CAUSAL_TYPE,
        dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
        bias=bool(choices.BIAS_TYPE),
        dropout=bool(choices.ENABLE_DROPOUT),
        # philox_width left at None -- Philox.for_arch(), matching the forward.
        # The backward has to reproduce the forward's mask exactly.
    )
    knobs = resolve_knobs(meta, BwdDqKnobs(
        block_dmodel=tile,
        padded_head=choices.PADDED_HEAD,
    ))
    assert knobs.block_dmodel == tile, (
        f'resolve_knobs returned block_dmodel={knobs.block_dmodel} for '
        f'BLOCK_DMODEL={tile}; the compiled tile must be the operator axis')

    def build():
        """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
        so ONLY `aotriton.flyc_compile` (run by ninja) may call this."""
        from fmha_bwd_dq_gfx1201_kernel import build_bwd_dq_module_primary
        return build_bwd_dq_module_primary(meta, knobs)

    # Two plain strings (item D): the vendored file, relative to
    # modules/flash/flyc/, and the `@flyc.kernel` def's own name inside it.
    # Read by codegen/flytune.py and flyc_compile.py off the `build` closure
    # WITHOUT ever calling it.
    build.flyc_source = 'fmha_bwd_dq_gfx1201_kernel.py'
    build.flyc_kernel_name = 'bwd_dq_kernel'

    # `block_m` is already a knob here, so the sidecar needs nothing added --
    # unlike flyc_bwd_dkdv.py, which has to re-derive its grid's block_n.
    return build, asdict(knobs)
