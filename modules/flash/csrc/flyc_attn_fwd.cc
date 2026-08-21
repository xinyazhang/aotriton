// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/_internal/pon.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/flyc.flyc_attn_fwd.h>
#include <flash/iface.op_attn_fwd.h>

#include "flyc_varlen.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

namespace {

// The four rows this shim can produce, keyed off the params struct alone --
// `VarlenType` itself never reaches here (attn_fwd.cc:99 already collapsed it
// into the sign of Num_seqlens by the time this context sees params).
struct FlycVarlenRow {
  int32_t varlen_bits;
  int32_t batch_size;
  int32_t num_seqlens;
};

// Classifies params into one row of the varlen table and, in the same place,
// checks the one invariant that makes the classification safe to trust: the
// kernel's own `nseq_idx = (num_seqlens != 0) ? num_seqlens : batch_size`
// must agree with what AOTriton computes independently from Num_seqlens and
// Q's batch axis. A silent disagreement here is not a crash -- it is a
// kernel launched over the wrong number of programs, addressing an
// in-bounds row that is simply the wrong one.
FlycVarlenRow
classify_varlen(const OpAttnFwdParams& params) {
  const int32_t Num_seqlens = params.Num_seqlens;
  const int32_t q_batch = static_cast<int32_t>(params.Q->size(0));
  const bool seq_strides_q_present = static_cast<bool>(*params.seq_strides_q);

  FlycVarlenRow row;
  if (Num_seqlens == 0) {
    // Dense: nothing packed, Q is BHSD with q_batch == B.
    row = { static_cast<int32_t>(kFlycVarlenDense), q_batch, 0 };
  } else if (Num_seqlens < 0) {
    // Padded varlen: BHSD, one sequence per batch slot, q_batch == N. The
    // count already arrived via Q's own batch axis, so num_seqlens is 0 --
    // FlyDSL's num_seqlens means "how many sequences are packed into a 1THD
    // tensor", and padded packs nothing.
    row = { static_cast<int32_t>(kFlycVarlenPadded), q_batch, 0 };
  } else if (!seq_strides_q_present) {
    // Compact varlen: 1THD, position reused from the cumulative length
    // array, so no seq_strides_q tensor is supplied.
    row = { static_cast<int32_t>(kFlycVarlenCompact), q_batch, Num_seqlens };
  } else {
    // Strided varlen: 1THD, position read from a dedicated array.
    row = { static_cast<int32_t>(kFlycVarlenStrided), q_batch, Num_seqlens };
  }

  const int32_t nseq_idx = row.num_seqlens != 0 ? row.num_seqlens : row.batch_size;
  const int32_t expected = Num_seqlens == 0 ? q_batch
                          : (Num_seqlens < 0 ? -Num_seqlens : Num_seqlens);
  if (nseq_idx != expected) {
    AOTRITON_LOG(LOG_ERROR,
                 "flyc varlen translation mismatch: kernel's nseq_idx would be %d "
                 "(batch_size=%d, num_seqlens=%d) but AOTriton independently computes "
                 "%d from Num_seqlens=%d, Q->size(0)=%d -- refusing to launch",
                 nseq_idx, row.batch_size, row.num_seqlens, expected, Num_seqlens, q_batch);
    throw std::runtime_error("flyc varlen translation mismatch (nseq_idx)");
  }
  return row;
}

} // anonymous namespace

int32_t
FlycAttnFwdContext::flyc_varlen_bits() const {
  return classify_varlen(*params).varlen_bits;
}

int32_t
FlycAttnFwdContext::flyc_batch_size() const {
  return classify_varlen(*params).batch_size;
}

int32_t
FlycAttnFwdContext::flyc_num_seqlens() const {
  return classify_varlen(*params).num_seqlens;
}

// The i32 dropout threshold. FlyDSL's `philox.dropout_threshold` (in
// modules/flash/flyc/philox.py, reached from fmha_abi_gfx1201.py's
// dropout_args): a uniform u32 reinterpreted as i32 is uniform on
// [-2**31, 2**31), so comparing against `(p - 0.5) * 0xFFFFFFFF` keeps a
// `1 - p` fraction with one signed compare and no per-element float
// conversion. This is bit-for-bit the same formula AOTriton's own Triton
// kernels already use (modules/flash/kernel/fwd_kernel.py and siblings:
// `((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)`), computed here in double
// precision and clamped rather than truncated by a narrowing cast, matching
// `dropout_threshold`'s own `max(-(2**31), min(2**31 - 1, t))`.
//
// When dropout is disabled, FlyDSL's `dropout_args` returns threshold 0 and
// scale 1.0 without evaluating the formula at all; mirrored here rather than
// evaluating it at p == 0, which is not the same value.
int32_t
FlycAttnFwdContext::flyc_idropout_p() const {
  if (!params->ENABLE_DROPOUT) {
    return 0;
  }
  const double p = static_cast<double>(params->dropout_p);
  const double raw = (p - 0.5) * static_cast<double>(0xFFFFFFFFu);
  const int64_t t = static_cast<int64_t>(raw);
  constexpr int64_t kInt32Min = -(int64_t{1} << 31);
  constexpr int64_t kInt32Max = (int64_t{1} << 31) - 1;
  return static_cast<int32_t>(std::clamp(t, kInt32Min, kInt32Max));
}

float
FlycAttnFwdContext::flyc_dropout_scale() const {
  if (!params->ENABLE_DROPOUT) {
    return 1.0f;
  }
  return 1.0f / (1.0f - params->dropout_p);
}

// Grid shape, derived from the vendored kernel's own launcher --
// `launch_flash_attn_aiw` in modules/flash/flyc/flash_attn_func_gfx1201_aiw.py:
//
//   nseq_idx    = num_seqlens != 0 ? num_seqlens : batch_size
//   num_q_tiles = cdiv(max_seqlen_q, BLOCK_M)
//   launcher.launch(grid=(num_head_q, num_q_tiles, nseq_idx), block=(BLOCK_SIZE, 1, 1))
//
// Unlike Triton's grid_calculator(), there is no persistent variant and no
// NUM_XCDS-dependent axis swap: flyc's grid is always (head, q_tile, seq) in
// that order. `block` is not reproduced here -- the block size comes from
// the aks2 directory entry (num_warps * warp_size), not from the caller; see
// TritonKernel::invoke.
dim3
FlycAttnFwdContext::grid_calculator() const {
  const auto row = classify_varlen(*params);
  const int32_t nseq_idx = row.num_seqlens != 0 ? row.num_seqlens : row.batch_size;

  const auto block_m_opt = perf().get_int("block_m");
  if (!block_m_opt) {
    AOTRITON_LOG(LOG_ERROR, "flyc grid_calculator: perf() is missing the 'block_m' key");
    throw std::runtime_error("flyc grid_calculator: missing 'block_m' in perf()");
  }
  const auto block_m = static_cast<uint32_t>(*block_m_opt);
  const auto num_q_tiles = AOTRITON_NS::cdiv<uint32_t>(static_cast<uint32_t>(params->Max_seqlen_q), block_m);

  return dim3 {
    static_cast<uint32_t>(params->Num_head_q),
    num_q_tiles,
    static_cast<uint32_t>(nseq_idx),
  };
}

} // namespace AOTRITON_NS::v3::flash
