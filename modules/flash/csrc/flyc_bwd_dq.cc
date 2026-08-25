// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/_internal/pon.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/flyc.flyc_bwd_dq.h>
#include <flash/iface.op_attn_bwd.h>

#include "flyc_common.h"

#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

namespace {

FlycVarlenRow
classify(const OpAttnBwdParams& params) {
  return flyc_classify_varlen(params.num_seqlens,
                              static_cast<int32_t>(params.Q->size(0)),
                              static_cast<bool>(*params.seq_strides_q));
}

} // anonymous namespace

int32_t
FlycBwdDqContext::flyc_varlen_bits() const {
  return classify(*params).varlen_bits;
}

int32_t
FlycBwdDqContext::flyc_num_seqlens() const {
  return classify(*params).num_seqlens;
}

int32_t
FlycBwdDqContext::flyc_idropout_p() const {
  return flyc_idropout_p_of(params->ENABLE_DROPOUT, params->dropout_p);
}

float
FlycBwdDqContext::flyc_dropout_scale() const {
  return flyc_dropout_scale_of(params->ENABLE_DROPOUT, params->dropout_p);
}

// Grid shape, derived from the vendored kernel's own launcher -- `launch_bwd_dq`
// in modules/flash/flyc/fmha_bwd_dq_gfx1201_kernel.py:
//
//   nseq_idx    = num_seqlens != 0 ? num_seqlens : batch_size
//   num_q_tiles = cdiv(max_seqlen_q, BLOCK_M)
//   launcher.launch(grid=(num_head_q, num_q_tiles, nseq_idx), block=(BLOCK_SIZE, 1, 1))
//
// This one DOES match the forward's shape -- (head, q_tile, seq) with the query
// head count -- because like the forward it walks Q tiles. Its sibling dK/dV
// does not; see the note there.
//
// `block_m` is a resolve_knobs output here, so unlike dK/dV's block_n it needs
// no derivation on the description side.
dim3
FlycBwdDqContext::grid_calculator() const {
  const auto row = classify(*params);

  const auto block_m_opt = perf().get_int("block_m");
  if (!block_m_opt) {
    AOTRITON_LOG(LOG_ERROR, "flyc dq grid_calculator: perf() is missing the 'block_m' key");
    throw std::runtime_error("flyc dq grid_calculator: missing 'block_m' in perf()");
  }
  const auto block_m = static_cast<uint32_t>(*block_m_opt);
  const auto num_q_tiles =
      AOTRITON_NS::cdiv<uint32_t>(static_cast<uint32_t>(params->max_seqlen_q), block_m);

  return dim3 {
    static_cast<uint32_t>(params->num_head_q),
    num_q_tiles,
    static_cast<uint32_t>(row.nseq_idx()),
  };
}

} // namespace AOTRITON_NS::v3::flash
