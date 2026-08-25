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

#include "flyc_common.h"

#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

namespace {

// `VarlenType` itself never reaches here -- attn_fwd.cc:99 already collapsed it
// into the sign of Num_seqlens by the time this context sees params. Note the
// capitalised field names: the forward operator spells them `Num_seqlens` and
// `Q` where the backward spells the first one lowercase, which is why
// flyc_classify_varlen takes scalars rather than a params struct.
FlycVarlenRow
classify(const OpAttnFwdParams& params) {
  return flyc_classify_varlen(params.Num_seqlens,
                              static_cast<int32_t>(params.Q->size(0)),
                              static_cast<bool>(*params.seq_strides_q));
}

} // anonymous namespace

int32_t
FlycAttnFwdContext::flyc_varlen_bits() const {
  return classify(*params).varlen_bits;
}

int32_t
FlycAttnFwdContext::flyc_num_seqlens() const {
  return classify(*params).num_seqlens;
}

int32_t
FlycAttnFwdContext::flyc_idropout_p() const {
  return flyc_idropout_p_of(params->ENABLE_DROPOUT, params->dropout_p);
}

float
FlycAttnFwdContext::flyc_dropout_scale() const {
  return flyc_dropout_scale_of(params->ENABLE_DROPOUT, params->dropout_p);
}

// Grid shape, derived from the vendored kernel's own launcher --
// `launch_flash_attn_aiw` in modules/flash/flyc/flash_attn_func_gfx1201_aiw.py:
//
//   nseq_idx    = num_seqlens != 0 ? num_seqlens : batch_size
//   num_q_tiles = cdiv(max_seqlen_q, BLOCK_M)
//   launcher.launch(grid=(num_head_q, num_q_tiles, nseq_idx), block=(BLOCK_SIZE, 1, 1))
//
// `batch_size` is no longer a kernarg (FlyDSL 67a3ace0) but is still needed
// here: it is half of nseq_idx, and the launcher that used to receive it is
// exactly the code this function replaces.
//
// Unlike Triton's grid_calculator(), there is no persistent variant and no
// NUM_XCDS-dependent axis swap: flyc's forward grid is always (head, q_tile,
// seq) in that order. `block` is not reproduced here -- the block size comes
// from the aks2 directory entry (num_warps * warp_size), not from the caller;
// see TritonKernel::invoke.
dim3
FlycAttnFwdContext::grid_calculator() const {
  const auto row = classify(*params);

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
    static_cast<uint32_t>(row.nseq_idx()),
  };
}

} // namespace AOTRITON_NS::v3::flash
