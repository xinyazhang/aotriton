// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/_internal/pon.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/flyc.flyc_bwd_dkdv.h>
#include <flash/iface.op_attn_bwd.h>

#include "flyc_common.h"

#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

namespace {

// Note the lowercase field names: op_attn_bwd spells `num_seqlens` where
// op_attn_fwd spells `Num_seqlens`, which is why flyc_classify_varlen takes
// scalars rather than a params struct.
FlycVarlenRow
classify(const OpAttnBwdParams& params) {
  return flyc_classify_varlen(params.num_seqlens,
                              static_cast<int32_t>(params.Q->size(0)),
                              static_cast<bool>(*params.seq_strides_q));
}

} // anonymous namespace

int32_t
FlycBwdDkdvContext::flyc_varlen_bits() const {
  return classify(*params).varlen_bits;
}

int32_t
FlycBwdDkdvContext::flyc_num_seqlens() const {
  return classify(*params).num_seqlens;
}

// The backward's dropout must reproduce the forward's mask exactly, so these
// two are the same translation the forward uses, from the same header -- not a
// backward-specific variant.
int32_t
FlycBwdDkdvContext::flyc_idropout_p() const {
  return flyc_idropout_p_of(params->ENABLE_DROPOUT, params->dropout_p);
}

float
FlycBwdDkdvContext::flyc_dropout_scale() const {
  return flyc_dropout_scale_of(params->ENABLE_DROPOUT, params->dropout_p);
}

// Grid shape, derived from the vendored kernel's own launcher -- `launch_bwd_dkdv`
// in modules/flash/flyc/fmha_bwd_dkdv_gfx1201_kernel.py:
//
//   nseq_idx     = num_seqlens != 0 ? num_seqlens : batch_size
//   num_kv_tiles = cdiv(max_seqlen_k, BLOCK_N)
//   launcher.launch(grid=(num_kv_tiles, num_head_k, nseq_idx), block=(BLOCK_SIZE, 1, 1))
//
// Not the forward's axis order, and not the dQ kernel's either: dK/dV walks KV
// tiles for a fixed Q range, so the tile count is x and the head axis is y, and
// the head is num_head_K -- one workgroup per KV head, with the GQA query heads
// summed inside. Writing (head, tile, seq) here as the forward does would run
// the right number of workgroups over the wrong blocks.
//
// BLOCK_N, not BLOCK_M, and it is NOT one of resolve_knobs' outputs: the builder
// derives it as `ROWS_PER_WAVE * NUM_TEAMS`. The ATI description recomputes it
// from the knobs and puts it in the sidecar under 'block_n'; see the comment at
// the bottom of modules/flash/aot/flyc_bwd_dkdv.py.
dim3
FlycBwdDkdvContext::grid_calculator() const {
  const auto row = classify(*params);

  const auto block_n_opt = perf().get_int("block_n");
  if (!block_n_opt) {
    AOTRITON_LOG(LOG_ERROR, "flyc dkdv grid_calculator: perf() is missing the 'block_n' key");
    throw std::runtime_error("flyc dkdv grid_calculator: missing 'block_n' in perf()");
  }
  const auto block_n = static_cast<uint32_t>(*block_n_opt);
  const auto num_kv_tiles =
      AOTRITON_NS::cdiv<uint32_t>(static_cast<uint32_t>(params->max_seqlen_k), block_n);

  return dim3 {
    num_kv_tiles,
    static_cast<uint32_t>(params->num_head_k),
    static_cast<uint32_t>(row.nseq_idx()),
  };
}

} // namespace AOTRITON_NS::v3::flash
