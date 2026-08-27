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

// Item I sub-step (f). Rounds from the TRUE head dims (params->Hdim_qk /
// Hdim_vo), not from params->BLOCK_DMODEL -- that field already holds the
// OPERATOR's (Triton) rounding from attn_fwd.cc's own binning site, and
// re-rounding a value already on a SUPERSET ladder against this backend's
// SUBSET ladder is equivalent to rounding the true head dim directly only
// because the subset property holds; going straight to the true head dims
// avoids relying on that equivalence at every call site. See flyc_common.h's
// flyc_round_up_rung for the -1-on-nothing-compiled/nothing-fits contract.
//
// `get_archmod_number(current_gpu)` is how a helper reaches the arch index:
// it takes no arguments by design (flyc.h), and lookup_optimal() evaluates
// every context helper before it computes its OWN arch_number/mod_number
// pair (see the item I comment at that call site) -- so this cannot reuse a
// local, it must ask again.
int16_t
FlycAttnFwdContext::flyc_block_dmodel() const {
  const auto [arch_number, mod_number] = get_archmod_number(current_gpu);
  (void)mod_number;
  const int32_t hdim = std::max(params->Hdim_qk, params->Hdim_vo);
  if (arch_number < 0) {
    // No compiled rung table for this GPU at all. lookup_optimal() rejects
    // the launch (hipErrorNoBinaryForGpu) right after context helpers run,
    // using the same get_archmod_number(gpu) call, so this value is never
    // actually consulted -- return the true, unrounded max rather than
    // fabricate a rung.
    return static_cast<int16_t>(hdim);
  }
  return flyc_round_up_rung(hdim, compiled_block_dmodel[arch_number], compiled_block_dmodel_count[arch_number]);
}

// Must follow flyc_block_dmodel()'s own rounding decision, not re-derive one:
// a kernel re-rounded to a wider rung with PADDED_HEAD left false is a
// silent wrong answer (padding is never applied even though the compiled
// tile is wider than the true head dim), not a build error. Calls
// flyc_block_dmodel() itself rather than reading
// scratch_params.flyc_block_dmodel -- context helpers are evaluated in
// declaration order in lookup_optimal(), so that member IS already populated
// by the time this runs today, but relying on that order is exactly the
// silent-failure mode flyc.h's context_helper_evaluate comment warns about.
bool
FlycAttnFwdContext::flyc_padded_head() const {
  const int32_t hdim_qk = params->Hdim_qk;
  const int32_t hdim_vo = params->Hdim_vo;
  const int16_t rounded = flyc_block_dmodel();
  return rounded != hdim_qk || rounded != hdim_vo;
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
// NUM_XCDS-dependent axis swap. `block` is not reproduced here -- the block
// size comes from the aks2 directory entry (num_warps * warp_size), not from
// the caller; see TritonKernel::invoke.
//
// §4.4: the axis order itself is not hardcoded -- both arches' descriptions
// put HEAD_FASTEST in the sidecar for this kernel today (gfx1201 by literal
// assignment, gfx950 as `Gfx950Knobs.GRID_AXIS_ORDER`, a flat resolved field),
// but this reads perf() rather than assuming so, per flyc_common.h's
// flyc_grid_axis_order.
dim3
FlycAttnFwdContext::grid_calculator() const {
  const auto row = classify(*params);
  const auto axis_order = flyc_grid_axis_order(perf(), "flyc fwd");

  const auto block_m_opt = perf().get_int("block_m");
  if (!block_m_opt) {
    AOTRITON_LOG(LOG_ERROR, "flyc grid_calculator: perf() is missing the 'block_m' key");
    throw std::runtime_error("flyc grid_calculator: missing 'block_m' in perf()");
  }
  const auto block_m = static_cast<uint32_t>(*block_m_opt);
  const auto num_q_tiles = AOTRITON_NS::cdiv<uint32_t>(static_cast<uint32_t>(params->Max_seqlen_q), block_m);
  const auto num_head_q = static_cast<uint32_t>(params->Num_head_q);
  const auto nseq_idx = static_cast<uint32_t>(row.nseq_idx());

  switch (axis_order) {
    case kFlycGridAxisHeadFastest:
      return dim3 { num_head_q, num_q_tiles, nseq_idx };
    case kFlycGridAxisTileFastest:
      // No knob set has ever asked the forward for this order; fail loudly
      // rather than silently walk the axes in the wrong shape.
      AOTRITON_LOG(LOG_ERROR, "flyc fwd grid_calculator: TILE_FASTEST is not implemented for this kernel");
      throw std::runtime_error("flyc fwd grid_calculator: unsupported GRID_AXIS_ORDER (TILE_FASTEST)");
  }
  throw std::runtime_error("flyc fwd grid_calculator: unreachable GRID_AXIS_ORDER");
}

} // namespace AOTRITON_NS::v3::flash
