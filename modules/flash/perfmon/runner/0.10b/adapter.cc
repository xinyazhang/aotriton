// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// flash adapter for 0.10b -- the first tag with a v3 API, and the only one
// where that API cannot select a backend.
//
// --- Differences from the 0.11 line ---------------------------------------
// Read from 0.10b's own include/aotriton/flash.h:
//
//  1. `attn_options` is EMPTY:
//
//         struct AOTRITON_API attn_options { };
//
//     `force_backend_index` arrives at 0.11b. There is therefore NO way to
//     ask this release for a specific backend -- the dispatcher's own choice
//     is the only thing measurable. enumerate_backends() below reports
//     exactly ONE backend for both ifaces, rather than the 2/3 that
//     modules/flash/tune/level_op.py specifies for later tags.
//
//     This is a real, disclosed coverage limitation of this subject, not a
//     shortcut: rows for 0.10b carry one measurement per (iface, entry)
//     where later tags carry two or three, and that single number is
//     "whatever 0.10b's dispatcher picked", which is also what a user of
//     0.10b would actually have got. Reporting 2/3 here and passing an
//     option this release ignores would silently produce two or three
//     IDENTICAL measurements labelled as different backends -- far worse
//     than one honest row.
//
//  2. `attn_bwd_params::D` is a plain `T2`, not a LazyTensor. LazyTensor
//     itself does not exist at this tag. So D is assigned directly.
//
//  3. There is NO `attn_bwd_params::DQ_ACC` field at all. The split-accumulator
//     backend it feeds does not exist yet, so there is nothing to allocate
//     and nothing to re-zero per iteration -- launch() below has no memset
//     for it, unlike every later adapter.
//
//  4. `attn_bwd_params::kVersion` is 1 here (it becomes 3 by 0.11b).
//     `attn_fwd_params::kVersion` is 1 at both. Both are read from the
//     struct rather than hardcoded, so this is only a note, not a
//     difference this file has to encode.
//
// Everything else -- field names, CausalType/WindowValue/VarlenType, the
// persistent atomic counter, the prepare() stream gap and the documented
// measurement simplifications -- matches head/adapter.cc, which carries the
// full commentary. Not restated here.
//
// NEVER RUN ON HARDWARE. Written against 0.10b's headers read in full; the
// only adapter validated on a GPU is head/adapter.cc (T14).

#include "../lib/common.h"

#include <perfmon/fill.h>

#include <aotriton/flash.h>
#include <aotriton/runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

using AOTRITON_NS::Stream;
using AOTRITON_NS::TensorView;
namespace flash = AOTRITON_NS::v3::flash;

using namespace perfmon_flash;

// 0.10b's attn_options cannot force a backend, so exactly one backend is
// observable per iface. See difference (1) in the header comment.
constexpr int kBackendCount = 1;

struct Context {
  int32_t iface = 0;
  int backend = 0;
  DeviceArena arena;

  void* atomic_counter_ptr = nullptr;

  flash::attn_fwd_params fwd_params;
  flash::attn_bwd_params bwd_params;
  flash::attn_options options;

  std::string describe_json;
};

// The describe() JSON deliberately does NOT claim a backend_index: no index
// was forced, so reporting one would misrepresent which backend ran. The
// explicit null is what tells the ingest side this row is dispatcher-choice
// rather than a specific backend that happens to be numbered 0.
const char* kDescribeJson = "{\"backend_index\": null, \"backend_forced\": false}";

void build_fwd(const pmon_entry* e, Context* ctx, hipStream_t s) {
  const auto dt = to_aotriton_dtype(e->dtype);
  const int32_t kv_heads = kv_heads_of(e);
  const uint64_t B = e->batch, HQ = e->n_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;
  auto* A = &ctx->arena;

  flash::attn_fwd_params& p = ctx->fwd_params;
  p.Q = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed, s, false);
  p.K = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed, s, false);
  p.V = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed, s, false);

  if (e->bias_type != 0) {
    p.B = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x1, s, false, flip);
  } else {
    p.B = TensorView<4>::get_null_tensor(dt);
  }

  p.Sm_scale = 1.0f / static_cast<float>(e->hdim);
  p.L = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed, s, /*zero=*/true);
  p.Out = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed, s, true);

  p.dropout_p = static_cast<float>(e->dropout_p);
  p.philox_seed_ptr = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset1 = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset2 = 0;
  p.philox_seed_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.encoded_softmax = TensorView<4>::get_null_tensor(dt);

  if (e->causal) {
    p.causal_type = flash::CausalType::WindowedAttention;
    p.window_left = flash::WindowValue::TopLeftAligned;
    p.window_right = flash::WindowValue::TopLeftAligned;
  } else {
    p.causal_type = flash::CausalType::None;
    p.window_left = 0;
    p.window_right = 0;
  }

  if (p.causal_type != flash::CausalType::None) {
    ctx->atomic_counter_ptr = A->alloc(sizeof(int32_t));
    perfmon::fill_zero(ctx->atomic_counter_ptr, sizeof(int32_t), s);
    p.persistent_atomic_counter =
        TensorView<0>(reinterpret_cast<intptr_t>(ctx->atomic_counter_ptr), AOTRITON_NS::kInt32);
  } else {
    p.persistent_atomic_counter = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);
  }

  p.varlen_type = flash::VarlenType::None;

  ctx->options = flash::attn_options();  // nothing to set: no fields
  ctx->describe_json = kDescribeJson;
}

void build_bwd(const pmon_entry* e, Context* ctx, hipStream_t s) {
  const auto dt = to_aotriton_dtype(e->dtype);
  const int32_t kv_heads = kv_heads_of(e);
  const uint64_t B = e->batch, HQ = e->n_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;
  auto* A = &ctx->arena;

  flash::attn_bwd_params& p = ctx->bwd_params;
  p.Q = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed, s, false);
  p.K = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x2, s, false);
  p.V = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x3, s, false);

  if (e->bias_type != 0) {
    p.B  = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x4, s, false, flip);
    p.DB = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x5, s, true, flip);
  } else {
    p.B = TensorView<4>::get_null_tensor(dt);
    p.DB = TensorView<4>::get_null_tensor(dt);
  }

  p.Sm_scale = 1.0f / static_cast<float>(e->hdim);
  p.Out = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0x6, s, false);
  p.DO  = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0x7, s, false);
  p.DK  = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x8, s, true);
  p.DV  = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x9, s, true);
  p.DQ  = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0xA, s, true);

  p.L = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xB, s, /*zero=*/false);
  // Plain T2 at this tag -- no LazyTensor wrapper. See difference (2).
  p.D = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xC, s, /*zero=*/false);

  // No DQ_ACC field exists at this tag -- see difference (3).

  p.dropout_p = static_cast<float>(e->dropout_p);
  p.philox_seed_ptr = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset1 = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset2 = 0;

  if (e->causal) {
    p.causal_type = flash::CausalType::WindowedAttention;
    p.window_left = flash::WindowValue::TopLeftAligned;
    p.window_right = flash::WindowValue::TopLeftAligned;
  } else {
    p.causal_type = flash::CausalType::None;
    p.window_left = 0;
    p.window_right = 0;
  }
  p.varlen_type = flash::VarlenType::None;

  ctx->options = flash::attn_options();
  ctx->describe_json = kDescribeJson;
}

}  // namespace

extern "C" int pmon_flash_enumerate_backends(const pmon_entry* entry, int* out, int max) {
  try {
    if (entry->iface != PMON_IFACE_ATTN_FWD && entry->iface != PMON_IFACE_ATTN_BWD) {
      return -1;
    }
    if (kBackendCount > max) return -1;
    out[0] = 0;
    return kBackendCount;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(0.10b) enumerate_backends failed: %s\n", ex.what());
    return -1;
  }
}

extern "C" int pmon_flash_prepare(const pmon_entry* entry, int backend, void** out_ctx) {
  auto* ctx = new Context();
  ctx->iface = entry->iface;
  ctx->backend = backend;
  hipStream_t scratch = nullptr;
  try {
    validate_entry(entry, backend, kBackendCount);
    hip_check(hipStreamCreate(&scratch), "hipStreamCreate (prepare scratch stream)");
    if (entry->iface == PMON_IFACE_ATTN_FWD) {
      build_fwd(entry, ctx, scratch);
    } else if (entry->iface == PMON_IFACE_ATTN_BWD) {
      build_bwd(entry, ctx, scratch);
    } else {
      throw std::invalid_argument("adapter(0.10b): unknown iface index: " +
                                  std::to_string(entry->iface));
    }
    hip_check(hipStreamSynchronize(scratch), "hipStreamSynchronize (prepare scratch stream)");
    hip_check(hipStreamDestroy(scratch), "hipStreamDestroy (prepare scratch stream)");
    *out_ctx = ctx;
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(0.10b) prepare failed: %s\n", ex.what());
    if (scratch) (void)hipStreamDestroy(scratch);
    ctx->arena.release_all();
    delete ctx;
    return -1;
  }
}

extern "C" int pmon_flash_launch(void* ctx_v, hipStream_t stream) {
  auto* ctx = static_cast<Context*>(ctx_v);
  hipError_t err = hipSuccess;
  if (ctx->iface == PMON_IFACE_ATTN_FWD) {
    if (ctx->atomic_counter_ptr) {
      perfmon::fill_zero(ctx->atomic_counter_ptr, sizeof(int32_t), stream);
    }
    err = flash::attn_fwd(ctx->fwd_params, flash::attn_fwd_params::kVersion, Stream(stream),
                          &ctx->options);
  } else if (ctx->iface == PMON_IFACE_ATTN_BWD) {
    // No dq_acc re-zeroing here: the field does not exist at this tag.
    err = flash::attn_bwd(ctx->bwd_params, flash::attn_bwd_params::kVersion, Stream(stream),
                          &ctx->options);
  } else {
    return -1;
  }
  return (err == hipSuccess) ? 0 : static_cast<int>(err);
}

extern "C" const char* pmon_flash_describe(void* ctx_v) {
  return static_cast<Context*>(ctx_v)->describe_json.c_str();
}

extern "C" void pmon_flash_release(void* ctx_v) {
  auto* ctx = static_cast<Context*>(ctx_v);
  ctx->arena.release_all();
  delete ctx;
}
