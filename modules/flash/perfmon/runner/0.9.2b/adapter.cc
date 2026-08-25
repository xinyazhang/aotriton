// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// flash adapter for 0.9.2b -- the only tag in the perfmon range that predates
// the v3 API entirely.
//
// --- What is different, and why this is a separate file ------------------
// Read from 0.9.2b's own include/aotriton/flash.h:
//
//  1. There is NO `AOTRITON_NS::v3` namespace and NO params struct of any
//     kind. The entry points are long positional free functions in
//     `AOTRITON_NS::v2::flash`:
//
//         attn_fwd(q, k, v, b, sm_scale, softmax_lse, Out, dropout_p,
//                  philox_seed, philox_offset1, philox_offset2,
//                  philox_seed_output, philox_offset_output,
//                  encoded_softmax, is_causal, atomic_for_causal,
//                  stream, extargs)
//
//         attn_bwd(q, k, v, b, sm_scale, out, dout, dq, dk, dv, db,
//                  softmax_lse, delta, dropout_p, philox_seed,
//                  philox_offset1, philox_offset2, is_causal, stream,
//                  extargs)
//
//     So there is nothing to "pack" in prepare() the way every later adapter
//     does -- the arguments are held as individual Context members and
//     spelled out at the call site in launch(). This is precisely the churn
//     rev0 §5 says to contain in a per-version file rather than branch on.
//
//  2. Causality is a plain `bool is_causal`, not CausalType/WindowValue.
//     Windowed attention does not exist at this tag, so a causal entry maps
//     to `is_causal = true` and nothing else. `CausalType::WindowedAttention`
//     with TopLeft-aligned windows -- what later adapters send for a causal
//     entry -- is the same computation for a full causal mask, so the rows
//     remain comparable; there is simply no window knob to set here.
//
//  3. `atomic_for_causal` is this tag's spelling of
//     `persistent_atomic_counter`, and it has the same requirement: a real,
//     zeroed int32 whenever causality is on, re-zeroed before every launch
//     (the persistent kernel does `atomic_add(1)` and a stale counter makes
//     later iterations exit with no tiles to claim -- silently reporting an
//     absurdly fast time rather than faulting).
//
//  4. `delta` is a plain `T2` buffer ("empty_like(softmax_lse)" per the
//     header's own comment). No LazyTensor, no DQ_ACC, no split accumulator.
//
//  5. Backend selection is via `FwdExtraArguments`/`BwdExtraArguments`,
//     which derive from `CppTune` and are compiled out entirely unless
//     AOTRITON_BUILD_FOR_TUNING is set -- which the shim build never sets.
//     There is therefore no way to force a backend at this tag, exactly as
//     at 0.10b: enumerate_backends() reports ONE backend, and describe()
//     records that it was the dispatcher's choice rather than a forced
//     index. Reporting 2/3 and passing an option this release ignores would
//     produce identical measurements labelled as different backends.
//
//  6. philox_offset2 is `int64_t` here (it becomes `uint64_t` on the v3
//     params struct). Immaterial at the 0 this adapter passes, noted so the
//     difference is not mistaken for an oversight.
//
// The measurement choices this file shares with the others -- deterministic
// fill, the padded-then-sliced bias layout, Sm_scale = 1/hdim, null philox
// tensors, standalone attn_bwd with generic L/delta -- are documented in
// head/adapter.cc and not restated here.
//
// --- Do not reuse this adapter for a later tag ----------------------------
// This file COMPILES CLEANLY against 0.10b's and 0.11.2b's headers too --
// verified, not assumed -- because those releases still ship the v2 API
// alongside v3. That makes it a trap: symlinking 0.10b/ or 0.11b/ to this
// directory would build and run, while silently measuring the v2 entry
// point on a release whose v3 path is the one users actually get.
//
// rev0 §4 is explicit that the API is a CHOICE, not just a fact about the
// release: "prefer the v3 API from 0.10b inclusive ... v3 is the API
// AOTriton is optimized around, it is what current PyTorch calls, and it
// avoids v2's shim overhead -- so measuring it is measuring what users
// actually get. Only 0.9.x predates v3 and must fall back." The compiler
// cannot enforce that preference; the tag -> directory mapping is what does.
//
// NEVER RUN ON HARDWARE. Written against 0.9.2b's headers read in full; the
// only adapter validated on a GPU is head/adapter.cc (T14). This is also the
// row rev0 §4 flags as most likely to drop out entirely if 0.9.2b cannot
// build against the unified ROCm.

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
namespace flash = AOTRITON_NS::v2::flash;

using namespace perfmon_flash;

// No backend forcing at this tag -- see difference (5).
constexpr int kBackendCount = 1;

const char* kDescribeJson = "{\"backend_index\": null, \"backend_forced\": false}";

// Every argument the v2 free functions take, held individually because there
// is no params struct to hold them -- see difference (1).
struct Context {
  int32_t iface = 0;
  int backend = 0;
  DeviceArena arena;

  TensorView<4> q, k, v, b;
  TensorView<4> out, dout, dq, dk, dv, db;
  TensorView<2> softmax_lse, delta;
  TensorView<4> encoded_softmax;

  // These five need explicit initializers, unlike their rank-2/rank-4
  // siblings above. At THIS tag `TensorView<0>` is a hand-written
  // specialization whose only constructor is `TensorView(intptr_t, DType)`
  // -- it has no default constructor (one arrives at 0.10b). Without an
  // initializer, Context's own implicit default constructor is deleted and
  // `new Context()` in prepare() fails to compile. get_null_tensor(dtype)
  // is the specialization's own null form, and is what these stay when the
  // corresponding feature is disabled.
  TensorView<0> philox_seed = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> philox_offset1 = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> philox_seed_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> philox_offset_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> atomic_for_causal = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);

  float sm_scale = 0.0f;
  float dropout_p = 0.0f;
  bool is_causal = false;

  void* atomic_counter_ptr = nullptr;

  std::string describe_json;
};

void build_common_inputs(const pmon_entry* e, Context* ctx, hipStream_t s) {
  const int32_t kv_heads = kv_heads_of(e);
  const uint64_t B = e->batch, HQ = e->n_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;
  const auto dt = to_aotriton_dtype(e->dtype);
  auto* A = &ctx->arena;

  ctx->q = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed, s, false);
  ctx->k = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x2, s, false);
  ctx->v = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x3, s, false);

  if (e->bias_type != 0) {
    ctx->b = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x4, s, false, flip);
  } else {
    ctx->b = TensorView<4>::get_null_tensor(dt);
  }

  ctx->sm_scale = 1.0f / static_cast<float>(e->hdim);
  ctx->dropout_p = static_cast<float>(e->dropout_p);
  ctx->is_causal = e->causal != 0;

  ctx->philox_seed = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  ctx->philox_offset1 = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  ctx->philox_seed_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  ctx->philox_offset_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  ctx->encoded_softmax = TensorView<4>::get_null_tensor(dt);
}

void build_fwd(const pmon_entry* e, Context* ctx, hipStream_t s) {
  const uint64_t B = e->batch, HQ = e->n_heads;
  const uint64_t SQ = e->seqlen_q, D = e->hdim;
  const bool flip = e->storage_flip != 0;
  auto* A = &ctx->arena;

  build_common_inputs(e, ctx, s);

  ctx->softmax_lse = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed, s, /*zero=*/true);
  ctx->out = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed, s,
                            /*zero=*/true);

  // atomic_for_causal: this tag's persistent_atomic_counter -- a real,
  // zeroed int32 when causal, null otherwise. See difference (3).
  if (ctx->is_causal) {
    ctx->atomic_counter_ptr = A->alloc(sizeof(int32_t));
    perfmon::fill_zero(ctx->atomic_counter_ptr, sizeof(int32_t), s);
    ctx->atomic_for_causal =
        TensorView<0>(reinterpret_cast<intptr_t>(ctx->atomic_counter_ptr), AOTRITON_NS::kInt32);
  } else {
    ctx->atomic_for_causal = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);
  }

  ctx->describe_json = kDescribeJson;
}

void build_bwd(const pmon_entry* e, Context* ctx, hipStream_t s) {
  const int32_t kv_heads = kv_heads_of(e);
  const uint64_t B = e->batch, HQ = e->n_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;
  const auto dt = to_aotriton_dtype(e->dtype);
  auto* A = &ctx->arena;

  build_common_inputs(e, ctx, s);

  ctx->out  = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0x6, s, false);
  ctx->dout = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0x7, s, false);
  ctx->dk   = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x8, s, true);
  ctx->dv   = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x9, s, true);
  ctx->dq   = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0xA, s, true);

  if (e->bias_type != 0) {
    ctx->db = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x5, s, /*zero=*/true, flip);
  } else {
    ctx->db = TensorView<4>::get_null_tensor(dt);
  }

  ctx->softmax_lse = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xB, s, /*zero=*/false);
  // "buffer, empty_like(softmax_lse)" per the header's own comment.
  ctx->delta = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xC, s, /*zero=*/false);

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
    std::fprintf(stderr, "adapter(0.9.2b) enumerate_backends failed: %s\n", ex.what());
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
      throw std::invalid_argument("adapter(0.9.2b): unknown iface index: " +
                                  std::to_string(entry->iface));
    }
    hip_check(hipStreamSynchronize(scratch), "hipStreamSynchronize (prepare scratch stream)");
    hip_check(hipStreamDestroy(scratch), "hipStreamDestroy (prepare scratch stream)");
    *out_ctx = ctx;
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(0.9.2b) prepare failed: %s\n", ex.what());
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
    err = flash::attn_fwd(ctx->q, ctx->k, ctx->v, ctx->b,
                          ctx->sm_scale,
                          ctx->softmax_lse,
                          ctx->out,
                          ctx->dropout_p,
                          ctx->philox_seed, ctx->philox_offset1, /*philox_offset2=*/0,
                          ctx->philox_seed_output, ctx->philox_offset_output,
                          ctx->encoded_softmax,
                          ctx->is_causal,
                          ctx->atomic_for_causal,
                          Stream(stream),
                          /*extargs=*/nullptr);
  } else if (ctx->iface == PMON_IFACE_ATTN_BWD) {
    err = flash::attn_bwd(ctx->q, ctx->k, ctx->v, ctx->b,
                          ctx->sm_scale,
                          ctx->out, ctx->dout,
                          ctx->dq, ctx->dk, ctx->dv, ctx->db,
                          ctx->softmax_lse, ctx->delta,
                          ctx->dropout_p,
                          ctx->philox_seed, ctx->philox_offset1, /*philox_offset2=*/0,
                          ctx->is_causal,
                          Stream(stream),
                          /*extargs=*/nullptr);
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
