// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// flash adapter for the 0.11 line (0.11b and 0.11.2b -- the latter a
// directory symlink to this one; both tags' include/aotriton/{flash,util}.h
// are identical in every respect this file touches).
//
// --- The one difference from head/adapter.cc ------------------------------
// LazyTensor at this generation has NO `eager` member:
//
//     template<int Rank> struct LazyTensor {
//       void* cookie = nullptr;
//       TensorView<Rank> (*acquire)(void* cookie) = nullptr;
//       void  (*dispose)(void* cookie) = nullptr;
//       operator bool() const {
//         return cookie != nullptr || acquire != nullptr || dispose != nullptr;
//       }
//       void free() { if (dispose && cookie) { (*dispose)(cookie); cookie = nullptr; } }
//     };
//
// `eager` (and the `acquire(LazyTensor<Rank>* self)` signature that goes
// with it) arrive at 0.12b. So attn_bwd_params::D and ::DQ_ACC are set here
// by pointing `cookie` at a TensorView owned by the Context and giving
// `acquire` a trampoline that returns it -- which is the same thing `eager`
// does at the later generation, just spelled through the callback the API
// offers here.
//
// `dispose` is deliberately left null. The buffer behind the cookie is owned
// by the Context's DeviceArena and freed in release(); a dispose that also
// freed it would double-free, and LazyTensor::free()'s own header comment
// ("FIXME: This design is prone to memory leaks") is a caller-side ownership
// warning, not an obligation on this adapter. Leaving dispose null means
// free() is a no-op, and the arena remains the single owner.
//
// The cookie must outlive every launch() call, so the TensorViews it points
// at are Context members, NOT locals of build_bwd() -- taking the address of
// a local here would leave AOTriton dereferencing a dangling pointer on the
// first captured iteration.
//
// Everything else -- params field names, kVersion values (fwd 1, bwd 3),
// attn_options::force_backend_index, CausalType/WindowValue/VarlenType --
// matches head/adapter.cc, which carries the full commentary on the
// measurement choices this file repeats (dq_acc re-zeroing, the persistent
// atomic counter, the prepare() stream gap, and the documented
// simplifications). Not restated here; see that file.
//
// --- KNOWN GAP, DEFERRED: the AITER ASM backend at this tag ---------------
// The backend counts this file reports come from backend_count_forceable(),
// which encodes modules/flash/tune/level_op.py's HEAD-era table (attn_fwd
// 2/1, attn_bwd 3/2). That table is very likely WRONG at this tag, in both
// directions, and neither has been checked on hardware:
//
//   attn_bwd backend 2 (kSlimAffine_AiterFmhaV3Bwd). At 0.11b/0.11.2b --
//     and ONLY at those two tags -- `aiter_bwd` is a SEPARATE exported
//     entry point in include/aotriton/flash.h, alongside `attn_bwd`. It is
//     absent at 0.10b (no AITER at all) and absent again from 0.12.1b
//     onward, where it was folded into attn_bwd's own backend dispatch.
//     That strongly suggests AITER is not in this tag's OpAttnBwdContext
//     backend enum, i.e. driving it through `attn_bwd` with
//     force_backend_index = 2 -- which is what launch() below does -- does
//     not reach it. Supporting it here means calling
//     `flash::aiter_bwd(params, kVersion, stream, &options)` for index 2
//     instead. Note aiter_bwd.cc at this tag opens with
//     `if (!in.DQ_ACC) return hipErrorInvalidMemcpyDirection;`, so the
//     cookie/acquire pair set up in build_bwd() is a prerequisite either
//     way.
//
//   attn_fwd backend 1 (kSlimAffine_AiterFmhaV3Fwd). This tag's flash.h
//     carries `aiter_fwd` COMMENTED OUT under "NOTE: DEFERRED TO NEXT
//     RELEASE", so the forward AITER backend does not exist here at all.
//     Reporting 2 forward backends on gfx942/gfx950 is therefore probably
//     one too many.
//
// Until both are resolved, treat this tag's backend axis as unverified:
// index 2 of attn_bwd may error out or silently measure a Triton backend,
// and index 1 of attn_fwd likewise. The fix is small (a per-tag backend
// count plus an aiter_bwd call for index 2) but it needs the tag's real
// backend enum to confirm, which means checking on hardware -- deliberately
// deferred rather than guessed at.
//
// NEVER RUN ON HARDWARE. Written against 0.11.2b's headers read in full;
// the only adapter validated on a GPU is head/adapter.cc (T14).

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

struct Context {
  int32_t iface = 0;
  int backend = 0;
  DeviceArena arena;

  void* dq_acc_ptr = nullptr;
  size_t dq_acc_bytes = 0;
  void* atomic_counter_ptr = nullptr;

  // Backing store for the two LazyTensor cookies below. Context members, not
  // locals -- see the header comment.
  TensorView<2> delta_view;
  TensorView<4> dq_acc_view;

  flash::attn_fwd_params fwd_params;
  flash::attn_bwd_params bwd_params;
  flash::attn_options options;

  std::string describe_json;
};

// LazyTensor trampolines: hand back the TensorView the cookie points at.
// The signature here takes `void* cookie` -- at 0.12b onward it becomes
// `LazyTensor<Rank>* self`, which is why this file exists.
TensorView<2> acquire_view2(void* cookie) {
  return *static_cast<TensorView<2>*>(cookie);
}
TensorView<4> acquire_view4(void* cookie) {
  return *static_cast<TensorView<4>*>(cookie);
}

void build_fwd(const pmon_entry* e, int backend, Context* ctx, hipStream_t s) {
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

  ctx->options = flash::attn_options();
  ctx->options.force_backend_index = backend;
  ctx->describe_json = "{\"backend_index\": " + std::to_string(backend) + "}";
}

void build_bwd(const pmon_entry* e, int backend, Context* ctx, hipStream_t s) {
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

  // D: the LazyTensor equivalent of head/adapter.cc's `p.D.eager = view`.
  ctx->delta_view = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xC, s, /*zero=*/false);
  p.D.cookie = &ctx->delta_view;
  p.D.acquire = &acquire_view2;
  p.D.dispose = nullptr;  // the arena owns the buffer -- see header comment

  // dq_acc: only backend 2 accumulates into it; backends 0/1 get a null-base
  // view sized like DQ. Always fp32, shaped like Q, never storage_flip'd.
  {
    Shape4 shp = bhsd_shape(B, HQ, SQ, D, /*storage_flip=*/false);
    if (backend == 2) {
      const int64_t count = static_cast<int64_t>(B) * HQ * SQ * D;
      ctx->dq_acc_bytes = static_cast<size_t>(count) * sizeof(float);
      ctx->dq_acc_ptr = A->alloc(ctx->dq_acc_bytes);
      perfmon::fill_zero(ctx->dq_acc_ptr, ctx->dq_acc_bytes, s);
      ctx->dq_acc_view = TensorView<4>(reinterpret_cast<intptr_t>(ctx->dq_acc_ptr), shp.sizes,
                                       shp.strides, AOTRITON_NS::kFloat32);
    } else {
      ctx->dq_acc_view = TensorView<4>(0, shp.sizes, shp.strides, AOTRITON_NS::kFloat32);
    }
    p.DQ_ACC.cookie = &ctx->dq_acc_view;
    p.DQ_ACC.acquire = &acquire_view4;
    p.DQ_ACC.dispose = nullptr;
  }

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
  ctx->options.force_backend_index = backend;
  ctx->describe_json = "{\"backend_index\": " + std::to_string(backend) + "}";
}

}  // namespace

extern "C" int pmon_flash_enumerate_backends(const pmon_entry* entry, int* out, int max) {
  try {
    const int count = backend_count_forceable(entry->iface, current_gpu_is_942_or_950());
    if (count == 0) return -1;
    if (count > max) return -1;
    for (int i = 0; i < count; ++i) out[i] = i;
    return count;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(0.11) enumerate_backends failed: %s\n", ex.what());
    return -1;
  }
}

extern "C" int pmon_flash_prepare(const pmon_entry* entry, int backend, void** out_ctx) {
  auto* ctx = new Context();
  ctx->iface = entry->iface;
  ctx->backend = backend;
  hipStream_t scratch = nullptr;
  try {
    validate_entry(entry, backend,
                   backend_count_forceable(entry->iface, current_gpu_is_942_or_950()));
    hip_check(hipStreamCreate(&scratch), "hipStreamCreate (prepare scratch stream)");
    if (entry->iface == PMON_IFACE_ATTN_FWD) {
      build_fwd(entry, backend, ctx, scratch);
    } else if (entry->iface == PMON_IFACE_ATTN_BWD) {
      build_bwd(entry, backend, ctx, scratch);
    } else {
      throw std::invalid_argument("adapter(0.11): unknown iface index: " +
                                  std::to_string(entry->iface));
    }
    hip_check(hipStreamSynchronize(scratch), "hipStreamSynchronize (prepare scratch stream)");
    hip_check(hipStreamDestroy(scratch), "hipStreamDestroy (prepare scratch stream)");
    *out_ctx = ctx;
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(0.11) prepare failed: %s\n", ex.what());
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
    if (ctx->backend == 2) {
      perfmon::fill_zero(ctx->dq_acc_ptr, ctx->dq_acc_bytes, stream);
    }
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
