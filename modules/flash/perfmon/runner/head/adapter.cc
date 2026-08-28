// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// flash adapter for the CURRENT v3 API: LazyTensor carries an `eager`
// TensorView and its acquire/dispose take `LazyTensor<Rank>* self`.
//
// Serves HEAD, 0.13b and 0.12.1b -- the latter two are directory symlinks to
// this one, verified identical by diffing each tag's own
// include/aotriton/{flash,util}.h. When HEAD's API next drifts, replace those
// symlinks with real directories rather than branching inside this file
// (rev0 §5: "never a branch in a file other subjects also compile").
//
// Distinguishing features of this generation, relative to its predecessors:
//   - attn_bwd_params::D is LazyTensor<2> and DQ_ACC is LazyTensor<4>, both
//     settable through `.eager` -- no cookie/acquire callback needed.
//   - attn_options::force_backend_index exists, so all 2/3 backends are
//     individually measurable.
//   - attn_bwd_params::kVersion is 3.
//
// This is the only adapter that has ever run on hardware: T14 passed on
// gfx942 / ROCm 7.14 against a real libaotritonpmon_v2.so, with attn_fwd
// backends 0-1 and attn_bwd backends 0-2 all measuring, and
// enumerate_backends returning the counts modules/flash/tune/level_op.py
// specifies. One real bug was found by that run and fixed: build_fwd()
// passed a NULL persistent_atomic_counter unconditionally, which faults the
// GPU at address 0x0 for any causal entry -- see the comment at its
// assignment.
//
// Paths NOT yet exercised on hardware: bias_type != 0, dropout_p > 0, GQA
// (gqa_ratio > 1), storage_flip, and fp16 (every T14 shape was bf16,
// non-GQA, unbiased, dropout-free).
//
// ---------------------------------------------------------------------------
// AOTRITON_NS, not `aotriton::`, throughout
// ---------------------------------------------------------------------------
// build_subject.sh builds every subject with `AOTRITON_NAME_SUFFIX=pmon`
// (rev0 §4's own worked example), which makes AOTRITON_NS expand to
// `aotritonpmon`. A hardcoded `aotriton::v3::flash::...` would fail to
// compile against every subject this project actually builds.
// <aotriton/flash.h> pulls in the per-subject <aotriton/config.h> that
// defines the macro correctly either way.
//
// ---------------------------------------------------------------------------
// dq_acc zeroing
// ---------------------------------------------------------------------------
// modules/flash/tune/level_op.py's backend-2 `direct_call` calls
// `zero_devm(devm.dq_acc)` immediately before EVERY attn_bwd call, because
// backend 2 (kSlimAffine_AiterFmhaV3Bwd) *accumulates* into dq_acc rather
// than overwriting it. timing.cc's capture calls launch() once per captured
// iteration, so launch() below issues the zeroing memset every time it runs
// -- stream-ordered, therefore one memset node per captured iteration.
//
// ---------------------------------------------------------------------------
// The prepare() ABI gap
// ---------------------------------------------------------------------------
// Device fill needs a hipStream_t, but `pmon_family_vtable::prepare` takes
// only (entry, backend, ctx). Resolved by creating a short-lived private
// stream inside prepare(), enqueuing every fill on it, and synchronizing
// before returning -- so every buffer is fully populated before prepare()
// hands control back, regardless of which stream a later launch() uses.
// Local to the adapter; core's timing.cc never calls fill itself.
//
// ---------------------------------------------------------------------------
// Documented simplifications (perfmon does zero accuracy checking, D4 --
// these values only need to be finite and representative):
//   * hdim_v == hdim; no entry generator varies hdim_v today.
//   * Sm_scale hardcoded to 1/hdim, FlashInputMetadata's 'l1' default.
//   * philox seed/offset tensors null, matching reference.py's
//     `philox_null` pattern. `entry->seed` drives perfmon's own device-side
//     fill only (rev0 §5.3), never AOTriton's internal dropout RNG.
//   * encoded_softmax left null (a null-base view is the documented
//     "disabled" signal).
//   * Backward's L and D are, in a real pipeline, produced by a preceding
//     forward/bwd_preprocess pass. This adapter measures attn_bwd in
//     isolation (D3: op-level fanout, no end-to-end chaining), so both are
//     filled with the generic filler rather than mathematically consistent
//     logsumexp/delta values.

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

  // Only set for attn_bwd backend 2 (AiterFmhaV3Bwd): re-zeroed by launch()
  // on every call, not just once here.
  void* dq_acc_ptr = nullptr;
  size_t dq_acc_bytes = 0;

  // Only set for attn_fwd with a non-None causal_type, which autoselects the
  // DYNAMIC persistent kernel. Like dq_acc, re-zeroed by launch() every call.
  void* atomic_counter_ptr = nullptr;

  flash::attn_fwd_params fwd_params;
  flash::attn_bwd_params bwd_params;
  flash::attn_options options;

  std::string describe_json;
};

void build_fwd(const pmon_entry* e, int backend, Context* ctx, hipStream_t s) {
  const auto dt = to_aotriton_dtype(e->dtype);
  const int32_t kv_heads = kv_heads_of(e);
  const uint64_t B = e->batch, HQ = e->n_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;
  auto* A = &ctx->arena;

  flash::attn_fwd_params& p = ctx->fwd_params;
  // p.A is left default-constructed: modules/flash/tune/calls.py's
  // `attn_fwd.direct_call` -- the authoritative in-repo caller for
  // kVersion 3 -- never sets it either.
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
  p.Out = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed, s,
                         /*zero=*/true);

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

  // persistent_atomic_counter MUST be a real, zeroed int32 whenever
  // causal_type != None. modules/flash/kernel/attn_torch_function.py
  // autoselects PersistentType.DYNAMIC in exactly that case, and the
  // persistent kernel's first act is
  // `tile_id = persistent_atomic_counter.atomic_add(1)`. Passing the null
  // tensor here faults the GPU at address 0x0 -- observed on gfx942 as
  // "Memory Fault Error ... faulting addr: 0x0, kernel: attn_fwd" the first
  // time this adapter was ever run (T14).
  //
  // Conversely it must stay NULL when causal_type == None: modules/flash/
  // aot/attn_fwd.py declares
  // `@ati.derives('persistent_atomic_counter', to=0, when=ati.eq('CAUSAL_TYPE', 0))`,
  // i.e. the AOT signature derives this argument to 0 in that case.
  if (p.causal_type != flash::CausalType::None) {
    ctx->atomic_counter_ptr = A->alloc(sizeof(int32_t));
    perfmon::fill_zero(ctx->atomic_counter_ptr, sizeof(int32_t), s);
    p.persistent_atomic_counter =
        TensorView<0>(reinterpret_cast<intptr_t>(ctx->atomic_counter_ptr), AOTRITON_NS::kInt32);
  } else {
    p.persistent_atomic_counter = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);
  }

  p.varlen_type = flash::VarlenType::None;
  // cu_seqlens_q/k and seq_strides_q/k stay at their default-constructed
  // value; `varlen_type == None` is what tells the kernel not to dereference
  // them, matching every other "disabled" tensor here.

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
    p.B = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x4, s, false, flip);
    // DB: modules/flash/tune/calls.py's `create_aotensor_like(inputs.b, ...)`
    // -> `torch.empty_like(b)`, whose default preserve_format keeps a
    // non-contiguous input's strides -- so DB gets the SAME padded-then-
    // sliced layout as `b` itself, not a fresh contiguous buffer.
    // Reproduced for the same reason B's padding is: a difference here could
    // change vectorized store patterns, i.e. the thing perfmon measures.
    p.DB = make_bias(A, B, HQ, SQ, SK, e->dtype, e->hdim, e->seed ^ 0x5, s, /*zero=*/true, flip);
  } else {
    p.B = TensorView<4>::get_null_tensor(dt);
    p.DB = TensorView<4>::get_null_tensor(dt);
  }

  p.Sm_scale = 1.0f / static_cast<float>(e->hdim);

  // Out/DO: in a real pipeline Out is attn_fwd's output and DO is dOut from
  // the loss; measured standalone here, so both are generic finite inputs.
  p.Out = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0x6, s, false);
  p.DO  = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0x7, s, false);
  p.DK  = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x8, s, true);
  p.DV  = make_and_fill4(A, bhsd_shape(B, HK, SK, D, flip), e->dtype, e->hdim, e->seed ^ 0x9, s, true);
  p.DQ  = make_and_fill4(A, bhsd_shape(B, HQ, SQ, D, flip), e->dtype, e->hdim, e->seed ^ 0xA, s, true);

  p.L = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xB, s, /*zero=*/false);
  // eager-wrapped real buffer, not lazily computed -- matches
  // modules/flash/tune/calls.py's `eager_delta()` pattern, where LazyTensor
  // is just the typing mechanism, not deferred computation.
  p.D.eager = make_lse_like(A, B, HQ, SQ, e->hdim, e->seed ^ 0xC, s, /*zero=*/false);

  // dq_acc: only backend 2 (AiterFmhaV3Bwd) reads/accumulates it; backends
  // 0/1 get a null-base eager view sized like DQ, matching
  // modules/flash/tune/calls.py's `eager_null_dq_acc` ("data_ptr=0: Triton
  // kernel will not access dq_acc, so the null pointer is safe").
  //
  // DQ_ACC is always a fresh fp32 accumulator shaped like Q and never
  // storage_flip'd -- it is not one of Q/K/V's storage buffers.
  {
    Shape4 shp = bhsd_shape(B, HQ, SQ, D, /*storage_flip=*/false);
    if (backend == 2) {
      const int64_t count = static_cast<int64_t>(B) * HQ * SQ * D;
      ctx->dq_acc_bytes = static_cast<size_t>(count) * sizeof(float);
      ctx->dq_acc_ptr = A->alloc(ctx->dq_acc_bytes);
      perfmon::fill_zero(ctx->dq_acc_ptr, ctx->dq_acc_bytes, s);
      p.DQ_ACC.eager = TensorView<4>(reinterpret_cast<intptr_t>(ctx->dq_acc_ptr), shp.sizes,
                                     shp.strides, AOTRITON_NS::kFloat32);
    } else {
      p.DQ_ACC.eager = TensorView<4>(0, shp.sizes, shp.strides, AOTRITON_NS::kFloat32);
    }
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
    if (count == 0) return -1;  // unknown iface
    // Fail loudly rather than silently truncating: a caller sizing its
    // buffer to `max` and getting back fewer entries than actually exist
    // would otherwise never learn its buffer was too small.
    if (count > max) return -1;
    for (int i = 0; i < count; ++i) out[i] = i;
    return count;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(head) enumerate_backends failed: %s\n", ex.what());
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
      throw std::invalid_argument("adapter(head): unknown iface index: " +
                                  std::to_string(entry->iface));
    }
    hip_check(hipStreamSynchronize(scratch), "hipStreamSynchronize (prepare scratch stream)");
    hip_check(hipStreamDestroy(scratch), "hipStreamDestroy (prepare scratch stream)");
    *out_ctx = ctx;
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter(head) prepare failed: %s\n", ex.what());
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
      // Re-zero the persistent tile counter every call: the kernel
      // atomically increments it, so a counter left at its post-launch value
      // makes every SUBSEQUENT iteration exit immediately with no tiles to
      // claim -- which would not fault, it would silently report an absurdly
      // fast time. Stream-ordered, so this is captured as one memset node
      // per iteration.
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
