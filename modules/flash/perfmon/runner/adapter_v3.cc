// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T12: libperfmon_flash@<subject>'s v3-API adapter (rev0
// D4, §4). Implements `pmon_family_entry()` for flash's two ifaces
// (attn_fwd, attn_bwd), against THIS repo's own `include/aotriton/flash.h`
// -- not a from-scratch reimplementation of the PyTorch integration this
// task's spec points at for the call sequence.
//
// UNVERIFIED IN THIS ENVIRONMENT: no ROCm/hipcc/HIP/AOTriton build exists
// here (see T05's commit message for how that was confirmed). This file has
// never been compiled, let alone linked against a real libaotriton*_v2.so.
// See adapter_v3/CMakeLists.txt's own header for the same disclosure on the
// build side, and perfmon-handoff0.md for the exact commands a human must
// run on a ROCm+GPU machine to actually verify it (T12's own Verify step is
// "T14" -- explicitly out of scope for this agent, per this task's briefing).
//
// ---------------------------------------------------------------------------
// AOTRITON_NS, not `aotriton::`, throughout this file
// ---------------------------------------------------------------------------
// T13's build_subject.sh builds every subject with `AOTRITON_NAME_SUFFIX=pmon`
// (perfmon-rev0.md §4's own worked example), which -- per
// `CMakeLists.txt:171` / `include/aotriton/config.h.in` -- makes
// AOTRITON_NS expand to `aotritonpmon`, not `aotriton`. A hardcoded
// `aotriton::v3::flash::...` would therefore fail to compile against every
// subject this project actually builds. `<aotriton/flash.h>` pulls in
// `<aotriton/config.h>` (generated per-subject at that subject's own configure
// time) which defines the macro correctly either way, so this file uses
// `AOTRITON_NS::` everywhere instead of the literal namespace name.
//
// ---------------------------------------------------------------------------
// Resolving the two PyTorch-reference-vs-spec-text questions from T12's
// research phase (recorded here, not silently picked one way):
// ---------------------------------------------------------------------------
// (1) Stream wrapping. T12's spec text says `launch` calls the entry point
//     "on the supplied stream -- aotriton::Stream(hipStream_t)"; the PyTorch
//     reference this task's spec points at (mha_all_aot.hip, release/2.12)
//     appears, per a fetched summary of that file, to hand a raw hipStream_t
//     to the launcher with no visible `aotriton::Stream` construction. These
//     are NOT actually in conflict: `include/aotriton/runtime.h:20`'s
//     `StreamTemplate(DeviceStreamType stream)` constructor is not
//     `explicit`, so a call site that writes `attn_fwd(params, kVersion,
//     stream, ...)` with a raw `hipStream_t stream` compiles via an implicit
//     conversion to `AOTRITON_NS::Stream` -- the PyTorch reference's
//     "raw stream" is just that implicit conversion happening at the call
//     site rather than being spelled out. This file spells it out explicitly
//     (`AOTRITON_NS::Stream(stream)`) for clarity, which is equivalent, not
//     a deviation.
// (2) dq_acc zeroing. T12's spec text says to zero dq_acc "inside the
//     captured loop (a memset node)"; the PyTorch reference summary
//     describes a `LazyTensorContext`/lazy-zero-materialization helper
//     instead of a visible memset call. This repo's OWN authoritative,
//     non-PyTorch tuning code -- `modules/flash/tune/level_op.py:104-124`
//     (backend 2's `direct_call`) -- resolves this cleanly: it calls
//     `zero_devm(devm.dq_acc)` immediately before EVERY `attn_bwd` call,
//     because backend 2 (kSlimAffine_AiterFmhaV3Bwd) *accumulates* into
//     dq_acc rather than overwriting it, so it must be re-zeroed on every
//     invocation, not once at allocation time. That is exactly "zero it
//     inside the captured loop": `timing.cc`'s hipgraph_ev100 capture calls
//     `vtable.launch()` once per captured iteration (kTimingIters times), so
//     this file's `launch()` issues the zeroing memset (`perfmon::fill_zero`
//     on `stream`, stream-ordered and therefore capturable) every time it
//     runs, which becomes one memset node per captured iteration -- the
//     PyTorch reference's lazy-tensor helper is a torch-integration detail
//     this file has no need to reproduce; level_op.py's plain zero-before-
//     each-call semantics is the actual, in-repo authority T12's spec text
//     was describing.
//
// ---------------------------------------------------------------------------
// Documented simplifications relative to modules/flash/tune/{calls,
// reference}.py (perfmon does zero accuracy checking, D4 -- these only need
// to be *finite* and *representative*, never numerically correct):
// ---------------------------------------------------------------------------
//  * hdim_v == hdim. `pmon_entry` (perfmon_abi.h) carries one `hdim`, unlike
//    `FlashInputMetadata.hdim`'s `int | tuple[int, int]` (hdim_qk != hdim_v)
//    escape hatch. No entry generator in modules/flash/perfmon/entry.py
//    varies hdim_v today, so this is not a loss of coverage, only a note for
//    if that ever changes.
//  * GQA: `kv_heads = entry->n_heads / entry->gqa_ratio` (perfmon_abi.h's
//    own doc comment for `gqa_ratio`). Assumes exact divisibility, matching
//    every (N_HEADS, gqa) pair modules/flash/perfmon/entry.py actually
//    generates ((5,1) and (10,5)->(10,2) heads). `prepare()` fails closed
//    (returns nonzero) rather than truncating if it doesn't divide evenly.
//  * `varlen`: modules/flash/perfmon/entry.py's own docstring records that
//    no varlen axis exists anywhere in the entry space yet ("KNOWN GAPS").
//    This adapter enforces that same limitation defensively: `entry->varlen
//    != 0` fails `prepare()` rather than silently ignoring the flag.
//  * `Sm_scale`: FlashInputMetadata's entry-space generators never override
//    `sm_scale` (always its 'l1' default, `1/hdim`) -- there is no
//    `sm_scale` field on `pmon_entry` to carry a different value even if
//    they did. Hardcoded to `1.0f / hdim` here, matching that default.
//  * philox seed/offset tensors: null (`T0::get_null_tensor`), offset2 = 0,
//    matching `modules/flash/tune/reference.py`'s own `philox_null` pattern
//    (used unconditionally there, independent of `dropout_p`) rather than
//    threading `entry->seed` through as a literal philox seed -- `seed` is
//    used only for perfmon's own device-side fill (rev0 §5.3), a completely
//    separate concern from AOTriton's internal dropout RNG.
//  * `persistent_atomic_counter` (attn_fwd) and `encoded_softmax`: left as
//    null tensors (`TensorView::operator bool()` reports false for a
//    null-base view, which is the documented "disabled" signal for both).
//  * Backward's `L` (logsumexp) and `D` (delta) are, in a real pipeline,
//    produced by a preceding forward/bwd_preprocess pass. Since this
//    adapter measures attn_bwd in isolation (T12/D3: op-level fanout, no
//    end-to-end chaining), both are filled with `perfmon::fill_uniform_scaled`
//    (the same generic filler used for Q/K/V) rather than mathematically
//    consistent logsumexp/delta values -- finite and non-degenerate is all
//    rev0 §5.3 requires; exact values are irrelevant to a family that skips
//    correctness checking entirely (D4).
//  * Bias padding: `modules/flash/tune/reference.py` allocates bias at
//    `round_to_8x(seqlen_k)` physical width and exposes a `[..., :seqlen_k]`
//    logical slice, presumably so the kernel's vectorized loads see the
//    stride pattern real callers produce. Reproduced here (not simplified to
//    an unpadded contiguous width) specifically because deviating could
//    change the very thing perfmon measures -- memory access performance --
//    not just numerical correctness.
//
// ---------------------------------------------------------------------------
// The `prepare()` ABI gap (a stream is not passed to prepare(), only to
// launch()): device fill (perfmon::fill_uniform_scaled/fill_zero) needs a
// hipStream_t to enqueue on, but `perfmon_family_vtable::prepare` takes only
// `(entry, backend, ctx)` -- perfmon_abi.h, unedited by this file. Resolved
// by creating a short-lived, private stream inside `prepare()`, enqueuing
// every fill on it, and synchronizing before returning -- so every buffer
// is fully populated, on the host's terms, before `prepare()` hands control
// back, regardless of which stream a later `launch()` call uses. This is a
// disclosed interpretation of an underspecified corner of the ABI, not a
// silent one; it does not require an ABI change because it is entirely
// local to this adapter (core's timing.cc never calls fill itself).

#include <perfmon/perfmon_abi.h>
#include <perfmon/fill.h>

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AOTRITON_NS::DType;
using AOTRITON_NS::LazyTensor;
using AOTRITON_NS::Stream;
using AOTRITON_NS::TensorView;
namespace flash = AOTRITON_NS::v3::flash;

// flash's own iface ordering -- deliberately NOT in perfmon_abi.h (see the
// note above `pmon_entry.iface`'s declaration: iface names are a family
// concern, D4). Must match `modules/flash/perfmon/__init__.py`'s
// `PerfDesc.IFACES = ('attn_fwd', 'attn_bwd')` / `list_ifaces()` exactly,
// since that Python ordering is what assigns `pmon_entry.iface`'s value in
// the first place (rev0 D8). Found missing (an undefined-identifier bug)
// during this file's own final self-review, before any compile was
// attempted -- never shipped in a prior commit.
constexpr int32_t PMON_IFACE_ATTN_FWD = 0;
constexpr int32_t PMON_IFACE_ATTN_BWD = 1;

// --- small local helpers, none of which is AOTriton- or HIP-version-------
// specific enough to belong in perfmon/core (rev0 D4) --------------------

int64_t round_to_8x(int64_t n) {
  return 8 * ((n + 7) / 8);
}

DType to_aotriton_dtype(int32_t pmon_dtype) {
  switch (pmon_dtype) {
    case PMON_DTYPE_FLOAT16:  return AOTRITON_NS::kFloat16;
    case PMON_DTYPE_BFLOAT16: return AOTRITON_NS::kBFloat16;
    case PMON_DTYPE_FLOAT32:  return AOTRITON_NS::kFloat32;
    default:
      throw std::invalid_argument("adapter_v3: unknown pmon_dtype: " +
                                   std::to_string(pmon_dtype));
  }
}

size_t dtype_bytes(int32_t pmon_dtype) {
  switch (pmon_dtype) {
    case PMON_DTYPE_FLOAT16:  return 2;
    case PMON_DTYPE_BFLOAT16: return 2;
    case PMON_DTYPE_FLOAT32:  return 4;
    default:
      throw std::invalid_argument("adapter_v3: unknown pmon_dtype: " +
                                   std::to_string(pmon_dtype));
  }
}

void hip_check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("adapter_v3: ") + what + " failed: " +
                              hipGetErrorString(err));
  }
}

// True on gfx942/gfx950, matching modules/flash/tune/level_op.py's
// `_gpu_arch()` (`torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]`)
// -- there is no arch string on `pmon_entry` (perfmon_abi.h), and the whole
// runner process is pinned to one GPU (rev0 D4), so device 0 (whichever
// physical GPU HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES resolves it to,
// exactly as torch.cuda would) is queried directly here instead.
bool current_gpu_is_942_or_950() {
  int dev = 0;
  hip_check(hipGetDevice(&dev), "hipGetDevice");
  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, dev), "hipGetDeviceProperties");
  std::string arch = prop.gcnArchName;
  size_t colon = arch.find(':');
  if (colon != std::string::npos) arch = arch.substr(0, colon);
  return arch == "gfx942" || arch == "gfx950";
}

// enumerate_backends' backend counts -- kept in exactly one place, per
// T12's spec ("Read that file; keep the counts in one place"), mirroring
// modules/flash/tune/level_op.py:88-102's `attn_fwd.BACKEND_COUNT` /
// `attn_bwd.BACKEND_COUNT` properties (2/1 and 3/2 respectively).
int backend_count(int32_t iface, bool is_942_950) {
  switch (iface) {
    case PMON_IFACE_ATTN_FWD: return is_942_950 ? 2 : 1;
    case PMON_IFACE_ATTN_BWD: return is_942_950 ? 3 : 2;
    default: return 0;
  }
}

struct Shape4 {
  std::array<uint64_t, 4> sizes;
  std::array<uint64_t, 4> strides;
};

// Logical shape (B, H, S, D) with strides for the (default) contiguous
// layout, or -- when `storage_flip` -- the layout
// `modules/flash/tune/reference.py:108-136` produces: allocate physically
// as (B, S, H, D) contiguous, then `torch.transpose(1, 2)` back to logical
// (B, H, S, D). Sizes are unaffected either way (a transpose view keeps the
// same *logical* shape as seen by size(i)/stride(i)); only strides differ.
Shape4 bhsd_shape(uint64_t b, uint64_t h, uint64_t s, uint64_t d, bool storage_flip) {
  Shape4 out;
  out.sizes = {b, h, s, d};
  if (!storage_flip) {
    out.strides = {h * s * d, s * d, d, 1};
  } else {
    out.strides = {h * s * d, d, h * d, 1};
  }
  return out;
}

// One measurement context: every device buffer this iface/backend needs,
// kept alive across repeated `launch()` calls (prepare() runs once;
// launch() runs kTimingIters times per timing.cc, T10), plus the AOTriton
// params/options structs launch() reuses unchanged on every call.
struct Context {
  int32_t iface = 0;
  int backend = 0;

  std::vector<void*> owned_ptrs;  // freed in release(), in reverse order

  // Only set for attn_bwd backend 2 (AiterFmhaV3Bwd): re-zeroed by launch()
  // on every call (see the dq_acc discussion above), not just once here.
  void* dq_acc_ptr = nullptr;
  size_t dq_acc_bytes = 0;

  flash::attn_fwd_params fwd_params;
  flash::attn_bwd_params bwd_params;
  flash::attn_options options;

  std::string describe_json;
};

void* alloc_device(Context* ctx, size_t bytes) {
  void* p = nullptr;
  hip_check(hipMalloc(&p, bytes), "hipMalloc");
  ctx->owned_ptrs.push_back(p);
  return p;
}

// Fills `count` elements of `dtype` at `dst` with perfmon's deterministic
// RNG (rev0 §5.3), scaled for `hdim`, on `stream`. `zero` selects
// `perfmon::fill_zero` instead (true outputs -- rev0 §5.3: "dq_acc and
// outputs are zeroed").
void fill(void* dst, int64_t count, int32_t dtype, int32_t hdim, uint64_t seed,
          hipStream_t stream, bool zero) {
  if (zero) {
    perfmon::fill_zero(dst, static_cast<size_t>(count) * dtype_bytes(dtype), stream);
  } else {
    perfmon::fill_uniform_scaled(dst, count, dtype, hdim, seed, stream);
  }
}

// --- attn_fwd ---------------------------------------------------------

void build_fwd(const pmon_entry* e, int backend, Context* ctx, hipStream_t scratch_stream) {
  const DType dt = to_aotriton_dtype(e->dtype);
  const int32_t q_heads = e->n_heads;
  if (e->gqa_ratio <= 0 || q_heads % e->gqa_ratio != 0) {
    throw std::invalid_argument("adapter_v3: n_heads not evenly divisible by gqa_ratio");
  }
  const int32_t kv_heads = q_heads / e->gqa_ratio;
  const uint64_t B = e->batch, HQ = q_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;

  auto make_and_fill = [&](Shape4 shp, bool zero) {
    const int64_t count = static_cast<int64_t>(shp.sizes[0]) * shp.sizes[1] *
                           shp.sizes[2] * shp.sizes[3];
    void* p = alloc_device(ctx, static_cast<size_t>(count) * dtype_bytes(e->dtype));
    fill(p, count, e->dtype, e->hdim, e->seed, scratch_stream, zero);
    return TensorView<4>(reinterpret_cast<intptr_t>(p), shp.sizes, shp.strides, dt);
  };

  flash::attn_fwd_params& p = ctx->fwd_params;
  // p.A (include/aotriton/flash.h:101) is left at its default-constructed
  // value: modules/flash/tune/calls.py's `attn_fwd.direct_call` -- the
  // authoritative in-repo caller for kVersion 3 -- never sets it either.
  p.Q = make_and_fill(bhsd_shape(B, HQ, SQ, D, flip), /*zero=*/false);
  p.K = make_and_fill(bhsd_shape(B, HK, SK, D, flip), /*zero=*/false);
  p.V = make_and_fill(bhsd_shape(B, HK, SK, D, flip), /*zero=*/false);

  if (e->bias_type != 0) {
    // modules/flash/tune/reference.py:106-128: physical width
    // round_to_8x(seqlen_k), logical [..., :seqlen_k] slice.
    const uint64_t padded_sk = static_cast<uint64_t>(round_to_8x(static_cast<int64_t>(SK)));
    Shape4 phys = bhsd_shape(B, HQ, SQ, padded_sk, flip);
    const int64_t alloc_count = static_cast<int64_t>(B) * HQ * SQ * padded_sk;
    void* bp = alloc_device(ctx, static_cast<size_t>(alloc_count) * dtype_bytes(e->dtype));
    fill(bp, alloc_count, e->dtype, e->hdim, e->seed ^ 0x1, scratch_stream, /*zero=*/false);
    Shape4 view = phys;
    // Logical slice to seqlen_k, always at logical index 3, regardless of
    // `flip`: modules/flash/tune/reference.py:106-128 allocates the bias
    // PHYSICALLY at (BATCH, Q_HEADS, seqlen_q, round_to_8x(seqlen_k)) --
    // already storage_flip-permuted at that point, per line 108-117's
    // `bdims = (bdims[i], bdims[j], bdims[k], bdims[l])` -- then slices
    // `b[:, :, :, :seqlen_k]` at physical index 3 BEFORE the later
    // `torch.transpose(x, y)` (line 129-136) that produces the logical
    // (B, H, S, D) view. `assert x != 3 and y != 3` (line 111) guarantees
    // that transpose never moves index 3, so the sliced dimension lands at
    // LOGICAL index 3 either way. `bhsd_shape()` above already returns
    // logical sizes {b, h, s, d} unconditionally (only strides differ under
    // flip -- see its own comment), so `view.sizes[3]` (not `flip ? 2 : 3`)
    // is the one to truncate here; the earlier `flip ? 2 : 3` version was a
    // bug caught in self-review, never shipped in a commit.
    view.sizes[3] = SK;  // last dim's stride is unaffected by the slice.
    p.B = TensorView<4>(reinterpret_cast<intptr_t>(bp), view.sizes, view.strides, dt);
  } else {
    p.B = TensorView<4>::get_null_tensor(dt);
  }

  p.Sm_scale = 1.0f / static_cast<float>(e->hdim);  // FlashInputMetadata's 'l1' default

  {
    // L (logsumexp): a real output, zeroed (rev0 §5.3). Logically
    // (B, HQ, SQ), but attn_fwd_params::L is declared T2 (rank 2,
    // include/aotriton/flash.h:103) -- (HQ, SQ) is flattened into one
    // dimension here, which is lossless because L is contiguous by
    // construction (it is never storage_flip'd -- that only applies to
    // Q/K/V/B, per modules/flash/tune/reference.py:108-136).
    const uint64_t HQ_SQ = HQ * SQ;
    std::array<uint64_t, 2> sizes{B, HQ_SQ};
    std::array<uint64_t, 2> strides{HQ_SQ, 1};
    const int64_t count = static_cast<int64_t>(B) * HQ_SQ;
    void* lp = alloc_device(ctx, static_cast<size_t>(count) * sizeof(float));
    perfmon::fill_zero(lp, static_cast<size_t>(count) * sizeof(float), scratch_stream);
    p.L = TensorView<2>(reinterpret_cast<intptr_t>(lp), sizes, strides, AOTRITON_NS::kFloat32);
  }

  p.Out = make_and_fill(bhsd_shape(B, HQ, SQ, D, flip), /*zero=*/true);

  p.dropout_p = static_cast<float>(e->dropout_p);
  p.philox_seed_ptr = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset1 = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset2 = 0;
  p.philox_seed_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.philox_offset_output = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  p.encoded_softmax = TensorView<4>::get_null_tensor(dt);
  p.persistent_atomic_counter = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);

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
  // cu_seqlens_q/k and seq_strides_q/k (T1) are left at attn_fwd_params'
  // default constructor value -- never assigned here. TensorView's own
  // default constructor (include/aotriton/util.h) only guarantees
  // `base_ == nullptr` (operator bool() == false); `varlen_type == None`
  // above is what tells the kernel not to dereference sizes()/strides() on
  // these, matching every other "disabled" tensor in this file
  // (encoded_softmax, persistent_atomic_counter, B when bias_type == 0).

  ctx->options = flash::attn_options();
  ctx->options.force_backend_index = backend;

  ctx->describe_json = "{\"backend_index\": " + std::to_string(backend) + "}";
}

// --- attn_bwd -----------------------------------------------------------

void build_bwd(const pmon_entry* e, int backend, Context* ctx, hipStream_t scratch_stream) {
  const DType dt = to_aotriton_dtype(e->dtype);
  const int32_t q_heads = e->n_heads;
  if (e->gqa_ratio <= 0 || q_heads % e->gqa_ratio != 0) {
    throw std::invalid_argument("adapter_v3: n_heads not evenly divisible by gqa_ratio");
  }
  const int32_t kv_heads = q_heads / e->gqa_ratio;
  const uint64_t B = e->batch, HQ = q_heads, HK = kv_heads;
  const uint64_t SQ = e->seqlen_q, SK = e->seqlen_k, D = e->hdim;
  const bool flip = e->storage_flip != 0;

  auto make_and_fill = [&](Shape4 shp, bool zero, uint64_t seed_salt = 0) {
    const int64_t count = static_cast<int64_t>(shp.sizes[0]) * shp.sizes[1] *
                           shp.sizes[2] * shp.sizes[3];
    void* p = alloc_device(ctx, static_cast<size_t>(count) * dtype_bytes(e->dtype));
    fill(p, count, e->dtype, e->hdim, e->seed ^ seed_salt, scratch_stream, zero);
    return TensorView<4>(reinterpret_cast<intptr_t>(p), shp.sizes, shp.strides, dt);
  };

  flash::attn_bwd_params& p = ctx->bwd_params;
  p.Q = make_and_fill(bhsd_shape(B, HQ, SQ, D, flip), false);
  p.K = make_and_fill(bhsd_shape(B, HK, SK, D, flip), false, 0x2);
  p.V = make_and_fill(bhsd_shape(B, HK, SK, D, flip), false, 0x3);

  if (e->bias_type != 0) {
    const uint64_t padded_sk = static_cast<uint64_t>(round_to_8x(static_cast<int64_t>(SK)));
    Shape4 phys = bhsd_shape(B, HQ, SQ, padded_sk, flip);
    const int64_t alloc_count = static_cast<int64_t>(B) * HQ * SQ * padded_sk;
    void* bp = alloc_device(ctx, static_cast<size_t>(alloc_count) * dtype_bytes(e->dtype));
    fill(bp, alloc_count, e->dtype, e->hdim, e->seed ^ 0x4, scratch_stream, false);
    Shape4 view = phys;
    view.sizes[3] = SK;  // see the identical, fully-explained slice in
                          // build_fwd() above -- always logical index 3.
    p.B = TensorView<4>(reinterpret_cast<intptr_t>(bp), view.sizes, view.strides, dt);

    // DB: modules/flash/tune/calls.py's `create_aotensor_like(inputs.b, ...)`
    // -> `torch.empty_like(b)` (python/tune/gpu_utils.py:428-432), whose
    // default `memory_format=torch.preserve_format` preserves a
    // non-contiguous input's strides -- so DB gets the SAME padded-then-
    // sliced physical layout as `b` itself (a logical (B,HQ,SQ,seqlen_k)
    // view over a round_to_8x(seqlen_k)-wide physical allocation), not a
    // fresh contiguous (B,HQ,SQ,SK) buffer. Reproduced identically here
    // (reusing `phys`/the slice-to-index-3 logic above) for the same
    // reason B's own padding is reproduced: a difference here could affect
    // vectorized store patterns, i.e. the very thing perfmon measures, not
    // just numerical correctness (D4 only requires finite/representative
    // values, but the physical layout is a separate concern from the
    // values written into it).
    const int64_t db_alloc_count = static_cast<int64_t>(B) * HQ * SQ * padded_sk;
    void* dbp = alloc_device(ctx, static_cast<size_t>(db_alloc_count) * dtype_bytes(e->dtype));
    fill(dbp, db_alloc_count, e->dtype, e->hdim, e->seed ^ 0x5, scratch_stream, /*zero=*/true);
    Shape4 db_view = phys;
    db_view.sizes[3] = SK;
    p.DB = TensorView<4>(reinterpret_cast<intptr_t>(dbp), db_view.sizes, db_view.strides, dt);
  } else {
    p.B = TensorView<4>::get_null_tensor(dt);
    p.DB = TensorView<4>::get_null_tensor(dt);
  }

  p.Sm_scale = 1.0f / static_cast<float>(e->hdim);

  // Out/DO: in a real pipeline Out is attn_fwd's output and DO is dOut from
  // the loss; measured standalone here (D3: op-level fanout, no end-to-end
  // chaining), so both are filled as generic finite inputs (see the
  // "documented simplifications" note above the includes).
  p.Out = make_and_fill(bhsd_shape(B, HQ, SQ, D, flip), false, 0x6);
  p.DO = make_and_fill(bhsd_shape(B, HQ, SQ, D, flip), false, 0x7);
  p.DK = make_and_fill(bhsd_shape(B, HK, SK, D, flip), /*zero=*/true, 0x8);
  p.DV = make_and_fill(bhsd_shape(B, HK, SK, D, flip), /*zero=*/true, 0x9);
  p.DQ = make_and_fill(bhsd_shape(B, HQ, SQ, D, flip), /*zero=*/true, 0xA);

  {
    const uint64_t HQ_SQ = HQ * SQ;
    std::array<uint64_t, 2> sizes{B, HQ_SQ};
    std::array<uint64_t, 2> strides{HQ_SQ, 1};
    const int64_t count = static_cast<int64_t>(B) * HQ_SQ;

    void* lp = alloc_device(ctx, static_cast<size_t>(count) * sizeof(float));
    perfmon::fill_uniform_scaled(lp, count, PMON_DTYPE_FLOAT32, e->hdim, e->seed ^ 0xB,
                                  scratch_stream);
    p.L = TensorView<2>(reinterpret_cast<intptr_t>(lp), sizes, strides, AOTRITON_NS::kFloat32);

    void* dp = alloc_device(ctx, static_cast<size_t>(count) * sizeof(float));
    perfmon::fill_uniform_scaled(dp, count, PMON_DTYPE_FLOAT32, e->hdim, e->seed ^ 0xC,
                                  scratch_stream);
    TensorView<2> delta_view(reinterpret_cast<intptr_t>(dp), sizes, strides,
                              AOTRITON_NS::kFloat32);
    p.D.eager = delta_view;  // eager-wrapped real buffer, not lazily computed
                              // -- matches modules/flash/tune/calls.py's
                              // `eager_delta()` pattern (wraps a precomputed
                              // buffer, LazyTensor is just the typing
                              // mechanism, not deferred computation here).
  }

  // dq_acc: only backend 2 (AiterFmhaV3Bwd) reads/accumulates it
  // (modules/flash/tune/level_op.py:108); backends 0/1 get a null-eager
  // view sized like DQ (matches modules/flash/tune/calls.py's
  // `eager_null_dq_acc`: "data_ptr=0: Triton kernel will not access
  // dq_acc, so the null pointer is safe").
  {
    Shape4 shp = bhsd_shape(B, HQ, SQ, D, /*storage_flip=*/false);  // DQ_ACC is
    // always a fresh fp32 accumulator allocation (modules/flash/tune/
    // level_op.py:121: `torch.zeros(*devm.q.size(), dtype=torch.float32,
    // ...)`), never storage_flip'd -- it is not one of Q/K/V's actual
    // storage buffers, just shaped like Q.
    if (backend == 2) {
      const int64_t count = static_cast<int64_t>(B) * HQ * SQ * D;
      ctx->dq_acc_bytes = static_cast<size_t>(count) * sizeof(float);
      ctx->dq_acc_ptr = alloc_device(ctx, ctx->dq_acc_bytes);
      // Zeroed for the first time here; launch() re-zeroes on every call
      // (see the file-level dq_acc discussion above) since backend 2
      // accumulates into it.
      perfmon::fill_zero(ctx->dq_acc_ptr, ctx->dq_acc_bytes, scratch_stream);
      p.DQ_ACC.eager =
          TensorView<4>(reinterpret_cast<intptr_t>(ctx->dq_acc_ptr), shp.sizes, shp.strides,
                        AOTRITON_NS::kFloat32);
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
  // cu_seqlens_q/k / seq_strides_q/k left at attn_bwd_params' default
  // constructor value -- see the identical note in build_fwd() above.

  ctx->options = flash::attn_options();
  ctx->options.force_backend_index = backend;

  ctx->describe_json = "{\"backend_index\": " + std::to_string(backend) + "}";
}

}  // namespace

// --- vtable entry points (extern "C", perfmon_abi.h's exact signatures) ---

extern "C" int pmon_flash_enumerate_backends(const pmon_entry* entry, int* out, int max) {
  try {
    const bool is_942_950 = current_gpu_is_942_or_950();
    const int count = backend_count(entry->iface, is_942_950);
    if (count == 0) return -1;  // unknown iface
    // Fail loudly rather than silently truncating (see the file-level note
    // on this being an underspecified corner of the ABI): a caller sizing
    // its buffer to `max` and getting back fewer entries than actually
    // exist would otherwise never learn its buffer was too small.
    if (count > max) return -1;
    for (int i = 0; i < count; ++i) out[i] = i;
    return count;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter_v3 enumerate_backends failed: %s\n", ex.what());
    return -1;
  }
}

extern "C" int pmon_flash_prepare(const pmon_entry* entry, int backend, void** out_ctx) {
  auto* ctx = new Context();
  ctx->iface = entry->iface;
  ctx->backend = backend;
  hipStream_t scratch = nullptr;
  try {
    const bool is_942_950 = current_gpu_is_942_or_950();
    const int count = backend_count(entry->iface, is_942_950);
    if (backend < 0 || backend >= count) {
      throw std::invalid_argument("adapter_v3: backend index out of range");
    }
    if (entry->varlen != 0) {
      throw std::invalid_argument("adapter_v3: varlen entries are not supported "
                                   "(no varlen axis exists in the entry space yet, "
                                   "see modules/flash/perfmon/entry.py's KNOWN GAPS)");
    }
    hip_check(hipStreamCreate(&scratch), "hipStreamCreate (prepare scratch stream)");
    if (entry->iface == PMON_IFACE_ATTN_FWD) {
      build_fwd(entry, backend, ctx, scratch);
    } else if (entry->iface == PMON_IFACE_ATTN_BWD) {
      build_bwd(entry, backend, ctx, scratch);
    } else {
      throw std::invalid_argument("adapter_v3: unknown iface index: " +
                                   std::to_string(entry->iface));
    }
    hip_check(hipStreamSynchronize(scratch), "hipStreamSynchronize (prepare scratch stream)");
    hip_check(hipStreamDestroy(scratch), "hipStreamDestroy (prepare scratch stream)");
    *out_ctx = ctx;
    return 0;
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "adapter_v3 prepare failed: %s\n", ex.what());
    if (scratch) hipStreamDestroy(scratch);
    for (auto it = ctx->owned_ptrs.rbegin(); it != ctx->owned_ptrs.rend(); ++it) {
      hipFree(*it);
    }
    delete ctx;
    return -1;
  }
}

extern "C" int pmon_flash_launch(void* ctx_v, hipStream_t stream) {
  auto* ctx = static_cast<Context*>(ctx_v);
  hipError_t err = hipSuccess;
  if (ctx->iface == PMON_IFACE_ATTN_FWD) {
    err = flash::attn_fwd(ctx->fwd_params, flash::attn_fwd_params::kVersion, Stream(stream),
                           &ctx->options);
  } else if (ctx->iface == PMON_IFACE_ATTN_BWD) {
    if (ctx->backend == 2) {
      // Re-zero dq_acc on every call -- see the file-level dq_acc
      // discussion above. Stream-ordered, so this becomes one memset node
      // per captured iteration inside timing.cc's hipgraph_ev100 capture.
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
  auto* ctx = static_cast<Context*>(ctx_v);
  return ctx->describe_json.c_str();
}

extern "C" void pmon_flash_release(void* ctx_v) {
  auto* ctx = static_cast<Context*>(ctx_v);
  for (auto it = ctx->owned_ptrs.rbegin(); it != ctx->owned_ptrs.rend(); ++it) {
    hipFree(*it);
  }
  delete ctx;
}

namespace {
const pmon_family_vtable kFlashVtable = {
    pmon_flash_enumerate_backends,
    pmon_flash_prepare,
    pmon_flash_launch,
    pmon_flash_describe,
    pmon_flash_release,
};
}  // namespace

extern "C" const pmon_family_vtable* pmon_family_entry(void) {
  return &kFlashVtable;
}
