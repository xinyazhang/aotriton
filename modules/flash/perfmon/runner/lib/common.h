// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Routines shared by every per-tag flash adapter under
// modules/flash/perfmon/runner/<tag>/adapter.cc.
//
// --- What belongs here, and what does not --------------------------------
//
// rev0 §5 requires that "a version's API churn is contained in one small
// file that is expected to differ per version, instead of accreting
// conditionals in shared code" -- so the dividing line is NOT "code used
// more than once". It is: does this depend on the part of AOTriton's API
// that drifts between tags?
//
// Verified stable across 0.9.2b, 0.10b, 0.11b, 0.11.2b, 0.12.1b, 0.13b and
// HEAD, by reading each tag's own include/aotriton/{util,dtypes}.h:
//   - TensorView<Rank>(intptr_t base, sizes, strides, DType)
//   - TensorView<0>(intptr_t base, DType)
//   - TensorView<Rank>::get_null_tensor(DType)
//   - kFloat16 / kBFloat16 / kFloat32 / kInt32 / kUInt64
// So buffer allocation, deterministic filling, dtype mapping, shape/stride
// computation and the GPU-arch probe all live here.
//
// What drifts, and therefore stays in the per-tag adapter:
//   - the params structs (v2 has none; v3's gained LazyTensor, then `eager`)
//   - attn_options (absent at 0.9.2b, empty at 0.10b, has
//     force_backend_index from 0.11b)
//   - the entry point's own signature (v2: a long positional free function;
//     v3: a params struct + version + stream + options)
//   - how many backends can be forced, which is a consequence of the above
//
// Everything here is compiled fresh per subject alongside that subject's
// adapter -- "shared" means shared source, never a shared binary. The one
// binary genuinely shared across subjects is libperfmon_core, which sees no
// AOTriton header at all (rev0 D4).

#ifndef PERFMON_FLASH_LIB_COMMON_H
#define PERFMON_FLASH_LIB_COMMON_H

#include <perfmon/perfmon_abi.h>

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/util.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace perfmon_flash {

// flash's own iface ordering -- deliberately NOT in perfmon_abi.h (iface
// names are a family concern, D4). Must match
// modules/flash/perfmon/__init__.py's `PerfDesc.IFACES = ('attn_fwd',
// 'attn_bwd')` exactly, since that Python ordering is what assigns
// `pmon_entry.iface`'s value in the first place (rev0 D8).
constexpr int32_t PMON_IFACE_ATTN_FWD = 0;
constexpr int32_t PMON_IFACE_ATTN_BWD = 1;

int64_t round_to_8x(int64_t n);
size_t dtype_bytes(int32_t pmon_dtype);
AOTRITON_NS::DType to_aotriton_dtype(int32_t pmon_dtype);
void hip_check(hipError_t err, const char* what);

// True on gfx942/gfx950, matching modules/flash/tune/level_op.py's
// `_gpu_arch()`. There is no arch string on `pmon_entry`, and the whole
// runner process is pinned to one GPU (rev0 D4), so device 0 is queried.
bool current_gpu_is_942_or_950();

// The backend counts modules/flash/tune/level_op.py specifies (attn_fwd
// 2/1, attn_bwd 3/2). Only meaningful for tags whose attn_options can
// actually force a backend index -- 0.9.2b and 0.10b cannot, and their
// adapters report 1 instead of calling this.
int backend_count_forceable(int32_t iface, bool is_942_950);

struct Shape4 {
  std::array<uint64_t, 4> sizes;
  std::array<uint64_t, 4> strides;
};

// Logical shape (B, H, S, D) with strides for the contiguous layout, or --
// when `storage_flip` -- the layout modules/flash/tune/reference.py
// produces: allocate physically as (B, S, H, D) contiguous, then transpose
// back to logical (B, H, S, D). Sizes are unaffected either way; only
// strides differ.
Shape4 bhsd_shape(uint64_t b, uint64_t h, uint64_t s, uint64_t d, bool storage_flip);

// Owns every device allocation one measurement context makes, freed in
// reverse order.
struct DeviceArena {
  std::vector<void*> owned_ptrs;
  void* alloc(size_t bytes);
  void release_all();
};

// Fills `count` elements of `dtype` at `dst` with perfmon's deterministic
// RNG (rev0 §5.3), scaled for `hdim`, on `stream`. `zero` selects
// perfmon::fill_zero instead (true outputs -- rev0 §5.3: "dq_acc and
// outputs are zeroed").
void fill(void* dst, int64_t count, int32_t dtype, int32_t hdim, uint64_t seed,
          hipStream_t stream, bool zero);

// Allocate + fill a rank-4 buffer and wrap it in a TensorView. The single
// most repeated operation in every adapter.
AOTRITON_NS::TensorView<4> make_and_fill4(DeviceArena* arena, Shape4 shp, int32_t dtype,
                                          int32_t hdim, uint64_t seed, hipStream_t stream,
                                          bool zero);

// L/D's rank-2 (B, H*S) buffer. Logically (B, HQ, SQ), but every tag in
// range declares L as rank 2; flattening (HQ, SQ) is lossless because it is
// contiguous by construction (never storage_flip'd -- that applies only to
// Q/K/V/B, per modules/flash/tune/reference.py).
AOTRITON_NS::TensorView<2> make_lse_like(DeviceArena* arena, uint64_t b, uint64_t hq,
                                         uint64_t sq, int32_t hdim, uint64_t seed,
                                         hipStream_t stream, bool zero);

// The bias buffer's padded-then-sliced layout: physical width
// round_to_8x(seqlen_k), logical [..., :seqlen_k] slice at logical index 3.
// Reproduced from modules/flash/tune/reference.py rather than simplified to
// an unpadded contiguous width, because deviating could change memory
// access performance -- the very thing perfmon measures.
AOTRITON_NS::TensorView<4> make_bias(DeviceArena* arena, uint64_t b, uint64_t hq, uint64_t sq,
                                     uint64_t sk, int32_t dtype, int32_t hdim, uint64_t seed,
                                     hipStream_t stream, bool zero, bool storage_flip);

// kv_heads = n_heads / gqa_ratio (perfmon_abi.h's own doc comment). Throws
// rather than truncating if it does not divide evenly.
int32_t kv_heads_of(const pmon_entry* e);

// Rejects entries no adapter in this tree supports, so each adapter does
// not restate them: varlen (no varlen axis exists in the entry space yet --
// see modules/flash/perfmon/entry.py's KNOWN GAPS) and a backend index
// outside the count the caller reports.
void validate_entry(const pmon_entry* e, int backend, int backend_count);

}  // namespace perfmon_flash

// --- the vtable's five entry points --------------------------------------
// Defined by each <tag>/adapter.cc; wired into pmon_family_entry() by
// lib/vtable.cc, which is identical for every tag and so is not restated in
// any of them.
extern "C" int pmon_flash_enumerate_backends(const pmon_entry* entry, int* out, int max);
extern "C" int pmon_flash_prepare(const pmon_entry* entry, int backend, void** out_ctx);
extern "C" int pmon_flash_launch(void* ctx, hipStream_t stream);
extern "C" const char* pmon_flash_describe(void* ctx);
extern "C" void pmon_flash_release(void* ctx);

#endif  // PERFMON_FLASH_LIB_COMMON_H
