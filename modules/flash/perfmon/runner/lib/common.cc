// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// See lib/common.h for what does and does not belong in this file.

#include "common.h"

#include <perfmon/fill.h>

#include <stdexcept>
#include <string>

namespace perfmon_flash {

int64_t round_to_8x(int64_t n) {
  return 8 * ((n + 7) / 8);
}

size_t dtype_bytes(int32_t pmon_dtype) {
  switch (pmon_dtype) {
    case PMON_DTYPE_FLOAT16:  return 2;
    case PMON_DTYPE_BFLOAT16: return 2;
    case PMON_DTYPE_FLOAT32:  return 4;
    default:
      throw std::invalid_argument("perfmon_flash: unknown pmon_dtype: " +
                                  std::to_string(pmon_dtype));
  }
}

AOTRITON_NS::DType to_aotriton_dtype(int32_t pmon_dtype) {
  switch (pmon_dtype) {
    case PMON_DTYPE_FLOAT16:  return AOTRITON_NS::kFloat16;
    case PMON_DTYPE_BFLOAT16: return AOTRITON_NS::kBFloat16;
    case PMON_DTYPE_FLOAT32:  return AOTRITON_NS::kFloat32;
    default:
      throw std::invalid_argument("perfmon_flash: unknown pmon_dtype: " +
                                  std::to_string(pmon_dtype));
  }
}

void hip_check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("perfmon_flash: ") + what + " failed: " +
                             hipGetErrorString(err));
  }
}

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

int backend_count_forceable(int32_t iface, bool is_942_950) {
  switch (iface) {
    case PMON_IFACE_ATTN_FWD: return is_942_950 ? 2 : 1;
    case PMON_IFACE_ATTN_BWD: return is_942_950 ? 3 : 2;
    default: return 0;
  }
}

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

void* DeviceArena::alloc(size_t bytes) {
  void* p = nullptr;
  hip_check(hipMalloc(&p, bytes), "hipMalloc");
  owned_ptrs.push_back(p);
  return p;
}

void DeviceArena::release_all() {
  for (auto it = owned_ptrs.rbegin(); it != owned_ptrs.rend(); ++it) {
    (void)hipFree(*it);
  }
  owned_ptrs.clear();
}

void fill(void* dst, int64_t count, int32_t dtype, int32_t hdim, uint64_t seed,
          hipStream_t stream, bool zero) {
  if (zero) {
    perfmon::fill_zero(dst, static_cast<size_t>(count) * dtype_bytes(dtype), stream);
  } else {
    perfmon::fill_uniform_scaled(dst, count, dtype, hdim, seed, stream);
  }
}

AOTRITON_NS::TensorView<4> make_and_fill4(DeviceArena* arena, Shape4 shp, int32_t dtype,
                                          int32_t hdim, uint64_t seed, hipStream_t stream,
                                          bool zero) {
  const int64_t count = static_cast<int64_t>(shp.sizes[0]) * shp.sizes[1] *
                        shp.sizes[2] * shp.sizes[3];
  void* p = arena->alloc(static_cast<size_t>(count) * dtype_bytes(dtype));
  fill(p, count, dtype, hdim, seed, stream, zero);
  return AOTRITON_NS::TensorView<4>(reinterpret_cast<intptr_t>(p), shp.sizes, shp.strides,
                                    to_aotriton_dtype(dtype));
}

AOTRITON_NS::TensorView<2> make_lse_like(DeviceArena* arena, uint64_t b, uint64_t hq,
                                         uint64_t sq, int32_t hdim, uint64_t seed,
                                         hipStream_t stream, bool zero) {
  const uint64_t hq_sq = hq * sq;
  std::array<uint64_t, 2> sizes{b, hq_sq};
  std::array<uint64_t, 2> strides{hq_sq, 1};
  const int64_t count = static_cast<int64_t>(b) * hq_sq;
  void* p = arena->alloc(static_cast<size_t>(count) * sizeof(float));
  // Always fp32, regardless of the entry's own dtype.
  fill(p, count, PMON_DTYPE_FLOAT32, hdim, seed, stream, zero);
  return AOTRITON_NS::TensorView<2>(reinterpret_cast<intptr_t>(p), sizes, strides,
                                    AOTRITON_NS::kFloat32);
}

AOTRITON_NS::TensorView<4> make_bias(DeviceArena* arena, uint64_t b, uint64_t hq, uint64_t sq,
                                     uint64_t sk, int32_t dtype, int32_t hdim, uint64_t seed,
                                     hipStream_t stream, bool zero, bool storage_flip) {
  const uint64_t padded_sk = static_cast<uint64_t>(round_to_8x(static_cast<int64_t>(sk)));
  Shape4 phys = bhsd_shape(b, hq, sq, padded_sk, storage_flip);
  const int64_t alloc_count = static_cast<int64_t>(b) * hq * sq * padded_sk;
  void* p = arena->alloc(static_cast<size_t>(alloc_count) * dtype_bytes(dtype));
  fill(p, alloc_count, dtype, hdim, seed, stream, zero);

  // Logical slice to seqlen_k at logical index 3, regardless of
  // storage_flip. modules/flash/tune/reference.py allocates the bias
  // physically already storage_flip-permuted, then slices at physical index
  // 3 BEFORE the transpose that produces the logical (B, H, S, D) view --
  // and its own `assert x != 3 and y != 3` guarantees that transpose never
  // moves index 3. bhsd_shape() returns logical sizes unconditionally (only
  // strides differ under flip), so sizes[3] is the one to truncate here.
  Shape4 view = phys;
  view.sizes[3] = sk;  // the last dim's stride is unaffected by the slice
  return AOTRITON_NS::TensorView<4>(reinterpret_cast<intptr_t>(p), view.sizes, view.strides,
                                    to_aotriton_dtype(dtype));
}

int32_t kv_heads_of(const pmon_entry* e) {
  if (e->gqa_ratio <= 0 || e->n_heads % e->gqa_ratio != 0) {
    throw std::invalid_argument("perfmon_flash: n_heads not evenly divisible by gqa_ratio");
  }
  return e->n_heads / e->gqa_ratio;
}

void validate_entry(const pmon_entry* e, int backend, int backend_count) {
  if (backend < 0 || backend >= backend_count) {
    throw std::invalid_argument("perfmon_flash: backend index out of range");
  }
  if (e->varlen != 0) {
    throw std::invalid_argument("perfmon_flash: varlen entries are not supported "
                                "(no varlen axis exists in the entry space yet, see "
                                "modules/flash/perfmon/entry.py's KNOWN GAPS)");
  }
}

}  // namespace perfmon_flash
