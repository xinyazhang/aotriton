// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T09: deterministic on-device input fill (rev0 §5.3).
//
// UNVERIFIED IN THIS ENVIRONMENT: this machine has no ROCm/hipcc/HIP
// headers and no GPU (see the T05/T06 commit messages for how that was
// confirmed). fill.h/fill.cc have never been compiled. Written strictly to
// rev0 §5.3 and this task's spec; see fill.cc's top-of-file comment for the
// specific HIP/hip_bfloat16 API surface this assumes.

#ifndef PERFMON_CORE_FILL_H
#define PERFMON_CORE_FILL_H

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace perfmon {

// Deterministically fills `count` elements at device pointer `dst` with
// values uniform in [-1, 1), scaled by 1/sqrt(hdim), using a cheap
// counter-based PRNG (splitmix64, mixed per-element with `seed`) -- rev0
// §5.3: "seed = hash(functional_pon, shape_pon, subject, iface)",
// COMPUTED BY THE CALLER (T10's measurement orchestration owns that hash;
// this function only consumes the already-derived `pmon_entry.seed`,
// perfmon_abi.h) and passed straight through unchanged so the same seed
// always reproduces the same buffer contents.
//
// `dtype` selects the on-device numeric format (perfmon_abi.h's
// pmon_dtype: PMON_DTYPE_FLOAT16 / PMON_DTYPE_BFLOAT16 / PMON_DTYPE_FLOAT32).
//
// A guard pass -- fused into the same kernel launch, not a second pass
// over global memory -- rejects non-finite and subnormal draws by
// re-mixing the counter and redrawing in place (bounded retries; see
// fill.cc). This is not an optional cleanup step: uninitialized VRAM read
// back as fp16/bf16 is frequently NaN/inf/denormal, and both propagate
// through softmax and can measurably change kernel timing on some
// hardware paths, which is the entire reason this task exists (rev0 §5.3).
//
// Q/K/V and (when present) bias and `do` are filled this way; outputs and
// `dq_acc` are NOT filled here -- see `fill_zero` below (rev0 §5.3:
// "dq_acc and outputs are zeroed").
//
// Enqueued on `stream`; callable inside a hipGraph capture, same
// constraint as pmon_family_vtable::launch (perfmon_abi.h) -- no host
// synchronization, only stream-ordered work.
void fill_uniform_scaled(void* dst, int64_t count, int32_t dtype, int32_t hdim,
                          uint64_t seed, hipStream_t stream);

// Zeroes `bytes` at device pointer `dst` on `stream`. A thin, documented
// wrapper over hipMemsetAsync so every "this buffer must be zero, not
// random" call site (dq_acc, outputs) goes through one place instead of
// each adapter reimplementing the size-in-bytes bookkeeping. Capture-safe,
// same as fill_uniform_scaled.
void fill_zero(void* dst, size_t bytes, hipStream_t stream);

// T09's Verify step: downloads `count` elements of `dtype` from device
// pointer `src` (host-synchronous -- NOT for use on any timed/hot path,
// debug/test harness only) and returns true iff every element is finite
// and either exactly zero or normal (a subnormal is rejected; exact zero
// is not, since it is not what the guard pass in fill_uniform_scaled is
// built to keep out and it is also what fill_zero legitimately produces).
bool debug_verify_finite_normal(const void* src, int64_t count, int32_t dtype);

}  // namespace perfmon

#endif  // PERFMON_CORE_FILL_H
