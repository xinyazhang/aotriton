// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_MODULES_FLASH_CSRC_FLYC_COMMON_H
#define AOTRITON_MODULES_FLASH_CSRC_FLYC_COMMON_H

// The host-side translations every flyc flash kernel needs: AOTriton's varlen
// encoding into FlyDSL's, and AOTriton's float dropout probability into
// FlyDSL's threshold/scale pair.
//
// These were `FlycAttnFwdContext` member functions until the backward kernels
// arrived. They are free functions taking plain scalars rather than a params
// struct, because the two params structs are different types that spell the
// same field differently -- `OpAttnFwdParams::Num_seqlens` against
// `OpAttnBwdParams::num_seqlens` -- so neither a shared base nor a template
// over the struct would bind. Passing the three inputs explicitly is the only
// spelling that serves both, and it has the side benefit of being directly
// unit-testable (modules/flash/tests/test_flyc_varlen_translation.cc) without
// constructing a context.
//
// `flyc_varlen.h` stays separate and holds the bit encoding itself. This
// header is the AOTriton-to-FlyDSL mapping that decides WHICH encoding a given
// call needs; that file is what those encodings ARE.

#include <aotriton/config.h>
#include <aotriton/_internal/log.h>

#include "flyc_varlen.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

// The three numbers a flyc kernel launch needs to describe its layout.
//
// `batch_size` is no longer a kernarg -- it left the FlyDSL signature in
// 67a3ace0 -- but it is still needed, because the grid's z extent is
// `num_seqlens != 0 ? num_seqlens : batch_size` and AOTriton computes its own
// grid. So it survives here as a grid input rather than an argument.
struct FlycVarlenRow {
  int32_t varlen_bits;
  int32_t batch_size;
  int32_t num_seqlens;

  // The kernel's own z-extent expression, spelled once.
  int32_t nseq_idx() const { return num_seqlens != 0 ? num_seqlens : batch_size; }
};

// Classifies a call into one row of the varlen table and, in the same place,
// checks the one invariant that makes the classification safe to trust: the
// kernel's `nseq_idx` must agree with what AOTriton computes independently
// from Num_seqlens and Q's batch axis. A silent disagreement here is not a
// crash -- it is a kernel launched over the wrong number of programs,
// addressing an in-bounds row that is simply the wrong one.
//
// `num_seqlens` is AOTriton's SIGNED three-way value (>0 packed, 0 dense, <0
// BHSD-padded); `q_batch` is Q->size(0), which is not Batch under packed
// varlen; `seq_strides_q_present` is whether a dedicated position array was
// supplied, the only thing separating strided varlen from compact.
inline FlycVarlenRow
flyc_classify_varlen(int32_t num_seqlens, int32_t q_batch, bool seq_strides_q_present) {
  FlycVarlenRow row;
  if (num_seqlens == 0) {
    // Dense: nothing packed, Q is BHSD with q_batch == B.
    row = { static_cast<int32_t>(kFlycVarlenDense), q_batch, 0 };
  } else if (num_seqlens < 0) {
    // Padded varlen: BHSD, one sequence per batch slot, q_batch == N. The
    // count already arrived via Q's own batch axis, so num_seqlens is 0 --
    // FlyDSL's num_seqlens means "how many sequences are packed into a 1THD
    // tensor", and padded packs nothing.
    row = { static_cast<int32_t>(kFlycVarlenPadded), q_batch, 0 };
  } else if (!seq_strides_q_present) {
    // Compact varlen: 1THD, position reused from the cumulative length array,
    // so no seq_strides_q tensor is supplied.
    row = { static_cast<int32_t>(kFlycVarlenCompact), q_batch, num_seqlens };
  } else {
    // Strided varlen: 1THD, position read from a dedicated array.
    row = { static_cast<int32_t>(kFlycVarlenStrided), q_batch, num_seqlens };
  }

  const int32_t expected = num_seqlens == 0 ? q_batch
                          : (num_seqlens < 0 ? -num_seqlens : num_seqlens);
  if (row.nseq_idx() != expected) {
    AOTRITON_LOG(LOG_ERROR,
                 "flyc varlen translation mismatch: kernel's nseq_idx would be %d "
                 "(batch_size=%d, num_seqlens=%d) but AOTriton independently computes "
                 "%d from Num_seqlens=%d, Q->size(0)=%d -- refusing to launch",
                 row.nseq_idx(), row.batch_size, row.num_seqlens, expected, num_seqlens, q_batch);
    throw std::runtime_error("flyc varlen translation mismatch (nseq_idx)");
  }
  return row;
}

// The i32 dropout threshold. FlyDSL's `philox.dropout_threshold` (in
// modules/flash/flyc/philox.py, reached from fmha_abi_gfx1201.py's
// dropout_args): a uniform u32 reinterpreted as i32 is uniform on
// [-2**31, 2**31), so comparing against `(p - 0.5) * 0xFFFFFFFF` keeps a
// `1 - p` fraction with one signed compare and no per-element float
// conversion. This is bit-for-bit the same formula AOTriton's own Triton
// kernels already use (modules/flash/kernel/fwd_kernel.py and siblings:
// `((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)`), computed here in double
// precision and clamped rather than truncated by a narrowing cast, matching
// `dropout_threshold`'s own `max(-(2**31), min(2**31 - 1, t))`.
//
// When dropout is disabled, FlyDSL's `dropout_args` returns threshold 0 and
// scale 1.0 without evaluating the formula at all; mirrored here rather than
// evaluating it at p == 0, which is not the same value.
inline int32_t
flyc_idropout_p_of(bool enable_dropout, float dropout_p) {
  if (!enable_dropout) {
    return 0;
  }
  const double p = static_cast<double>(dropout_p);
  const double raw = (p - 0.5) * static_cast<double>(0xFFFFFFFFu);
  const int64_t t = static_cast<int64_t>(raw);
  constexpr int64_t kInt32Min = -(int64_t{1} << 31);
  constexpr int64_t kInt32Max = (int64_t{1} << 31) - 1;
  return static_cast<int32_t>(std::clamp(t, kInt32Min, kInt32Max));
}

inline float
flyc_dropout_scale_of(bool enable_dropout, float dropout_p) {
  if (!enable_dropout) {
    return 1.0f;
  }
  return 1.0f / (1.0f - dropout_p);
}

// Item I sub-step (f): find the smallest COMPILED rung >= `hdim` in the
// arch-indexed table codegen_compiled_rung_table_defs (python/codegen/flyc.py)
// generates on each FlycXXXContext (`compiled_block_dmodel` /
// `compiled_block_dmodel_count`, both static). Mirrors
// <aotriton/_internal/util.h>'s round_value -- used at the OPERATOR's own
// BLOCK_DMODEL binning site, attn_fwd.cc / attn_bwd.cc, BEFORE a backend is
// chosen -- but over a raw (pointer, count) pair rather than a
// std::vector<int32_t>, since that is what the generated static arrays are.
//
// Returns -1, the same "not found" sentinel round_value uses, when `table`
// is empty (this arch compiles nothing for this kernel -- e.g. gfx950 for a
// gfx1201-only flyc kernel, see codegen/flyc.py's comment on why that row is
// legitimately empty) or when `hdim` exceeds every compiled rung. -1 is never
// a valid BLOCK_DMODEL choice, so it reaches godel_number()'s digit-compare
// unchanged, is rejected there with a logged "Unsupported" message, and
// lookup_optimal() returns hipErrorNotSupported -- the same graceful
// rejection shape as any other out-of-range functional value, not a crash
// and not a silently wrong kernel.
inline int16_t
flyc_round_up_rung(int32_t hdim, const int16_t* table, int count) {
  for (int i = 0; i < count; ++i) {
    if (table[i] >= hdim) {
      return table[i];
    }
  }
  return -1;
}

}  // namespace AOTRITON_NS::v3::flash

#endif
