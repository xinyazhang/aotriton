// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone test for modules/flash/csrc/flyc_common.h's varlen translation --
// there is no C++ test harness anywhere in this project (no gtest, no CMake
// add_test target), so this is a plain main() rather than an invented
// framework. It is NOT part of the CMake build: v3src/CMakeLists.txt globs
// "modules/*/csrc/*.cc" recursively, and this file deliberately lives under
// modules/flash/tests/ (no "csrc" path component) so it is never swept into
// libaotriton_v2.so.
//
// It used to drive the translation through `FlycAttnFwdContext`, which needed
// a fake parent context and a generated per-kernel header just to reach three
// member functions. The translation is now a free function taking scalars
// (it has to be: the forward and backward params structs spell `Num_seqlens`
// differently), so the test calls it directly and needs neither.
//
// Exercises the FOUR rows of the varlen table
// (modules/flash/flyc/PLAN-PHASE2.md section 7.1 / FlyDSL's
// sdpa-varlen-plan.md section 1.3), each checked as the triple
// (varlen_bits, batch_size, num_seqlens) -- a single dense-only test would
// pass three of the four wrong implementations -- and calls
// flyc_varlen_selfcheck() once, per modules/flash/csrc/flyc_varlen.h's own
// instructions (the bitfield-decodes-the-encoder check clang cannot make at
// compile time).
//
// Build -- header-only now, so no generated per-kernel header and no .cc to
// compile alongside:
//
//   g++ -std=c++20 -D__HIP_PLATFORM_AMD__=1
//       -I . -I include -I <any build dir>/include
//       modules/flash/tests/test_flyc_varlen_translation.cc
//       -o /tmp/flyc_varlen_test
//   /tmp/flyc_varlen_test
//
// The build dir is only needed for aotriton/config.h. No HIP headers, no
// generated per-kernel header, no libaotriton_v2.so.

#include <aotriton/config.h>

#include "../csrc/flyc_common.h"

#include <cstdio>
#include <cstdlib>

// AOTRITON_LOG on flyc_classify_varlen's error path pulls in three symbols that
// live in libaotriton_v2.so. Stubbed rather than linked, so this stays a
// two-command test with no library build: the mismatch branch it guards is
// unreachable from the four rows below, all of which are valid by
// construction.
namespace AOTRITON_NS {
const DebugConfig& debug_config() { static DebugConfig c{}; return c; }
const char* log_level_name(int) noexcept { return "TEST"; }
const char* log_basename(const char* path) noexcept { return path; }
} // namespace AOTRITON_NS

namespace {

int g_failures = 0;

void
check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s\n", what);
  }
}

void
check_row(const char* name, int32_t num_seqlens_in, int32_t q_batch,
          bool seq_strides_q_present, int32_t expect_bits, int32_t expect_batch,
          int32_t expect_nseq) {
  const auto row = AOTRITON_NS::v3::flash::flyc_classify_varlen(
      num_seqlens_in, q_batch, seq_strides_q_present);

  std::fprintf(stderr, "-- %s: varlen_bits=0x%04x batch_size=%d num_seqlens=%d\n",
               name, row.varlen_bits, row.batch_size, row.num_seqlens);

  char msg[128];
  std::snprintf(msg, sizeof(msg), "%s: varlen_bits", name);
  check(row.varlen_bits == expect_bits, msg);
  std::snprintf(msg, sizeof(msg), "%s: batch_size", name);
  check(row.batch_size == expect_batch, msg);
  std::snprintf(msg, sizeof(msg), "%s: num_seqlens", name);
  check(row.num_seqlens == expect_nseq, msg);
}

} // anonymous namespace

int
main() {
  using AOTRITON_NS::v3::flash::kFlycVarlenCompact;
  using AOTRITON_NS::v3::flash::kFlycVarlenDense;
  using AOTRITON_NS::v3::flash::kFlycVarlenPadded;
  using AOTRITON_NS::v3::flash::kFlycVarlenStrided;

  // Row 1: dense. Num_seqlens == 0, batch_size == Q->size(0) == B, no packing.
  check_row("dense", /*num_seqlens=*/0, /*q_batch=*/4, /*seq_strides_q=*/false,
            static_cast<int32_t>(kFlycVarlenDense), /*batch=*/4, /*nseq=*/0);

  // Row 2: compact varlen. Num_seqlens > 0, seq_strides_q is a null tensor.
  // batch_size == Q->size(0) == 1 (packed 1THD), not Batch.
  check_row("compact", /*num_seqlens=*/7, /*q_batch=*/1, /*seq_strides_q=*/false,
            static_cast<int32_t>(kFlycVarlenCompact), /*batch=*/1, /*nseq=*/7);

  // Row 3: padded varlen. Num_seqlens < 0; batch_size == Q->size(0) == N and
  // must equal abs(Num_seqlens) -- exactly the mandatory assertion's
  // non-trivial case. num_seqlens reported as 0: nothing is "packed", the
  // count already rode in on batch_size.
  check_row("padded", /*num_seqlens=*/-5, /*q_batch=*/5, /*seq_strides_q=*/false,
            static_cast<int32_t>(kFlycVarlenPadded), /*batch=*/5, /*nseq=*/0);

  // Row 4: strided varlen. Num_seqlens > 0, seq_strides_q is NOT a null
  // tensor -- the only bit distinguishing this row from compact.
  check_row("strided", /*num_seqlens=*/9, /*q_batch=*/1, /*seq_strides_q=*/true,
            static_cast<int32_t>(kFlycVarlenStrided), /*batch=*/1, /*nseq=*/9);

  // The check the language will not make at compile time on clang: that the
  // bitfield view decodes what the shift-based encoder produced.
  check(AOTRITON_NS::v3::flash::flyc_varlen_selfcheck(), "flyc_varlen_selfcheck()");

  if (g_failures == 0) {
    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
