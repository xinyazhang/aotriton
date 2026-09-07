// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone test that FlyDSL's varlen encoding and AOTriton's are one
// encoding -- there is no C++ test harness anywhere in this project (no gtest,
// no CMake add_test target), so this is a plain main() rather than an invented
// framework. It is NOT part of the CMake build: v3src/CMakeLists.txt globs
// "modules/*/csrc/*.cc" recursively, and this file deliberately lives under
// modules/flash/tests/ (no "csrc" path component) so it is never swept into
// libaotriton_v2.so.
//
// WHAT IT NO LONGER TESTS, AND WHY. This file used to drive
// flyc_classify_varlen over the four rows of the varlen table, back when that
// function INFERRED the layout from the sign of a tri-state Num_seqlens plus
// the nullness of seq_strides_q. The public API now carries varlen_bits, so
// there is nothing to infer: the layout word arrives built. The inference
// survives only as the kVersion 3/6 compatibility path, varlen_bits_of() in
// modules/flash/csrc/varlen.h, and modules/flash/tests/test_varlen_translation.cc
// is where it is now tested. Re-testing it here would test upstream's function
// through a wrapper this side no longer has.
//
// What remains is the one thing flyc still owns: that flyc_varlen.h and
// varlen.h describe the same wire format. Most of that check is
// static_assert'd in flyc_varlen.h itself and so has already passed by the time
// this runs -- shifts, axis constants, and the four legacy rows through both
// encoders. This binary covers the piece the language will not let us check at
// compile time, and then re-checks the four rows against the DECODER as well,
// which the static_asserts cannot reach: they compare encoder output, so an
// encode/decode pair that was wrong in the same direction would satisfy them.
//
// Build -- header-only, so no generated per-kernel header and no .cc to
// compile alongside:
//
//   g++ -std=c++20 -D__HIP_PLATFORM_AMD__=1
//       -I . -I include -I <any build dir>/include -I <rocm>/include
//       modules/flash/tests/test_flyc_varlen_translation.cc
//       -o /tmp/flyc_varlen_test
//   /tmp/flyc_varlen_test
//
// The build dir is only needed for aotriton/config.h. HIP headers ARE needed
// now, where they were not before: varlen.h includes <aotriton/flash.h> for
// VarlenBits, which reaches runtime.h. Still no generated per-kernel header and
// no libaotriton_v2.so -- and no logging stubs either, since nothing this file
// touches has an AOTRITON_LOG error path.

#include <aotriton/config.h>

#include "../csrc/flyc_varlen.h"

#include <cstdio>
#include <cstdlib>

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

// One legacy row, taken apart by AOTriton's decoder and compared against the
// axes FlyDSL's named constants were built from. The encoders agreeing is
// static_assert'd; this is the decoder agreeing, on both sides of the word.
void
check_row(const char* name, uint32_t flyc_bits,
          uint32_t stacked, uint32_t length, uint32_t position) {
  namespace flash = AOTRITON_NS::v3::flash;
  const auto v = flash::internal::varlen_from_wire(flyc_bits);

  std::fprintf(stderr, "-- %s: 0x%04x -> q(%u,%u,%u) k(%u,%u,%u) lse=%u\n",
               name, flyc_bits,
               unsigned(v.qmode.stacked), unsigned(v.qmode.length), unsigned(v.qmode.position),
               unsigned(v.kmode.stacked), unsigned(v.kmode.length), unsigned(v.kmode.position),
               unsigned(v.lse_layout));

  char msg[128];
  std::snprintf(msg, sizeof(msg), "%s: q axes", name);
  check(v.qmode.stacked == stacked && v.qmode.length == length
        && v.qmode.position == position, msg);
  // AOTriton cannot express a per-side difference today, so every word it
  // builds sets both sides identically; a K side that decoded differently
  // would mean the K_SIDE shift disagreed between the two headers.
  std::snprintf(msg, sizeof(msg), "%s: k axes match q", name);
  check(v.kmode.stacked == stacked && v.kmode.length == length
        && v.kmode.position == position, msg);
  std::snprintf(msg, sizeof(msg), "%s: no stray reserved bits", name);
  check(v.qmode.reserved == 0 && v.kmode.reserved == 0 && v.reserved == 0
        && v.lse_layout == flash::VarlenLseLayout::HT, msg);
}

} // anonymous namespace

int
main() {
  namespace flash = AOTRITON_NS::v3::flash;

  check_row("dense", flash::kFlycVarlenDense,
            flash::FlycVarlenStacked::BHSD,
            flash::FlycVarlenLength::MAX,
            flash::FlycVarlenPosition::IMPLIED);
  check_row("compact", flash::kFlycVarlenCompact,
            flash::FlycVarlenStacked::T1HD,
            flash::FlycVarlenLength::CUMULATIVE,
            flash::FlycVarlenPosition::REUSE);
  check_row("padded", flash::kFlycVarlenPadded,
            flash::FlycVarlenStacked::BHSD,
            flash::FlycVarlenLength::CUMULATIVE,
            flash::FlycVarlenPosition::IMPLIED);
  check_row("strided", flash::kFlycVarlenStrided,
            flash::FlycVarlenStacked::T1HD,
            flash::FlycVarlenLength::CUMULATIVE,
            flash::FlycVarlenPosition::ARRAY);

  // The check the language will not make at compile time on clang: that
  // flyc_varlen.h's bitfield view decodes what its own shift-based encoder
  // produced. Independent of everything above, which goes through AOTriton's
  // decoder rather than FlyDSL's bitfields.
  check(flash::flyc_varlen_selfcheck(), "flyc_varlen_selfcheck()");

  if (g_failures == 0) {
    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
