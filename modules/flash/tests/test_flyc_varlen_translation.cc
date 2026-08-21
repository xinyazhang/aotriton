// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone test for modules/flash/csrc/flyc_attn_fwd.cc's varlen
// translation -- there is no C++ test harness anywhere in this project (no
// gtest, no CMake add_test target), so this is a plain main() rather than an
// invented framework. It is NOT part of the CMake build: v3src/CMakeLists.txt
// globs "modules/*/csrc/*.cc" recursively, and this file deliberately lives
// under modules/flash/tests/ (no "csrc" path component) so it is never swept
// into libaotriton_v2.so.
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
// Build (see also the header-generation recipe used to verify
// flyc_attn_fwd.cc itself):
//
//   d=/tmp/flycgen is a directory holding a generated flash/flyc.flyc_attn_fwd.h
//   ROCM_INC is the rocm sdk's include directory
//
//   g++ -std=c++20 -D__HIP_PLATFORM_AMD__=1
//       -I . -I include -I build-0.14-shim-gfx1201_gfx950/include
//       -I <d> -I build-0.14-shim-gfx1201_gfx950/v3src -I <ROCM_INC>
//       modules/flash/csrc/flyc_attn_fwd.cc
//       modules/flash/tests/test_flyc_varlen_translation.cc
//       -o /tmp/flyc_varlen_test
//   /tmp/flyc_varlen_test
//
// Note: <d> must come BEFORE build-0.14-shim-gfx1201_gfx950/v3src on the
// include path. Both directories have a flash/flyc.flyc_attn_fwd.h -- the
// v3src copy is a stale, previously-generated one -- so <d>'s freshly
// generated copy has to be found first. flash/iface.op_attn_fwd.h only
// exists under v3src, so that directory still has to be on the path.

#include <aotriton/config.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/flyc.flyc_attn_fwd.h>
#include <flash/iface.op_attn_fwd.h>

#include "../csrc/flyc_varlen.h"

#include <cstdio>
#include <cstdlib>

using AOTRITON_NS::DType;
using AOTRITON_NS::TensorView;
using AOTRITON_NS::v3::flash::FlycAttnFwdContext;
using AOTRITON_NS::v3::flash::OpAttnFwdParams;

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

// Builds a params struct with only the fields the varlen helpers read
// (Q, Num_seqlens, seq_strides_q); everything else is value-initialized
// (null pointers, zero), which is safe because the helpers under test never
// touch them.
OpAttnFwdParams
make_params(const TensorView<4>& q, int32_t num_seqlens, const TensorView<1>& seq_strides_q) {
  OpAttnFwdParams params{};
  params.Q = &q;
  params.Num_seqlens = num_seqlens;
  params.seq_strides_q = &seq_strides_q;
  return params;
}

struct Row {
  const char* name;
  int32_t expect_varlen_bits;
  int32_t expect_batch_size;
  int32_t expect_num_seqlens;
};

// FlycAttnFwdContext has no default constructor -- only the templated
// (ParentContext, bool) one the generated header provides for building a
// sub-context off a caller's operator context (params/call_options copied
// across, see flyc.flyc_attn_fwd.h). This is a minimal stand-in for that
// caller context, supplying just the two fields the template reads.
struct FakeParentContext {
  const OpAttnFwdParams* params;
  const AOTRITON_NS::v3::flash::attn_options* call_options;
};

void
check_row(const char* name, OpAttnFwdParams& params, int32_t expect_bits,
          int32_t expect_batch, int32_t expect_nseq) {
  FakeParentContext parent{&params, nullptr};
  FlycAttnFwdContext ctx(parent, /*condition=*/true);

  const int32_t bits = ctx.flyc_varlen_bits();
  const int32_t batch_size = ctx.flyc_batch_size();
  const int32_t num_seqlens = ctx.flyc_num_seqlens();

  std::fprintf(stderr, "-- %s: varlen_bits=0x%04x batch_size=%d num_seqlens=%d\n",
               name, bits, batch_size, num_seqlens);

  char msg[128];
  std::snprintf(msg, sizeof(msg), "%s: varlen_bits", name);
  check(bits == expect_bits, msg);
  std::snprintf(msg, sizeof(msg), "%s: batch_size", name);
  check(batch_size == expect_batch, msg);
  std::snprintf(msg, sizeof(msg), "%s: num_seqlens", name);
  check(num_seqlens == expect_nseq, msg);
}

} // anonymous namespace

int
main() {
  using AOTRITON_NS::v3::flash::kFlycVarlenCompact;
  using AOTRITON_NS::v3::flash::kFlycVarlenDense;
  using AOTRITON_NS::v3::flash::kFlycVarlenPadded;
  using AOTRITON_NS::v3::flash::kFlycVarlenStrided;

  const TensorView<1> null_seqinfo{}; // base_ == nullptr: "no seq_strides_q tensor"
  TensorView<1> real_seqinfo{intptr_t(0x1000), {8}, {1}, DType::kInt32}; // base_ != nullptr

  // Row 1: dense. Num_seqlens == 0, batch_size == Q->size(0) == B, no packing.
  {
    TensorView<4> q{intptr_t(0x2000), {4, 1, 1, 1}, {1, 1, 1, 1}, DType::kFloat16};
    auto params = make_params(q, /*num_seqlens=*/0, null_seqinfo);
    check_row("dense", params, static_cast<int32_t>(kFlycVarlenDense), /*batch=*/4, /*nseq=*/0);
  }

  // Row 2: compact varlen. Num_seqlens > 0, seq_strides_q is a null tensor.
  // batch_size == Q->size(0) == 1 (packed 1THD), not Batch.
  {
    TensorView<4> q{intptr_t(0x2000), {1, 1, 1, 1}, {1, 1, 1, 1}, DType::kFloat16};
    auto params = make_params(q, /*num_seqlens=*/7, null_seqinfo);
    check_row("compact", params, static_cast<int32_t>(kFlycVarlenCompact), /*batch=*/1, /*nseq=*/7);
  }

  // Row 3: padded varlen. Num_seqlens < 0; batch_size == Q->size(0) == N and
  // must equal abs(Num_seqlens) -- exactly the mandatory assertion's
  // non-trivial case. num_seqlens reported as 0: nothing is "packed", the
  // count already rode in on batch_size.
  {
    TensorView<4> q{intptr_t(0x2000), {5, 1, 1, 1}, {1, 1, 1, 1}, DType::kFloat16};
    auto params = make_params(q, /*num_seqlens=*/-5, null_seqinfo);
    check_row("padded", params, static_cast<int32_t>(kFlycVarlenPadded), /*batch=*/5, /*nseq=*/0);
  }

  // Row 4: strided varlen. Num_seqlens > 0, seq_strides_q is NOT a null
  // tensor -- the only bit distinguishing this row from compact.
  {
    TensorView<4> q{intptr_t(0x2000), {1, 1, 1, 1}, {1, 1, 1, 1}, DType::kFloat16};
    auto params = make_params(q, /*num_seqlens=*/9, real_seqinfo);
    check_row("strided", params, static_cast<int32_t>(kFlycVarlenStrided), /*batch=*/1, /*nseq=*/9);
  }

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
