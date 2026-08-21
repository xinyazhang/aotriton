// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_MODULES_FLASH_CSRC_FLYC_VARLEN_H
#define AOTRITON_MODULES_FLASH_CSRC_FLYC_VARLEN_H

// FlyDSL's `varlen_bits` layout descriptor, as C++ bitfields.
//
// The canonical specification is FlyDSL's
// `kernels/attention/parity/sdpa-varlen-plan.md` (§1 for the axes, §2 for the
// encoding). This header is the C++ transcription of it, and exists because the
// AOTriton side has to BUILD these words while the kernel side only decodes
// them.
//
// Why name the fields at all rather than paste the four hex constants the spec
// tabulates: the hex is a *result*, not the definition. `0x0B0B` says nothing
// about why compact varlen differs from padded, whereas `stacked=1,
// length=CUMULATIVE, position=REUSE` against `stacked=0, length=CUMULATIVE,
// position=IMPLIED` says exactly what differs and what does not. Written as
// fields, a wrong translation looks wrong; written as hex, every value looks
// equally arbitrary. FlyDSL itself cannot use bitfields -- it is traced Python,
// which has no such notion, so its decoder shifts and masks.
//
// TWO REPRESENTATIONS, AND WHY.
//
// `FlycVarlenBits` is the bitfield view, for reading a word apart: logging, a
// debugger, an assertion. `flyc_varlen_bits()` is the ENCODER, and it shifts
// rather than filling in that struct, for one reason: clang rejects
//
//     constexpr bit_cast involving bit-field is not yet supported
//
// so a bitfield-built constant cannot be `constexpr` on the compiler half this
// tree is built with. GCC accepts it; relying on that would make the header
// compile in one build and not the other.
//
// That leaves two spellings of one layout, which is normally the thing to
// avoid. They are pinned together instead of trusted: the encoder's output is
// `static_assert`ed against the spec's own hex below, and
// `flyc_varlen_selfcheck()` asserts the bitfield view decodes every one of
// those constants back to the fields that produced it. The first check is
// compile-time; the second cannot be, on clang, so it is called from the unit
// test. Neither representation is allowed to drift silently.
//
// Only the HOST builds these. Nothing here is shared with device code.

#include <aotriton/config.h>

#include <bit>
#include <cstdint>

namespace AOTRITON_NS::v3::flash {

// --- axis values (sdpa-varlen-plan.md §1) ------------------------------------
//
// Struct-of-constants rather than `enum class`, matching the house idiom in
// include/aotriton/flash.h (CausalType, VarlenType, WindowValue) and for the
// reason recorded there: an enum class will not convert to its underlying type
// without a cast, and these are assigned straight into bitfields.

// A. Is the token axis stacked?
struct FlycVarlenStacked {
  static constexpr uint32_t BHSD = 0;   // rank-4, sequence selected by batch index
  static constexpr uint32_t T1HD = 1;   // 1THD, sequence selected by a row offset
};

// B. How is the length of sequence z given?
struct FlycVarlenLength {
  static constexpr uint32_t MAX        = 0;   // every sequence is Max_seqlen
  static constexpr uint32_t CUMULATIVE = 1;   // a[z+1] - a[z], array is (N+1,)
  static constexpr uint32_t INDIVIDUAL = 2;   // a[z], array is (N,)
};

// C. Where does sequence z start along the token axis?
struct FlycVarlenPosition {
  static constexpr uint32_t IMPLIED = 0;  // 0 if BHSD, else z * Max_seqlen
  static constexpr uint32_t REUSE   = 1;  // the length array, already loaded
  static constexpr uint32_t ARRAY   = 2;  // a second, position-only array
};

// LSE memory arrangement (§3.2). HT is AOTriton's and the default; TH is
// Transformer Engine's. AOTriton never asks for TH, so every word this header
// builds leaves this zero -- the field is transcribed because the spec reserves
// two bits for it, and a future caller wanting TH should find it here rather
// than inventing a second encoding.
struct FlycVarlenLseLayout {
  static constexpr uint32_t HT = 0;   // (H, T), offset (b*H + h)*S + s
  static constexpr uint32_t TH = 1;   // (T, H), offset (b*S + s)*H + h
};

// --- the word itself (sdpa-varlen-plan.md §2) --------------------------------

// The decoding view. A plain struct rather than a union with a `uint32_t raw`:
// reading a union's inactive member is undefined, so `std::bit_cast` is the
// legal spelling of the same reinterpretation, and it works fine at runtime on
// both compilers -- it is only `constexpr` bit_cast that clang has not
// implemented.
struct FlycVarlenBits {
  // Q side, bits 7:0
  uint32_t q_stacked  : 1;
  uint32_t q_length   : 2;
  uint32_t q_position : 2;
  uint32_t q_reserved : 3;
  // K side, bits 15:8 -- the same byte layout, so the kernel has ONE decoder
  // called twice. AOTriton cannot express a per-side difference today, so every
  // word built here sets both sides identically; the fields stay separate
  // because the ENCODING distinguishes them and collapsing that would have to
  // be undone the day a caller does.
  uint32_t k_stacked  : 1;
  uint32_t k_length   : 2;
  uint32_t k_position : 2;
  uint32_t k_reserved : 3;
  uint32_t lse_layout : 2;   // bits 17:16
  uint32_t reserved18 : 6;   // bits 23:18
  uint32_t reserved24 : 8;   // bits 31:24, reserved for paged KV (§9.1)
};

static_assert(sizeof(FlycVarlenBits) == 4, "varlen_bits is a u32 on the wire");

// Bit positions, from §2. The single place the layout is stated as numbers; the
// bitfield struct above states the same thing as declaration order, and
// `flyc_varlen_selfcheck()` is what stops the two disagreeing.
struct FlycVarlenShift {
  static constexpr uint32_t STACKED    = 0;   // within a side byte
  static constexpr uint32_t LENGTH     = 1;
  static constexpr uint32_t POSITION   = 3;
  static constexpr uint32_t K_SIDE     = 8;   // K's byte, relative to Q's
  static constexpr uint32_t LSE_LAYOUT = 16;
};

// One side's byte.
constexpr uint32_t
flyc_varlen_side(uint32_t stacked, uint32_t length, uint32_t position) {
  return (stacked  << FlycVarlenShift::STACKED)
       | (length   << FlycVarlenShift::LENGTH)
       | (position << FlycVarlenShift::POSITION);
}

// One side's axis triple, applied to both sides. Every configuration AOTriton
// can express is symmetric (§1.3), so this is the only builder needed; a
// per-side variant should be added the day something needs it, not before.
constexpr uint32_t
flyc_varlen_bits(uint32_t stacked, uint32_t length, uint32_t position,
                 uint32_t lse_layout = FlycVarlenLseLayout::HT) {
  const uint32_t side = flyc_varlen_side(stacked, length, position);
  return side
       | (side << FlycVarlenShift::K_SIDE)
       | (lse_layout << FlycVarlenShift::LSE_LAYOUT);
}

// --- the four configurations AOTriton can produce ----------------------------
//
// Named for AOTriton's VarlenType, which is what the caller was thinking in,
// even though that enum has already been erased by the time the shim runs (see
// modules/flash/csrc/attn_fwd.cc, where it collapses into the sign of
// Num_seqlens plus the nullness of seq_strides_q/k).

constexpr uint32_t kFlycVarlenDense = flyc_varlen_bits(
    FlycVarlenStacked::BHSD, FlycVarlenLength::MAX, FlycVarlenPosition::IMPLIED);

constexpr uint32_t kFlycVarlenCompact = flyc_varlen_bits(
    FlycVarlenStacked::T1HD, FlycVarlenLength::CUMULATIVE, FlycVarlenPosition::REUSE);

constexpr uint32_t kFlycVarlenPadded = flyc_varlen_bits(
    FlycVarlenStacked::BHSD, FlycVarlenLength::CUMULATIVE, FlycVarlenPosition::IMPLIED);

constexpr uint32_t kFlycVarlenStrided = flyc_varlen_bits(
    FlycVarlenStacked::T1HD, FlycVarlenLength::CUMULATIVE, FlycVarlenPosition::ARRAY);

// The wire values from sdpa-varlen-plan.md §2's table. These asserts are the
// point of the file: they pin an implementation-defined bit allocation to the
// words the kernel actually decodes, so a compiler that packed the other way
// fails the build instead of producing a plausible wrong address.
static_assert(kFlycVarlenDense   == 0x0000u, "dense must encode as 0x0000");
static_assert(kFlycVarlenCompact == 0x0B0Bu, "compact varlen must encode as 0x0B0B");
static_assert(kFlycVarlenPadded  == 0x0202u, "padded varlen must encode as 0x0202");
static_assert(kFlycVarlenStrided == 0x1313u, "strided varlen must encode as 0x1313");

// The check the language will not let us make at compile time: that the
// bitfield view decodes what the shift encoder produced. Returns true, or
// aborts via assert. Call it once from the unit test -- if a compiler ever
// allocates bitfields in the other order, this is what says so, rather than the
// kernel reading a plausible wrong layout.
inline bool flyc_varlen_selfcheck() {
  struct Case { uint32_t raw, stacked, length, position; };
  const Case cases[] = {
    {kFlycVarlenDense,   FlycVarlenStacked::BHSD,
     FlycVarlenLength::MAX,        FlycVarlenPosition::IMPLIED},
    {kFlycVarlenCompact, FlycVarlenStacked::T1HD,
     FlycVarlenLength::CUMULATIVE, FlycVarlenPosition::REUSE},
    {kFlycVarlenPadded,  FlycVarlenStacked::BHSD,
     FlycVarlenLength::CUMULATIVE, FlycVarlenPosition::IMPLIED},
    {kFlycVarlenStrided, FlycVarlenStacked::T1HD,
     FlycVarlenLength::CUMULATIVE, FlycVarlenPosition::ARRAY},
  };
  for (const auto& c : cases) {
    const auto b = std::bit_cast<FlycVarlenBits>(c.raw);
    if (b.q_stacked  != c.stacked  || b.k_stacked  != c.stacked)  return false;
    if (b.q_length   != c.length   || b.k_length   != c.length)   return false;
    if (b.q_position != c.position || b.k_position != c.position) return false;
    if (b.lse_layout != FlycVarlenLseLayout::HT)                  return false;
    if (b.q_reserved || b.k_reserved || b.reserved18 || b.reserved24) return false;
  }
  return true;
}

}  // namespace AOTRITON_NS::v3::flash

#endif
