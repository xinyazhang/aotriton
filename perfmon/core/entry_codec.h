// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T13 gap-fill: decodes the wire `<entry_pon>` text (as
// produced by `modules/flash/perfmon/__init__.py`'s `PerfDesc.functional_pon`
// + `.shape_pon`, both `render_pon`-encoded, `python/utils/pon.py`) into the
// field set `perfmon_abi.h`'s `pmon_entry` needs.
//
// WHY THIS FILE EXISTS (a genuine gap found while implementing T13, not
// something any earlier task's spec text assigned): T12/T13 both require a
// `perfmon/subjects/<id>/bin/runner` executable that decodes
// `protocol.h`'s wire commands (`measure <entry_pon> <iface> <backend>`,
// `enumerate <entry_pon> <iface>`) and calls the linked family vtable, but
// no earlier task ever specified the code that turns the wire's PON text
// into a `pmon_entry`. Grepped every already-committed file and
// `perfmon-exec0.md`/`perfmon-rev0.md` in full: no such logic, and no task
// assigns it. Written now, as part of T13 (the task that actually needs a
// working runner binary), with every judgment call disclosed below and in
// entry_codec.cc.
//
// Deliberately HIP-free (no perfmon_abi.h, no <hip/hip_runtime.h>) so this
// stays g++-testable on a machine with no ROCm install, same rationale as
// protocol.h/stats.h (T07)'s own header comments. `PmonEntryFields` below
// mirrors `pmon_entry` (perfmon_abi.h) field-for-field and MUST be kept in
// sync with it by hand -- there is no single source of truth shared between
// the two because perfmon_abi.h's own top comment forbids it from including
// anything but <hip/hip_runtime.h> and <stdint.h>. main.cc does the trivial
// memberwise copy into the real `pmon_entry` (a HIP-touching translation
// unit, so IT is one of this project's usual "written but not compiled
// here" files; this header/its .cc are not).
//
// --- Disclosed design decisions (none of these are dictated by any task's
// spec text; each is a first-cut choice made to produce a working runner) --
//
// 1. `iface` is NOT read from the PON text's own `iface='attn_fwd'` key.
//    That key is human/Python-side bookkeeping (rev0 §7's on-disk format);
//    turning a family's iface NAME back into an index would need a
//    family-specific name table, which would make this HIP-free, family-
//    NEUTRAL parser family-specific -- exactly what D4 exists to prevent.
//    Instead, `iface` is supplied by the caller as the already-resolved
//    integer index protocol.h's wire shape carries as its OWN token
//    (`<iface>`, separate from `<entry_pon>`). This is a real, disclosed
//    requirement on whatever later task builds the wire commands (a
//    `runner_proxy.py`, out of this task's scope): it must send `<iface>`
//    as a decimal integer (0=attn_fwd, 1=attn_bwd for flash, matching
//    `PerfDesc.list_ifaces()` order), not the iface name.
//
// 2. GQA-tuple wire-tokenization hazard: `shape_pon`'s `N_HEADS` field can
//    be a Python tuple, e.g. `N_HEADS=(10, 2)` -- `repr()` of a tuple always
//    inserts ", " (comma-SPACE) between elements. protocol.h's wire commands
//    are whitespace-tokenized (`parse_command`), so this embedded space
//    would fragment a single `<entry_pon>` token into two. This is a real
//    bug in the wire format as specified across already-committed files
//    (T02's `functional_pon`/`shape_pon` split, T07's whitespace-tokenized
//    protocol) that this task's own investigation found -- reported here,
//    not silently special-cased away. `main.cc` works around it locally
//    (rejoining any extra whitespace-split fragments with a single space
//    before calling `parse_entry_pon`, exploiting the fact that a single
//    embedded ", " is the only source of extra whitespace `render_pon` ever
//    produces for this field set) rather than changing the already-shipped
//    wire protocol or Python PON renderer.
//
// 3. `seed`: rev0 §5.3 says the caller sets `pmon_entry.seed` to
//    `hash(functional_pon, shape_pon, subject, iface)`, but neither
//    `functional_pon`/`shape_pon` (T02, already committed) nor the wire
//    protocol (T07) actually carries a seed value end to end -- nothing
//    upstream computes or transmits one. `parse_entry_pon` therefore derives
//    it itself, deterministically, from the exact strings it already has
//    (`entry_pon`, `iface`) via a plain FNV-1a 64-bit hash.
//    This satisfies every property any already-completed task's Verify step
//    actually checks (T09: "two runs with the same seed are bit-identical"
//    -- trivially true for a pure function of its inputs) without inventing
//    a new cross-language hash contract nothing currently needs to match.

#ifndef PERFMON_CORE_ENTRY_CODEC_H
#define PERFMON_CORE_ENTRY_CODEC_H

#include <cstdint>
#include <string>

namespace perfmon {

// Field-for-field mirror of perfmon_abi.h's `pmon_entry` -- see this file's
// header comment for why it is a separate, HIP-free type rather than the
// real one.
struct PmonEntryFields {
  int32_t  iface = 0;
  int32_t  dtype = 0;
  int32_t  hdim = 0;
  int32_t  seqlen_q = 0;
  int32_t  seqlen_k = 0;
  int32_t  causal = 0;
  double   dropout_p = 0.0;
  int32_t  bias_type = 0;
  int32_t  batch = 0;
  int32_t  n_heads = 0;
  int32_t  gqa_ratio = 1;
  int32_t  varlen = 0;
  int32_t  storage_flip = 0;
  uint64_t seed = 0;
};

// Parses `entry_pon` (the reassembled, single-space-rejoined
// `functional_pon` + ';' + `shape_pon` text -- see this header's item 2
// above) into a `PmonEntryFields`. `iface` is the already-resolved integer
// index (item 1 above), and also feeds the seed derivation (item 3).
//
// Throws std::invalid_argument, with a message naming the offending key,
// on any missing/malformed field -- never returns a partially-filled
// struct silently.
PmonEntryFields parse_entry_pon(const std::string& entry_pon, int32_t iface);

// dtype name ("float16"/"bfloat16"/"float32") -> perfmon_abi.h's
// `pmon_dtype` enum value. Exposed separately (not just used internally)
// so a unit test can check it against the literal enum values without
// including the HIP-dependent perfmon_abi.h.
int32_t dtype_name_to_pmon_dtype(const std::string& name);

// FNV-1a 64-bit over `s`. Exposed for the seed-determinism unit test (item
// 3 above); not claimed to match any hash function outside this file.
uint64_t fnv1a64(const std::string& s);

}  // namespace perfmon

#endif  // PERFMON_CORE_ENTRY_CODEC_H
