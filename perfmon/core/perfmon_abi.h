/* Copyright © 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * perfmon_abi.h -- the plain-C boundary between libperfmon_core
 * (AOTriton-NEUTRAL: depends only on HIP + amd-smi) and
 * libperfmon_flash@<subject> (AOTriton- and family-dependent).
 *
 * perfmon-rev0.md D4, transcribed verbatim for the struct/vtable shape,
 * with `pmon_entry`'s field list filled in per perfmon-exec0.md T06 (D4's
 * prose only says "POD: dtype, hdim, seqlens, ..."; T06 spells out the
 * complete field set required).
 *
 * Rules (D4, T06), non-negotiable:
 *  - extern "C" throughout; no C++ types, no AOTriton headers, no templates.
 *  - No AOTriton type crosses this boundary -- only HIP types (hipStream_t)
 *    and POD C. libperfmon_core must never see an AOTriton header.
 *  - One binary per AOTriton version (D4): this header is owned by
 *    libperfmon_core and must NEVER be edited to add a per-version field.
 *    Per-version divergence lives in adapter_v3.cc / overrides/<tag>.cc,
 *    never here.
 *  - PMON_ABI_VERSION is bumped whenever this header's binary layout
 *    changes, and checked by core at startup against the value the family
 *    library was built against, so a mismatched libperfmon_flash@<subject>
 *    fails loudly instead of silently corrupting a measurement.
 *
 * PMON_ABI_VERSION 2 (T12 follow-up to T06): added `pmon_iface`/
 * `pmon_entry.iface`. Found missing while implementing T12's adapter_v3.cc:
 * `protocol.h` (T07, already committed) documents the wire command shape as
 * `measure <entry_pon> <iface> <backend>` / `enumerate <entry_pon> <iface>`
 * -- the runner's main() decodes `<entry_pon>` plus the separate `<iface>`
 * token into one `pmon_entry` and calls `prepare()`/`enumerate_backends()`
 * with it, so `prepare()` needs to know from the entry alone whether to pack
 * `attn_fwd_params` or `attn_bwd_params`. `rev0 §7`'s own `functional_pon`
 * lists `iface` as a field of its own, a sibling of `dtype`/`causal`/etc, not
 * encoded in any of them (example: `"iface='attn_fwd';dtype='bfloat16';..."`)
 * -- so this was a genuine omission in T06's field list, not a per-family
 * concern (every family's `PerfDesc.list_ifaces()` returns iface names, D8),
 * hence it belongs in the core ABI. This is a disclosed, versioned fix to an
 * already-committed file, not a silent adaptation.
 */

#ifndef PERFMON_CORE_PERFMON_ABI_H
#define PERFMON_CORE_PERFMON_ABI_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any binary-layout change to this header. */
#define PMON_ABI_VERSION 2

/* dtype enum, carried in pmon_entry as a plain int32_t so the struct stays
 * a flat POD with no dependency on any AOTriton or torch dtype enum.
 * Values are perfmon's own and stable only within one PMON_ABI_VERSION. */
enum pmon_dtype {
  PMON_DTYPE_FLOAT16  = 0,
  PMON_DTYPE_BFLOAT16 = 1,
  PMON_DTYPE_FLOAT32  = 2,
};

/* NOTE: deliberately NOT an enum of named ifaces (unlike pmon_dtype above).
 * `iface` selects among the strings a family's PerfDesc.list_ifaces() (D8)
 * returns -- e.g. flash's ['attn_fwd', 'attn_bwd'] -- and that list, and the
 * order within it, is a family concern. Baking iface NAMES into this
 * core-owned, family-neutral header would be exactly the upward knowledge
 * flow D4 exists to prevent (libperfmon_core must never encode an
 * AOTriton/family-specific concept). So `pmon_entry.iface` below is just
 * "index into this entry's family's list_ifaces() order"; core carries it
 * through uninterpreted, and each family's own adapter (e.g. adapter_v3.cc)
 * documents its own mapping (flash: attn_fwd=0, attn_bwd=1) locally. */

/* POD entry description -- everything a family adapter needs to pack an
 * AOTriton params struct and launch a kernel, and nothing else. This is
 * the wire-shaped struct core passes into the family vtable's
 * enumerate_backends/prepare calls; `describe()` (below) returns the
 * separate human/JSON-readable view of whatever backend got selected. */
typedef struct pmon_entry {
  int32_t  iface;           /* index into this family's list_ifaces(), see
                              * the note above pmon_dtype's declaration */
  int32_t  dtype;          /* enum pmon_dtype */
  int32_t  hdim;
  int32_t  seqlen_q;
  int32_t  seqlen_k;
  int32_t  causal;          /* 0 or 1 */
  double   dropout_p;
  int32_t  bias_type;
  int32_t  batch;
  int32_t  n_heads;         /* query heads */
  int32_t  gqa_ratio;       /* n_heads / kv_heads; 1 when not GQA */
  int32_t  varlen;          /* 0 or 1 */
  int32_t  storage_flip;    /* 0 or 1 */
  uint64_t seed;            /* rev0 §5.3 deterministic device-side RNG seed */
} pmon_entry;

/* Per-family vtable, exactly as specified in D4.
 *
 * `launch` is called by core's timing loop (timing.cc) INSIDE a hipGraph
 * capture (rev0 §5.1) -- it must enqueue only stream-ordered HIP work on
 * the given stream and must not synchronize or block.
 */
typedef struct pmon_family_vtable {
  int         (*enumerate_backends)(const pmon_entry* entry, int* out, int max);
  int         (*prepare)(const pmon_entry* entry, int backend, void** ctx);
  int         (*launch)(void* ctx, hipStream_t stream);
  const char* (*describe)(void* ctx);
  void        (*release)(void* ctx);
} pmon_family_vtable;

/* The one symbol every libperfmon_flash@<subject> exports. */
const pmon_family_vtable* pmon_family_entry(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PERFMON_CORE_PERFMON_ABI_H */
