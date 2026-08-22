// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T10: hipGraph timing, per rev0 §5.1.
//
// T05's hardware question is ANSWERED (gfx942 / ROCm 7.14; see
// probe_graph_timing.cc's header for the numbers): in-graph
// hipEventElapsedTime works, provided the event records opt in with
// hipEventRecordWithFlags(..., hipEventRecordExternal). A DEFAULT-flag
// record during capture is an internal capture dependency marker, not a
// node -- capturing [memset, record, kernel, record] x 100 that way yields
// 200 nodes instead of 400 and every elapsed-time read fails with
// hipErrorInvalidResourceHandle.
//
// With the external flag, timing.cc captures the whole
// [L2-flush memset, record, launch, record] x N loop in a single pass and
// does nothing else to the graph. It is do_bench's loop, captured: a flat
// M + N*N_nodes chain with no nesting and no post-hoc editing. The
// methodology rev0 §5.1 specifies is unchanged.
//
// Verified end to end on gfx942 via T14 (attn_fwd and attn_bwd, every
// backend). The stream_ev fallback below is exercised only via
// PERFMON_TIMING_METHOD -- capture has not failed on any configuration
// tried, so it has never been reached automatically.

#ifndef PERFMON_CORE_TIMING_H
#define PERFMON_CORE_TIMING_H

#include "perfmon_abi.h"
#include "stats.h"
#include "thermal.h"

#include <cstdint>
#include <string>

namespace perfmon {

// N in rev0 §5.1: timed iterations, and hence samples, in BOTH methods.
constexpr int kTimingIters = 100;

// Untimed warmup iterations before the timed pass in the stream_ev
// fallback, mirroring the graph warmup hipgraph_ev100 does.
constexpr int kWarmupIters = 10;

// 1 GiB L2-flush buffer, matching triton.testing.do_bench and
// python/tune/gpu_utils.py:210. Mandatory between EVERY timed iteration,
// in both methods -- not just hipgraph_ev100 -- per T10's spec.
constexpr size_t kL2FlushBytes = size_t(1) << 30;

// gpu_utils.py's own wait_gpu_temperature() default threshold (85.0 C
// junction). Reused here rather than inventing a second project-wide
// default -- see thermal.h/thermal_amdsmi.cc for what backs this on a real
// build; on the stub build (T08 default) this gate is a no-op.
constexpr double kDefaultThermalThresholdC = 85.0;

// One measurement's result, including which timing method actually
// produced it (T10: "The returned record always carries `timing_method`").
struct MeasurementRecord {
  std::string timing_method;  // "hipgraph_ev100" or "stream_ev"
  bool l2_flush = true;       // always true -- see kL2FlushBytes above
  Stats stats;                // n/mean/median/stddev/min/p05/p95 (stats.h)
  pmon_thermal thermal;       // thermal.valid == false -> caller must
                               // serialize this measurement's "thermal" key
                               // as JSON null, never as a zero-reading
                               // struct (thermal.h's pmon_thermal doc).
  uint64_t seed = 0;          // rev0 §5.3: recorded so any number is
                               // reproducible from the entry + this seed.
};

// Times `vtable.launch(ctx, stream)` for kTimingIters iterations on
// `stream`, with the mandatory L2-flush memset between iterations.
//
// Gates on `thermal::wait_until_cool(thermal_threshold_c)` before timing,
// then takes one `thermal_snapshot()` to attach to the record (reflecting
// conditions the measurement was actually taken under, not conditions
// after the fact).
//
// Tries hipgraph_ev100 (capture kTimingIters iterations into a hipGraph,
// instantiate, warm up, launch once, read back all per-iteration
// hipEventElapsedTime values) first. If `hipStreamBeginCapture`/
// `hipStreamEndCapture` fails for this backend -- rev0 §5.1's documented
// possibility -- falls back to stream_ev for THIS CALL ONLY; the
// measurement is never dropped, only its `timing_method` differs.
//
// Both methods now time the identical thing: kTimingIters iterations of
// [1 GiB L2 flush, launch], one independent event pair per iteration, with
// the flush OUTSIDE the pair. They differ only in whether the work is
// dispatched from a graph or straight from the stream, which on gfx942 is
// worth a couple of percent. rev0 D6's "never compare across
// timing_method" still applies, but the gap it guards against is now small
// rather than structural.
//
// `seed` is the caller-derived rev0 §5.3 seed (hash(functional_pon,
// shape_pon, subject, iface)) already used to fill `ctx`'s input buffers;
// it is only carried through into the returned record here, not
// recomputed.
MeasurementRecord measure_launch(const pmon_family_vtable& vtable, void* ctx, hipStream_t stream,
                                  uint64_t seed,
                                  double thermal_threshold_c = kDefaultThermalThresholdC);

}  // namespace perfmon

#endif  // PERFMON_CORE_TIMING_H
