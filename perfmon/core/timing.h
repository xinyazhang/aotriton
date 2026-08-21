// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T10: hipGraph capture timing, per rev0 §5.1.
//
// UNVERIFIED IN THIS ENVIRONMENT (no ROCm/hipcc/HIP, no GPU -- confirmed in
// T05's commit). Written strictly to spec, and depends directly on T05's
// still-outstanding hardware result: T05's probe_graph_timing.cc exists to
// answer whether hipgraph_ev100 (this file's primary method) is even
// viable on the target ROCm/GPU combination. If T05 shows in-graph
// hipEventElapsedTime is unreliable, measure_launch's automatic
// hipgraph_ev100 -> batched_ev fallback (below) is the documented escape
// hatch, but this has not been exercised on real hardware either.

#ifndef PERFMON_CORE_TIMING_H
#define PERFMON_CORE_TIMING_H

#include "perfmon_abi.h"
#include "stats.h"
#include "thermal.h"

#include <cstdint>
#include <string>

namespace perfmon {

// N in rev0 §5.1: iterations captured/timed per hipgraph_ev100 sample, and
// per batch in batched_ev.
constexpr int kTimingIters = 100;

// K in rev0 §5.1's batched_ev fallback: number of batches, each yielding
// one per-batch-mean sample. compute_stats() then runs over these K means,
// not over the underlying K*N individual iterations.
constexpr int kBatchedEvBatches = 20;

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
  std::string timing_method;  // "hipgraph_ev100" or "batched_ev"
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
// possibility -- falls back to batched_ev (kBatchedEvBatches batches of
// kTimingIters conventionally-timed iterations each, per-batch mean as one
// sample) for THIS CALL ONLY; the measurement is never dropped, only its
// `timing_method` differs.
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
