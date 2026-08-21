// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "timing.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace perfmon {

namespace {

// Reused across calls in this process rather than allocated per
// measurement -- perfmon's runner is one process per subject per GPU
// (rev0 D4), living for many measurements, so paying hipMalloc's cost only
// once is the natural lifetime for this buffer.
void* flush_buffer() {
  static void* buf = [] {
    void* p = nullptr;
    hipError_t err = hipMalloc(&p, kL2FlushBytes);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("timing: hipMalloc(flush buffer) failed: ") +
                                hipGetErrorString(err));
    }
    return p;
  }();
  return buf;
}

void check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("timing: ") + what + " failed: " +
                              hipGetErrorString(err));
  }
}

void check_launch(int rc, const char* what) {
  if (rc != 0) {
    throw std::runtime_error(std::string("timing: ") + what +
                              " returned nonzero: " + std::to_string(rc));
  }
}

// Primary method (rev0 §5.1 `hipgraph_ev100`): capture kTimingIters
// iterations of [L2-flush memset, event record, family launch, event
// record] into one hipGraph, instantiate, warm up, launch once,
// read back every per-iteration hipEventElapsedTime.
//
// Returns false (leaving `out_ms` untouched) if hipGraph capture itself
// fails -- the caller falls back to batched_ev in that case, per T10's
// spec ("If capture fails for a given backend, fall back to batched_ev for
// that backend and record which was used. Never drop the measurement.").
// Any OTHER HIP error (instantiate/launch/event-read, all of which are
// unexpected once capture itself succeeded) is a hard failure via
// exceptions, not a silent fallback -- rev0 §5.1 frames capture failure
// specifically, not arbitrary downstream HIP errors, as the fallback
// trigger.
bool try_hipgraph_ev100(const pmon_family_vtable& vtable, void* ctx, hipStream_t stream,
                         std::vector<double>* out_ms) {
  void* flush_buf = flush_buffer();

  std::vector<hipEvent_t> ev(kTimingIters + 1);
  for (auto& e : ev) check(hipEventCreate(&e), "hipEventCreate");

  hipError_t begin_err = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  check(begin_err, "hipStreamBeginCapture");

  for (int i = 0; i < kTimingIters; ++i) {
    check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
    check(hipEventRecord(ev[i], stream), "hipEventRecord");
    check_launch(vtable.launch(ctx, stream), "vtable.launch (in-graph)");
    check(hipEventRecord(ev[i + 1], stream), "hipEventRecord");
  }

  hipGraph_t graph = nullptr;
  hipError_t end_err = hipStreamEndCapture(stream, &graph);
  if (end_err != hipSuccess) {
    // Capture failed -- this is exactly the documented fallback trigger.
    // Best-effort cleanup of what we created before returning false.
    for (auto& e : ev) hipEventDestroy(e);
    return false;
  }

  hipGraphExec_t graph_exec = nullptr;
  check(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "hipGraphInstantiate");

  // Warm up the instantiated graph before the timed launch (T10 spec).
  check(hipGraphLaunch(graph_exec, stream), "hipGraphLaunch (warmup)");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warmup)");

  check(hipGraphLaunch(graph_exec, stream), "hipGraphLaunch (timed)");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (timed)");

  out_ms->resize(kTimingIters);
  for (int i = 0; i < kTimingIters; ++i) {
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, ev[i], ev[i + 1]), "hipEventElapsedTime");
    (*out_ms)[i] = static_cast<double>(ms);
  }

  check(hipGraphExecDestroy(graph_exec), "hipGraphExecDestroy");
  check(hipGraphDestroy(graph), "hipGraphDestroy");
  for (auto& e : ev) check(hipEventDestroy(e), "hipEventDestroy");

  return true;
}

// Fallback method (rev0 §5.1 `batched_ev`): kBatchedEvBatches batches,
// each running kTimingIters conventionally-issued (non-graph) iterations
// of [L2-flush memset, family launch] bracketed by one event pair per
// batch; the batch's per-iteration mean (elapsed_ms / kTimingIters) is the
// one sample that batch contributes. `out_ms` ends up holding
// kBatchedEvBatches values, not kTimingIters*kBatchedEvBatches -- matching
// T10's "per-batch mean as the sample" wording exactly.
void run_batched_ev(const pmon_family_vtable& vtable, void* ctx, hipStream_t stream,
                     std::vector<double>* out_ms) {
  void* flush_buf = flush_buffer();

  hipEvent_t start_ev, end_ev;
  check(hipEventCreate(&start_ev), "hipEventCreate");
  check(hipEventCreate(&end_ev), "hipEventCreate");

  // Warm up once, conventionally, before the first timed batch.
  for (int i = 0; i < kTimingIters; ++i) {
    check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
    check_launch(vtable.launch(ctx, stream), "vtable.launch (warmup)");
  }
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warmup)");

  out_ms->resize(kBatchedEvBatches);
  for (int b = 0; b < kBatchedEvBatches; ++b) {
    check(hipEventRecord(start_ev, stream), "hipEventRecord (batch start)");
    for (int i = 0; i < kTimingIters; ++i) {
      check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
      check_launch(vtable.launch(ctx, stream), "vtable.launch (batched)");
    }
    check(hipEventRecord(end_ev, stream), "hipEventRecord (batch end)");
    check(hipStreamSynchronize(stream), "hipStreamSynchronize (batch)");

    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start_ev, end_ev), "hipEventElapsedTime (batch)");
    (*out_ms)[b] = static_cast<double>(elapsed_ms) / static_cast<double>(kTimingIters);
  }

  check(hipEventDestroy(start_ev), "hipEventDestroy");
  check(hipEventDestroy(end_ev), "hipEventDestroy");
}

}  // namespace

MeasurementRecord measure_launch(const pmon_family_vtable& vtable, void* ctx, hipStream_t stream,
                                  uint64_t seed, double thermal_threshold_c) {
  // Thermal gate first -- on the stub build (T08 default) this returns
  // true immediately; on the amd-smi build it blocks until safe. Either
  // way, take the snapshot right after gating so it reflects the
  // conditions the measurement is actually taken under.
  wait_until_cool(thermal_threshold_c);
  pmon_thermal thermal = thermal_snapshot();

  std::vector<double> samples_ms;
  std::string method;
  if (try_hipgraph_ev100(vtable, ctx, stream, &samples_ms)) {
    method = "hipgraph_ev100";
  } else {
    run_batched_ev(vtable, ctx, stream, &samples_ms);
    method = "batched_ev";
  }

  MeasurementRecord record;
  record.timing_method = method;
  record.l2_flush = true;
  record.stats = compute_stats(samples_ms);
  record.thermal = thermal;
  record.seed = seed;
  return record;
}

}  // namespace perfmon
