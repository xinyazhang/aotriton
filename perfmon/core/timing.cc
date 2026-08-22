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

// Releases the HIP objects one hipgraph_ev100 attempt owns. Every exit
// path from try_hipgraph_ev100 -- normal return, early `return false`, or
// a check() throw -- runs this, because the runner is one long-lived
// process serving many measurements (rev0 D4) and leaking 2N events plus
// two graphs per failed measure would accumulate.
struct GraphScratch {
  std::vector<hipEvent_t> start_ev;
  std::vector<hipEvent_t> end_ev;
  hipGraph_t child = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  // Destructor cleanup is best-effort: these run on the failure path too,
  // where a HIP error is already being reported, so the [[nodiscard]]
  // status of each destroy is deliberately discarded rather than allowed
  // to mask the original error.
  ~GraphScratch() {
    if (graph_exec) static_cast<void>(hipGraphExecDestroy(graph_exec));
    if (graph) static_cast<void>(hipGraphDestroy(graph));
    if (child) static_cast<void>(hipGraphDestroy(child));
    for (auto& e : start_ev) if (e) static_cast<void>(hipEventDestroy(e));
    for (auto& e : end_ev) if (e) static_cast<void>(hipEventDestroy(e));
  }
};

// Primary method (rev0 §5.1 `hipgraph_ev100`): kTimingIters iterations of
// [L2-flush memset, event record, family launch, event record] in one
// hipGraph -- instantiate, warm up, launch once, read back every
// per-iteration hipEventElapsedTime.
//
// The graph is ASSEMBLED EXPLICITLY (hipGraphAddMemsetNode /
// hipGraphAddEventRecordNode / hipGraphAddChildGraphNode) rather than
// captured whole, because T05's hardware probe found that HIP's stream
// capture SILENTLY DISCARDS hipEventRecord: capturing
// [record, kernel, record] yields a graph whose hipGraphGetNodes count is
// 1, not 3, and every subsequent hipEventElapsedTime on those events fails
// with hipErrorInvalidResourceHandle because they were never recorded.
// Measured on gfx942 / ROCm 7.14. Built explicitly instead, the same
// events read back correctly and agree with conventional (non-graph)
// timing to 0.05%.
//
// Stream capture is still used, but only for the ONE opaque
// `vtable.launch(ctx, stream)` -- the family call is an AOTriton entry
// point whose kernel launches perfmon cannot enumerate as explicit nodes
// (rev0 D4 keeps this library AOTriton-neutral). That single-launch
// capture becomes a child-graph node instantiated kTimingIters times.
//
// Events are 2N INDEPENDENT objects (start_ev[i], end_ev[i]), never a
// shared ev[i+1] straddling adjacent iterations: a shared end/start event
// is recorded twice per graph, and its surviving timestamp is the LATER
// record, which silently folds the next iteration's 1 GiB flush memset
// into the sample.
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
  GraphScratch scratch;

  // --- 1. Capture ONE family launch into a child graph -------------------
  // No hipEventRecord inside this capture: see the note above on capture
  // dropping event nodes. Only the family's own kernel launches go here.
  hipError_t begin_err = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  if (begin_err != hipSuccess) {
    return false;  // documented fallback trigger
  }
  int launch_rc = vtable.launch(ctx, stream);
  // End the capture unconditionally, even when the launch failed, so the
  // stream is never left in capturing state for the batched_ev fallback.
  hipError_t end_err = hipStreamEndCapture(stream, &scratch.child);
  check_launch(launch_rc, "vtable.launch (child-graph capture)");
  if (end_err != hipSuccess) {
    return false;  // documented fallback trigger
  }

  size_t child_nodes = 0;
  check(hipGraphGetNodes(scratch.child, nullptr, &child_nodes), "hipGraphGetNodes (child)");
  if (child_nodes == 0) {
    throw std::runtime_error(
      "timing: vtable.launch enqueued no work on the captured stream -- the "
      "family adapter must launch on the stream it is given, never the null "
      "stream (rev0 D4/T12).");
  }

  // --- 2. Assemble the timed graph explicitly ---------------------------
  check(hipGraphCreate(&scratch.graph, 0), "hipGraphCreate");

  scratch.start_ev.assign(kTimingIters, nullptr);
  scratch.end_ev.assign(kTimingIters, nullptr);
  for (int i = 0; i < kTimingIters; ++i) {
    check(hipEventCreateWithFlags(&scratch.start_ev[i], hipEventDefault), "hipEventCreateWithFlags");
    check(hipEventCreateWithFlags(&scratch.end_ev[i], hipEventDefault), "hipEventCreateWithFlags");
  }

  hipMemsetParams flush_params = {};
  flush_params.dst = flush_buf;
  flush_params.value = 0;
  flush_params.elementSize = 1;
  flush_params.width = kL2FlushBytes;
  flush_params.height = 1;
  flush_params.pitch = 0;

  // A single linear dependency chain: every iteration's flush waits on the
  // previous iteration's end-record, so the iterations run strictly in
  // sequence and each timed window contains exactly one launch.
  hipGraphNode_t prev = nullptr;
  for (int i = 0; i < kTimingIters; ++i) {
    hipGraphNode_t flush_node = nullptr;
    hipGraphNode_t start_node = nullptr;
    hipGraphNode_t child_node = nullptr;
    hipGraphNode_t end_node = nullptr;
    const hipGraphNode_t* deps = prev ? &prev : nullptr;
    size_t ndeps = prev ? 1 : 0;

    check(hipGraphAddMemsetNode(&flush_node, scratch.graph, deps, ndeps, &flush_params),
          "hipGraphAddMemsetNode (L2 flush)");
    check(hipGraphAddEventRecordNode(&start_node, scratch.graph, &flush_node, 1,
                                     scratch.start_ev[i]),
          "hipGraphAddEventRecordNode (start)");
    check(hipGraphAddChildGraphNode(&child_node, scratch.graph, &start_node, 1, scratch.child),
          "hipGraphAddChildGraphNode (family launch)");
    check(hipGraphAddEventRecordNode(&end_node, scratch.graph, &child_node, 1, scratch.end_ev[i]),
          "hipGraphAddEventRecordNode (end)");
    prev = end_node;
  }

  check(hipGraphInstantiate(&scratch.graph_exec, scratch.graph, nullptr, nullptr, 0),
        "hipGraphInstantiate");

  // Warm up the instantiated graph before the timed launch (T10 spec).
  check(hipGraphLaunch(scratch.graph_exec, stream), "hipGraphLaunch (warmup)");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warmup)");

  check(hipGraphLaunch(scratch.graph_exec, stream), "hipGraphLaunch (timed)");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (timed)");

  out_ms->resize(kTimingIters);
  for (int i = 0; i < kTimingIters; ++i) {
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, scratch.start_ev[i], scratch.end_ev[i]),
          "hipEventElapsedTime");
    (*out_ms)[i] = static_cast<double>(ms);
  }

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
