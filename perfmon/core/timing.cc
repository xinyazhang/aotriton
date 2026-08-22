// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "timing.h"

#include <cstdlib>
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
// a graph per failed measure would accumulate.
struct GraphScratch {
  std::vector<hipEvent_t> start_ev;
  std::vector<hipEvent_t> end_ev;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  // Destructor cleanup is best-effort: these run on the failure path too,
  // where a HIP error is already being reported, so the [[nodiscard]]
  // status of each destroy is deliberately discarded rather than allowed
  // to mask the original error.
  ~GraphScratch() {
    if (graph_exec) static_cast<void>(hipGraphExecDestroy(graph_exec));
    if (graph) static_cast<void>(hipGraphDestroy(graph));
    for (auto& e : start_ev) if (e) static_cast<void>(hipEventDestroy(e));
    for (auto& e : end_ev) if (e) static_cast<void>(hipEventDestroy(e));
  }
};

// Primary method (rev0 §5.1 `hipgraph_ev100`): kTimingIters iterations of
// [L2-flush memset, event record, family launch, event record] in one
// hipGraph -- instantiate, warm up, launch once, read back every
// per-iteration hipEventElapsedTime.
//
// Captured in ONE pass, events included. T05 originally concluded that
// stream capture "silently discards hipEventRecord", because capturing
// [record, kernel, record] produced a 1-node graph and every later
// hipEventElapsedTime failed with hipErrorInvalidResourceHandle. That was
// the right observation about the wrong API: a DEFAULT-flag record during
// capture is an internal capture dependency marker (what cross-stream
// fork/join is built from), and is deliberately not materialised as a node.
//
// hipEventRecordWithFlags(ev, stream, hipEventRecordExternal) is the opt-in
// -- "Event is captured in the graph as an external event node when
// performing stream capture" (hip_runtime_api.h). With it, capturing
// [memset, record, kernel, record] x 100 yields all 400 nodes and every
// event reads back. Measured on gfx942 / ROCm 7.14: 0.02143 ms in-graph vs
// 0.02107 ms conventional.
//
// So the timed graph is just do_bench's loop, captured: a flat chain of
// M + N*kTimingIters nodes, nothing nested, no post-hoc graph surgery, and
// no need for perfmon to understand any node type the family emits.
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

  scratch.start_ev.assign(kTimingIters, nullptr);
  scratch.end_ev.assign(kTimingIters, nullptr);
  for (int i = 0; i < kTimingIters; ++i) {
    check(hipEventCreateWithFlags(&scratch.start_ev[i], hipEventDefault),
          "hipEventCreateWithFlags");
    check(hipEventCreateWithFlags(&scratch.end_ev[i], hipEventDefault),
          "hipEventCreateWithFlags");
  }

  hipError_t begin_err = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  if (begin_err != hipSuccess) {
    return false;  // documented fallback trigger
  }

  hipError_t rec_err = hipSuccess;
  int launch_rc = 0;
  for (int i = 0; i < kTimingIters; ++i) {
    rec_err = hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream);
    if (rec_err != hipSuccess) break;
    rec_err = hipEventRecordWithFlags(scratch.start_ev[i], stream, hipEventRecordExternal);
    if (rec_err != hipSuccess) break;
    launch_rc = vtable.launch(ctx, stream);
    if (launch_rc != 0) break;
    rec_err = hipEventRecordWithFlags(scratch.end_ev[i], stream, hipEventRecordExternal);
    if (rec_err != hipSuccess) break;
  }
  // End the capture unconditionally, even after a failure above, so the
  // stream is never left capturing for the batched_ev fallback.
  hipError_t end_err = hipStreamEndCapture(stream, &scratch.graph);
  check_launch(launch_rc, "vtable.launch (timed-loop capture)");
  check(rec_err, "capturing the timed loop");
  if (end_err != hipSuccess) {
    return false;  // documented fallback trigger
  }

  // The graph must contain the 2N external event-record nodes just asked
  // for. A HIP that accepted hipEventRecordExternal but did not materialise
  // the nodes would otherwise surface as a confusing elapsed-time failure
  // further down; and a launch that enqueued nothing at all (the adapter
  // using the null stream instead of the one it was given, rev0 D4/T12)
  // shows up here as a graph with only the flush and event nodes.
  size_t num_nodes = 0;
  check(hipGraphGetNodes(scratch.graph, nullptr, &num_nodes), "hipGraphGetNodes");
  std::vector<hipGraphNode_t> nodes(num_nodes);
  check(hipGraphGetNodes(scratch.graph, nodes.data(), &num_nodes), "hipGraphGetNodes (fill)");
  size_t event_nodes = 0;
  for (auto n : nodes) {
    hipGraphNodeType t;
    check(hipGraphNodeGetType(n, &t), "hipGraphNodeGetType");
    if (t == hipGraphNodeTypeEventRecord) ++event_nodes;
  }
  if (event_nodes != static_cast<size_t>(2 * kTimingIters)) {
    throw std::runtime_error(
      "timing: captured graph has " + std::to_string(event_nodes) +
      " event-record nodes, expected " + std::to_string(2 * kTimingIters) +
      " -- hipEventRecordExternal did not survive stream capture on this HIP.");
  }
  const size_t work_nodes = num_nodes - event_nodes;
  if (work_nodes <= static_cast<size_t>(kTimingIters)) {
    throw std::runtime_error(
      "timing: captured graph has " + std::to_string(work_nodes) +
      " non-event nodes for " + std::to_string(kTimingIters) +
      " flushes -- vtable.launch enqueued no work on the stream it was "
      "given (rev0 D4/T12).");
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

  // PERFMON_TIMING_METHOD forces one method instead of "hipgraph_ev100,
  // falling back to batched_ev if capture fails". This is a CALIBRATION
  // hook, not a production knob: batched_ev is otherwise unreachable,
  // because capture has never failed on any backend tried, so there would
  // be no way to gather the cross-method data rev0 D6 implicitly depends
  // on (its "never compare across timing_method" rule is only actionable
  // if someone has measured how far apart the methods actually are).
  //
  // Unset -> the normal automatic behaviour. The chosen method is recorded
  // in the returned record exactly as before, so forced runs stay
  // distinguishable in the data from automatic ones.
  const char* forced = std::getenv("PERFMON_TIMING_METHOD");
  const std::string want = forced ? forced : "";
  if (!want.empty() && want != "hipgraph_ev100" && want != "batched_ev") {
    throw std::runtime_error("timing: PERFMON_TIMING_METHOD must be "
                              "'hipgraph_ev100' or 'batched_ev', got '" + want + "'");
  }

  std::vector<double> samples_ms;
  std::string method;
  if (want == "batched_ev") {
    run_batched_ev(vtable, ctx, stream, &samples_ms);
    method = "batched_ev";
  } else if (try_hipgraph_ev100(vtable, ctx, stream, &samples_ms)) {
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
