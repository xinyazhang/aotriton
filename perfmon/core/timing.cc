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
// fails -- the caller falls back to stream_ev in that case, per T10's
// spec ("If capture fails for a given backend, fall back to stream_ev for
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
  // stream is never left capturing for the stream_ev fallback.
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

// Fallback method (`stream_ev`), for a backend that refuses stream
// capture: the SAME kTimingIters iterations of
// [L2-flush memset, record start, family launch, record end], issued
// directly on the stream instead of captured into a graph. Zero graph
// dependency, which is the whole point of having it (rev0 §5.1).
//
// rev0 §5.1 specifies this fallback as "K=20 batches x N=100 back-to-back
// calls, two events per batch, per-batch mean as the sample", and it was
// implemented that way until it was first actually executed. With only two
// events per batch the 1 GiB L2 flush is unavoidably INSIDE the timed
// window, so every sample was kernel + flush. Measured on gfx942 against
// the same shapes:
//
//   seqlen  reported  true kernel   error      (flush measured at 0.2164 ms)
//      128   0.23364      0.02253   +937%
//      512   0.25475      0.04435   +474%
//     4096   0.62049      0.39040    +59%
//
// i.e. a near-constant +0.21 ms, which is unusable at small shapes and
// wrong at every shape. Per-iteration events fix it because the flush then
// sits before the start record, exactly as in hipgraph_ev100 and in
// python/tune/gpu_utils.py:do_bench.
//
// The batch structure bought nothing once that was clear: it existed to
// amortise event overhead, but 2N per-iteration events cost nothing
// measurable here (do_bench uses exactly that shape), and per-batch means
// discard the per-iteration distribution that stats.h reports p05/p95 over.
// So the fallback is now N=100 samples, the same count and shape
// hipgraph_ev100 produces -- the two methods differ only in whether the
// work is dispatched from a graph or from the stream.
//
// RENAMED from "batched_ev" deliberately: there are no batches any more,
// and `timing_method` is a published field that rev0 D6 keys its
// "never compare across methods" rule on. A name that lies about the
// method is worse than a name that differs from the design doc.
void run_stream_ev(const pmon_family_vtable& vtable, void* ctx, hipStream_t stream,
                    std::vector<double>* out_ms) {
  void* flush_buf = flush_buffer();

  std::vector<hipEvent_t> start_ev(kTimingIters, nullptr), end_ev(kTimingIters, nullptr);
  for (int i = 0; i < kTimingIters; ++i) {
    check(hipEventCreateWithFlags(&start_ev[i], hipEventDefault), "hipEventCreateWithFlags");
    check(hipEventCreateWithFlags(&end_ev[i], hipEventDefault), "hipEventCreateWithFlags");
  }

  // Warm up before the timed pass, matching hipgraph_ev100's graph warmup.
  for (int i = 0; i < kWarmupIters; ++i) {
    check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
    check_launch(vtable.launch(ctx, stream), "vtable.launch (warmup)");
  }
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warmup)");

  for (int i = 0; i < kTimingIters; ++i) {
    check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
    check(hipEventRecord(start_ev[i], stream), "hipEventRecord (start)");
    check_launch(vtable.launch(ctx, stream), "vtable.launch (stream)");
    check(hipEventRecord(end_ev[i], stream), "hipEventRecord (end)");
  }
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (timed)");

  out_ms->resize(kTimingIters);
  for (int i = 0; i < kTimingIters; ++i) {
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start_ev[i], end_ev[i]), "hipEventElapsedTime");
    (*out_ms)[i] = static_cast<double>(ms);
  }

  for (int i = 0; i < kTimingIters; ++i) {
    check(hipEventDestroy(start_ev[i]), "hipEventDestroy");
    check(hipEventDestroy(end_ev[i]), "hipEventDestroy");
  }
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
  // falling back to stream_ev if capture fails".
  //
  // This is a CALIBRATION hook, not a production knob, and it is the
  // INTENDED way stream_ev gets run. hipgraph_ev100 is expected to work
  // uniformly, so the automatic fallback should never fire; forcing
  // stream_ev is how the two methods get compared against each other and
  // against gpu_utils.py:do_bench. rev0 D6's "never compare across
  // timing_method" rule is only actionable once someone has measured how
  // far apart the methods actually are -- currently a couple of percent.
  //
  // Unset -> the normal automatic behaviour. The chosen method is recorded
  // in the returned record either way, so a forced run and an (unexpected)
  // automatic fallback are not distinguishable from the record alone --
  // see the fallback's own comment below.
  const char* forced = std::getenv("PERFMON_TIMING_METHOD");
  const std::string want = forced ? forced : "";
  if (!want.empty() && want != "hipgraph_ev100" && want != "stream_ev") {
    throw std::runtime_error("timing: PERFMON_TIMING_METHOD must be "
                              "'hipgraph_ev100' or 'stream_ev', got '" + want + "'");
  }

  std::vector<double> samples_ms;
  std::string method;
  if (want == "stream_ev") {
    run_stream_ev(vtable, ctx, stream, &samples_ms);
    method = "stream_ev";
  } else if (try_hipgraph_ev100(vtable, ctx, stream, &samples_ms)) {
    method = "hipgraph_ev100";
  } else {
    // UNEXPECTED. hipgraph_ev100 works uniformly across every backend
    // tried, so reaching here means stream capture was refused -- an
    // anomaly worth investigating, not a routine slower path.
    //
    // T10's spec is explicit that the measurement must not be dropped
    // ("fall back ... and record which was used. Never drop the
    // measurement."), so it is still taken. But the ONLY trace of the
    // switch is `timing_method` in the record, which means a silently
    // method-switched row can sit in published data looking like any
    // other -- the same shape of silent degradation rev0 §5.2 refuses to
    // allow for the thermal gate. Whoever writes T27/T28 must therefore
    // treat a stream_ev row from an automatic fallback as a finding, not
    // merely as a row that cannot be diffed against hipgraph_ev100.
    run_stream_ev(vtable, ctx, stream, &samples_ms);
    method = "stream_ev";
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
