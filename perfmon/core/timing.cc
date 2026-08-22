// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "timing.h"

#include <cstdlib>
#include <map>
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

// Returns `graph`'s nodes in execution order, or an empty vector if the
// graph is not one straight chain.
//
// Capturing a single stream that never forks produces exactly that: one
// root, one leaf, every node with in-degree and out-degree <= 1, and
// edges == nodes - 1. Verified for attn_fwd (2 nodes per launch) and
// attn_bwd backend 2 (4 nodes per launch, including its dq_acc memset) on
// gfx942. Anything else means the family launch forked onto a second
// stream, which this file's event insertion does not model -- the caller
// treats that as "cannot time by graph" rather than guessing.
std::vector<hipGraphNode_t> linear_chain(hipGraph_t graph) {
  size_t num_nodes = 0;
  check(hipGraphGetNodes(graph, nullptr, &num_nodes), "hipGraphGetNodes");
  std::vector<hipGraphNode_t> nodes(num_nodes);
  if (num_nodes == 0) return {};
  check(hipGraphGetNodes(graph, nodes.data(), &num_nodes), "hipGraphGetNodes (fill)");

  size_t num_edges = 0;
  check(hipGraphGetEdges(graph, nullptr, nullptr, &num_edges), "hipGraphGetEdges");
  if (num_edges != num_nodes - 1) return {};
  std::vector<hipGraphNode_t> efrom(num_edges), eto(num_edges);
  if (num_edges > 0) {
    check(hipGraphGetEdges(graph, efrom.data(), eto.data(), &num_edges),
          "hipGraphGetEdges (fill)");
  }

  std::map<hipGraphNode_t, hipGraphNode_t> next;
  std::map<hipGraphNode_t, int> indeg, outdeg;
  for (auto n : nodes) { indeg[n] = 0; outdeg[n] = 0; }
  for (size_t e = 0; e < num_edges; ++e) {
    if (++outdeg[efrom[e]] > 1 || ++indeg[eto[e]] > 1) return {};
    next[efrom[e]] = eto[e];
  }

  hipGraphNode_t root = nullptr;
  for (auto n : nodes) {
    if (indeg[n] == 0) {
      if (root) return {};  // more than one root
      root = n;
    }
  }
  if (!root) return {};

  std::vector<hipGraphNode_t> order;
  for (hipGraphNode_t n = root; ; ) {
    order.push_back(n);
    auto it = next.find(n);
    if (it == next.end()) break;
    n = it->second;
  }
  if (order.size() != num_nodes) return {};
  return order;
}

// Splices `ev_node` into the chain between `after` and `before`, so the
// timestamp is taken strictly between them. Matching
// python/tune/gpu_utils.py:do_bench, where `start_event.record()` is issued
// in stream order ahead of `fn()` rather than concurrently with it.
void splice_after(hipGraph_t graph, hipGraphNode_t after, hipGraphNode_t before,
                   hipEvent_t ev, hipGraphNode_t* out) {
  check(hipGraphAddEventRecordNode(out, graph, &after, 1, ev),
        "hipGraphAddEventRecordNode");
  if (before) {
    check(hipGraphRemoveDependencies(graph, &after, &before, 1),
          "hipGraphRemoveDependencies");
    check(hipGraphAddDependencies(graph, out, &before, 1), "hipGraphAddDependencies");
  }
}

// Primary method (rev0 §5.1 `hipgraph_ev100`): kTimingIters iterations of
// [L2-flush memset, event record, family launch, event record] in one
// hipGraph -- instantiate, warm up, launch once, read back every
// per-iteration hipEventElapsedTime.
//
// Built in two steps, because T05's hardware probe found that HIP's stream
// capture SILENTLY DISCARDS hipEventRecord: capturing
// [record, kernel, record] yields a graph whose hipGraphGetNodes count is
// 1, not 3, and every subsequent hipEventElapsedTime on those events fails
// with hipErrorInvalidResourceHandle because they were never recorded.
// Measured on gfx942 / ROCm 7.14. Added as explicit nodes instead, the
// same events read back correctly and agree with conventional (non-graph)
// timing to 0.05%.
//
// Capture is otherwise faithful, so the whole loop is captured in one go
// and only the event nodes are added afterwards. The result is a flat
// chain of M + N*kTimingIters nodes, exactly the shape
// python/tune/gpu_utils.py:do_bench produces on a stream, with nothing
// nested and no per-node-type knowledge on perfmon's side.
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

  // --- 1. Learn how many nodes ONE launch contributes --------------------
  // Used only to check that the full-loop capture below came out the size
  // it should. A family whose per-call node count varies would otherwise
  // put every event node at the wrong place in the chain and produce
  // plausible, wrong timings instead of an error.
  hipError_t begin_err = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  if (begin_err != hipSuccess) {
    return false;  // documented fallback trigger
  }
  int probe_rc = vtable.launch(ctx, stream);
  // End the capture unconditionally, even when the launch failed, so the
  // stream is never left capturing for the batched_ev fallback.
  hipError_t probe_end = hipStreamEndCapture(stream, &scratch.child);
  check_launch(probe_rc, "vtable.launch (node-count probe)");
  if (probe_end != hipSuccess) {
    return false;  // documented fallback trigger
  }
  size_t launch_nodes = 0;
  check(hipGraphGetNodes(scratch.child, nullptr, &launch_nodes), "hipGraphGetNodes (probe)");
  if (launch_nodes == 0) {
    throw std::runtime_error(
      "timing: vtable.launch enqueued no work on the captured stream -- the "
      "family adapter must launch on the stream it is given, never the null "
      "stream (rev0 D4/T12).");
  }

  // --- 2. Capture the WHOLE timed loop -----------------------------------
  // Stream capture handles kernels and memsets faithfully; the only thing
  // it drops is hipEventRecord (T05). So capture all kTimingIters
  // iterations of [L2-flush memset, family launch] in one go and add the
  // event nodes afterwards, in step 3.
  //
  // This is the whole graph: M + N*kTimingIters nodes in one straight
  // chain, the same shape python/tune/gpu_utils.py:do_bench produces on a
  // stream. Nothing is nested. An earlier version captured a single launch
  // and wrapped it in one hipGraphAddChildGraphNode per iteration; that
  // cost ~11.5 us per iteration in graph-node dispatch, all of it inside
  // the timed window -- a +67% artifact on a 17 us kernel. Letting capture
  // build the chain also means perfmon never has to know how to copy a
  // kernel node, so no family can outgrow it by using a node type this
  // file did not anticipate.
  begin_err = hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal);
  if (begin_err != hipSuccess) {
    return false;
  }
  int launch_rc = 0;
  for (int i = 0; i < kTimingIters && launch_rc == 0; ++i) {
    hipError_t ferr = hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream);
    if (ferr != hipSuccess) {
      launch_rc = -1;
      break;
    }
    launch_rc = vtable.launch(ctx, stream);
  }
  hipError_t end_err = hipStreamEndCapture(stream, &scratch.graph);
  check_launch(launch_rc, "vtable.launch (timed-loop capture)");
  if (end_err != hipSuccess) {
    return false;  // documented fallback trigger
  }

  // --- 3. Splice the event nodes into the captured chain -----------------
  std::vector<hipGraphNode_t> chain = linear_chain(scratch.graph);
  const size_t per_iter = 1 + launch_nodes;  // flush + this launch's nodes
  if (chain.empty() || chain.size() != per_iter * static_cast<size_t>(kTimingIters)) {
    throw std::runtime_error(
      "timing: captured timed loop is not the expected straight chain of " +
      std::to_string(per_iter * kTimingIters) + " nodes (got " +
      std::to_string(chain.size()) + "). A family launch that forks onto "
      "another stream, or whose node count varies per call, cannot have its "
      "per-iteration event nodes placed correctly.");
  }

  scratch.start_ev.assign(kTimingIters, nullptr);
  scratch.end_ev.assign(kTimingIters, nullptr);
  for (int i = 0; i < kTimingIters; ++i) {
    check(hipEventCreateWithFlags(&scratch.start_ev[i], hipEventDefault), "hipEventCreateWithFlags");
    check(hipEventCreateWithFlags(&scratch.end_ev[i], hipEventDefault), "hipEventCreateWithFlags");
  }

  for (int i = 0; i < kTimingIters; ++i) {
    const size_t base = static_cast<size_t>(i) * per_iter;
    hipGraphNode_t flush_node = chain[base];              // L2 flush, before the window
    hipGraphNode_t first_launch = chain[base + 1];        // first node of the launch
    hipGraphNode_t last_launch = chain[base + per_iter - 1];
    // The node the end-record must precede: next iteration's flush, or
    // nothing at all on the final iteration (it becomes the new leaf).
    hipGraphNode_t next_flush =
        (i + 1 < kTimingIters) ? chain[base + per_iter] : nullptr;

    hipGraphNode_t start_node = nullptr;
    hipGraphNode_t end_node = nullptr;
    splice_after(scratch.graph, flush_node, first_launch, scratch.start_ev[i], &start_node);
    splice_after(scratch.graph, last_launch, next_flush, scratch.end_ev[i], &end_node);
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
