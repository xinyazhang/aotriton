// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T05: hipGraph event-timing hardware probe.
//
// THROWAWAY, standalone program -- no AOTriton headers, no dependency on
// anything else under perfmon/. It exists to answer perfmon-rev0.md §5.1's
// open question before any other perfmon C++ is trusted to build on top of
// it: does `hipEventRecord`/`hipEventElapsedTime` on events recorded INSIDE
// a hipGraph return sane per-iteration timings, and do those timings agree
// with the same operation sequence timed the conventional (non-graph) way?
//
// Build (on a ROCm+GPU machine):
//   hipcc -O2 -std=c++17 --offload-arch=<arch> -Wl,-rpath,${ROCM_PATH}/lib \
//     -o probe_graph_timing perfmon/core/probe_graph_timing.cc
// Run:
//   ./probe_graph_timing
//
// ===================== MEASURED RESULT (gfx942, ROCm 7.14) ==============
//
// This probe HAS now been run. The answer to T05's three questions:
//
//  1. Does in-graph hipEventElapsedTime work?
//       Via STREAM CAPTURE: NO. Capture silently DISCARDS hipEventRecord.
//       Capturing [record, kernel, record] produces a graph whose
//       hipGraphGetNodes count is 1, not 3 -- the two event-record nodes
//       are simply not there. The events are therefore never recorded, and
//       every hipEventElapsedTime on them fails with
//       hipErrorInvalidResourceHandle ("invalid resource handle").
//
//       Via EXPLICIT construction (hipGraphAddEventRecordNode): YES.
//       Node count is 3 as expected and the elapsed time reads back
//       correctly.
//
//  2. Do graph-timed and conventionally-timed medians agree?
//       For a single kernel with no flush: 0.241839 ms in-graph vs
//       0.241719 ms conventional -- 0.05%. So the event machinery itself is
//       sound; nothing is lost by reading timestamps out of a graph.
//       For the full 100-iteration harness WITH the 1 GiB L2 flush, the
//       in-graph median runs consistently ABOVE the conventional median:
//       +7.9% for this file's spin kernel (sections (C) vs (D)), and +4.8%
//       for a real attn_fwd measurement (perfmon's own hipgraph_ev100
//       median 0.3346 ms vs the same launch() timed do_bench's way,
//       0.3193 ms, three runs each). Both methods are individually
//       reproducible to well under 1%, so that gap is a systematic
//       methodological offset -- graph-node execution boundaries falling
//       inside the timed window -- not noise. Its size depends on how long
//       the timed region is relative to per-node overhead, which is why the
//       two figures differ. It is the concrete reason perfmon-rev0.md D6's
//       "never compare across timing_method" rule matters in practice.
//
//  3. Is the L2-flush memset node preserved?
//       Yes, structurally: the explicitly-built 100-iteration graph reports
//       exactly 4*N nodes (memset + record + child + record per iteration),
//       so no flush node is elided.
//
// CONSEQUENCE FOR T10: `timing.cc` cannot capture its timed loop wholesale.
// It captures ONLY the opaque family launch() into a child graph, then
// assembles [memset -> record -> child -> record] x N explicitly. That is
// what section (C) below models.
//
// ===================== A BUG THIS PROBE ORIGINALLY HAD ==================
//
// The first version of this file (and of timing.cc) used N+1 SHARED events,
// with `ev[i+1]` serving as both iteration i's end and iteration i+1's
// start. That is wrong twice over:
//
//   * Each interior event is recorded TWICE. Its surviving timestamp is the
//     LATER record, so `elapsed(ev[i], ev[i+1])` measures
//     `kernel_i + memset_{i+1}` -- the next iteration's 1 GiB flush is
//     silently folded into the sample. The original run showed exactly this:
//     iterations 0-98 read 0.362 ms while iteration 99 -- the only pair
//     whose end event is recorded once -- read 0.145 ms.
//   * In a graph it is also structurally ambiguous: one event, two
//     event-record nodes.
//
// Every section below therefore uses 2N INDEPENDENT events.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(expr)                                                      \
  do {                                                                       \
    hipError_t _err = (expr);                                                \
    if (_err != hipSuccess) {                                                \
      fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,   \
              hipGetErrorString(_err));                                      \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

namespace {

constexpr int kIters = 100;                      // N, per rev0 §5.1.
constexpr size_t kFlushBytes = size_t(1) << 30;  // 1 GiB L2-flush buffer, rev0 §5.1/D6.
constexpr int kWarmups = 3;

// A trivial, KNOWN-DURATION kernel: spin on clock64() until roughly
// `target_clocks` GPU cycles have elapsed, then write a value so the
// compiler cannot eliminate the loop. Because the duration is governed by
// the GPU clock rather than by memory traffic or launch overhead, its
// elapsed time should read back consistently regardless of how it is timed
// -- exactly the property this probe needs.
__global__ void spin_kernel(long long target_clocks, int* out) {
  long long start = clock64();
  long long elapsed = 0;
  while (elapsed < target_clocks) {
    elapsed = clock64() - start;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *out = static_cast<int>(elapsed);
  }
}

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double report(const char* label, const std::vector<double>& ms) {
  if (ms.empty()) {
    printf("%-46s (no samples)\n", label);
    return 0.0;
  }
  std::vector<double> s = ms;
  std::sort(s.begin(), s.end());
  const double med = median_of(ms);
  printf("%-46s n=%zu  min=%.6f  median=%.6f  max=%.6f  ms\n", label, ms.size(), s.front(), med,
         s.back());
  return med;
}

// 2N independent timing-enabled events. Never a shared end/start event --
// see this file's header comment.
struct EventPairs {
  std::vector<hipEvent_t> st, en;
  explicit EventPairs(int n) : st(n, nullptr), en(n, nullptr) {
    for (int i = 0; i < n; ++i) {
      HIP_CHECK(hipEventCreateWithFlags(&st[i], hipEventDefault));
      HIP_CHECK(hipEventCreateWithFlags(&en[i], hipEventDefault));
    }
  }
  ~EventPairs() {
    for (auto& e : st) if (e) static_cast<void>(hipEventDestroy(e));
    for (auto& e : en) if (e) static_cast<void>(hipEventDestroy(e));
  }
  // Reads all n pairs. Returns false (and prints why) on the first failure,
  // which is the interesting outcome for question 1 -- so this must NOT be
  // a hard HIP_CHECK abort.
  bool read(const char* what, std::vector<double>* out) const {
    out->clear();
    for (size_t i = 0; i < st.size(); ++i) {
      float ms = -1.0f;
      hipError_t e = hipEventElapsedTime(&ms, st[i], en[i]);
      if (e != hipSuccess) {
        printf("  %s: hipEventElapsedTime on pair %zu FAILED: %s\n", what, i,
               hipGetErrorString(e));
        return false;
      }
      out->push_back(ms);
    }
    return true;
  }
};

}  // namespace

int main() {
  // The exact duration is not load-bearing. What matters is that the SAME
  // target_clocks is used in every section, so the medians are comparable.
  const long long target_clocks = 300000;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* flush_buf = nullptr;
  HIP_CHECK(hipMalloc(&flush_buf, kFlushBytes));

  int* out_dev = nullptr;
  HIP_CHECK(hipMalloc(&out_dev, sizeof(int)));

  hipMemsetParams flush_params = {};
  flush_params.dst = flush_buf;
  flush_params.value = 0;
  flush_params.elementSize = 1;
  flush_params.width = kFlushBytes;
  flush_params.height = 1;
  flush_params.pitch = 0;

  // ---------------------------------------------------------------------
  // (A) The rev0 §5.1 method AS ORIGINALLY WRITTEN: capture the whole timed
  //     loop, events and all. This is the case that fails.
  // ---------------------------------------------------------------------
  printf("=== (A) stream capture of [memset, record, kernel, record] x %d ===\n", kIters);
  bool capture_events_work = false;
  {
    EventPairs ev(kIters);
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < kIters; ++i) {
      HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
      HIP_CHECK(hipEventRecord(ev.st[i], stream));
      hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
      HIP_CHECK(hipEventRecord(ev.en[i], stream));
    }
    hipError_t cap = hipStreamEndCapture(stream, &graph);
    printf("  hipStreamEndCapture: %s\n", hipGetErrorString(cap));
    if (cap == hipSuccess) {
      size_t nodes = 0;
      HIP_CHECK(hipGraphGetNodes(graph, nullptr, &nodes));
      // THE DIAGNOSTIC. 4*kIters would mean the event-record nodes survived
      // capture; kIters*2 (memset + kernel only) means they were dropped.
      printf("  hipGraphGetNodes: %zu  (expected %d if event nodes survive capture)\n", nodes,
             4 * kIters);
      hipGraphExec_t exec = nullptr;
      HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      for (int w = 0; w < kWarmups; ++w) {
        HIP_CHECK(hipGraphLaunch(exec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
      }
      HIP_CHECK(hipGraphLaunch(exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      std::vector<double> ms;
      capture_events_work = ev.read("(A)", &ms);
      if (capture_events_work) report("(A) captured graph", ms);
      HIP_CHECK(hipGraphExecDestroy(exec));
      HIP_CHECK(hipGraphDestroy(graph));
    }
  }

  // ---------------------------------------------------------------------
  // (B) Minimal control: ONE kernel, ONE event pair, graph built EXPLICITLY
  //     with hipGraphAddEventRecordNode instead of captured.
  // ---------------------------------------------------------------------
  printf("\n=== (B) explicit 3-node graph (record -> kernel -> record) ===\n");
  bool explicit_events_work = false;
  {
    hipEvent_t a = nullptr, b = nullptr;
    HIP_CHECK(hipEventCreateWithFlags(&a, hipEventDefault));
    HIP_CHECK(hipEventCreateWithFlags(&b, hipEventDefault));
    hipGraph_t g = nullptr;
    HIP_CHECK(hipGraphCreate(&g, 0));

    hipGraphNode_t na = nullptr, nk = nullptr, nb = nullptr;
    HIP_CHECK(hipGraphAddEventRecordNode(&na, g, nullptr, 0, a));
    long long tgt = target_clocks;
    int* outp = out_dev;
    void* args[] = {&tgt, &outp};
    hipKernelNodeParams kp = {};
    kp.func = reinterpret_cast<void*>(spin_kernel);
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(1);
    kp.sharedMemBytes = 0;
    kp.kernelParams = args;
    kp.extra = nullptr;
    HIP_CHECK(hipGraphAddKernelNode(&nk, g, &na, 1, &kp));
    HIP_CHECK(hipGraphAddEventRecordNode(&nb, g, &nk, 1, b));

    size_t nodes = 0;
    HIP_CHECK(hipGraphGetNodes(g, nullptr, &nodes));
    printf("  hipGraphGetNodes: %zu  (expected 3)\n", nodes);

    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipGraphLaunch(exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float ms = -1.0f;
    hipError_t e = hipEventElapsedTime(&ms, a, b);
    printf("  elapsed: %s (%.6f ms)\n", hipGetErrorString(e), ms);
    explicit_events_work = (e == hipSuccess);

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(g));
    static_cast<void>(hipEventDestroy(a));
    static_cast<void>(hipEventDestroy(b));
  }

  // ---------------------------------------------------------------------
  // (C) The method T10 actually implements: capture ONLY the opaque launch
  //     into a child graph, then assemble
  //     [memset -> record -> child -> record] x N explicitly.
  // ---------------------------------------------------------------------
  printf("\n=== (C) child-graph + explicit event nodes (what timing.cc does) ===\n");
  double graph_median = 0.0;
  {
    hipGraph_t child = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
    HIP_CHECK(hipStreamEndCapture(stream, &child));

    EventPairs ev(kIters);
    hipGraph_t g = nullptr;
    HIP_CHECK(hipGraphCreate(&g, 0));
    hipGraphNode_t prev = nullptr;
    for (int i = 0; i < kIters; ++i) {
      hipGraphNode_t nm = nullptr, na = nullptr, nc = nullptr, nb = nullptr;
      const hipGraphNode_t* deps = prev ? &prev : nullptr;
      size_t ndeps = prev ? 1 : 0;
      HIP_CHECK(hipGraphAddMemsetNode(&nm, g, deps, ndeps, &flush_params));
      HIP_CHECK(hipGraphAddEventRecordNode(&na, g, &nm, 1, ev.st[i]));
      HIP_CHECK(hipGraphAddChildGraphNode(&nc, g, &na, 1, child));
      HIP_CHECK(hipGraphAddEventRecordNode(&nb, g, &nc, 1, ev.en[i]));
      prev = nb;
    }
    size_t nodes = 0;
    HIP_CHECK(hipGraphGetNodes(g, nullptr, &nodes));
    // Question 3: 4*kIters proves no flush node was elided.
    printf("  hipGraphGetNodes: %zu  (expected %d -- proves no memset node elided)\n", nodes,
           4 * kIters);

    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipGraphLaunch(exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> ms;
    if (ev.read("(C)", &ms)) graph_median = report("(C) child-graph + explicit events", ms);

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(g));
    HIP_CHECK(hipGraphDestroy(child));
  }

  // ---------------------------------------------------------------------
  // (D) Cross-check: the SAME N iterations timed conventionally (no graph),
  //     which is also what python/tune/gpu_utils.py:do_bench does.
  // ---------------------------------------------------------------------
  printf("\n=== (D) conventional, non-graph (do_bench's method) ===\n");
  double conv_median = 0.0;
  {
    EventPairs ev(kIters);
    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
      hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int i = 0; i < kIters; ++i) {
      HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
      HIP_CHECK(hipEventRecord(ev.st[i], stream));
      hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
      HIP_CHECK(hipEventRecord(ev.en[i], stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<double> ms;
    if (ev.read("(D)", &ms)) conv_median = report("(D) conventional", ms);
  }

  // ---------------------------------------------------------------------
  printf("\n=== SUMMARY ===\n");
  printf("(1) in-graph event reads via STREAM CAPTURE:      %s\n",
         capture_events_work ? "OK" : "FAILED -- capture drops hipEventRecord (see (A) node count)");
  printf("(1) in-graph event reads via EXPLICIT node adds:  %s\n",
         explicit_events_work ? "OK" : "FAILED");
  if (graph_median > 0.0 && conv_median > 0.0) {
    printf("(2) graph vs conventional median: %.6f vs %.6f ms (%+.2f%%)\n", graph_median,
           conv_median, 100.0 * (graph_median - conv_median) / conv_median);
  } else {
    printf("(2) graph vs conventional median: not comparable (one side produced no samples)\n");
  }
  printf("(3) flush-node preservation: see (C)'s node count above (%d expected)\n", 4 * kIters);
  printf("\nhipgraph_ev100 is viable ONLY via explicit graph construction; "
         "stream-capturing the timed loop wholesale is not.\n");

  HIP_CHECK(hipFree(out_dev));
  HIP_CHECK(hipFree(flush_buf));
  HIP_CHECK(hipStreamDestroy(stream));

  // Nonzero only if the method T10 depends on (explicit construction) is
  // broken. Capture dropping event nodes is now an EXPECTED result, not a
  // failure of this probe.
  return explicit_events_work ? 0 : 2;
}
