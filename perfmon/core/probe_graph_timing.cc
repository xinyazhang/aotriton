// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T05: hipGraph event-timing hardware probe.
//
// THROWAWAY, standalone program -- no AOTriton headers, no dependency on
// anything else under perfmon/. It answers perfmon-rev0.md §5.1's open
// question: can `hipEventRecord`/`hipEventElapsedTime` time individual
// iterations from inside a captured hipGraph, and do those timings agree
// with the same sequence timed conventionally?
//
// Build (on a ROCm+GPU machine):
//   hipcc -O2 -std=c++17 --offload-arch=<arch> -Wl,-rpath,${ROCM_PATH}/lib \
//     -o probe_graph_timing perfmon/core/probe_graph_timing.cc
//
// ===================== MEASURED RESULT (gfx942, ROCm 7.14) ==============
//
// YES -- but only with the right flag, and that flag is the whole finding.
//
//   1. `hipEventRecord` (equivalently hipEventRecordDefault) during stream
//      capture does NOT produce a readable timestamp. Capturing
//      [memset, record, kernel, record] x 100 yields 200 nodes, not 400:
//      the event records are absent, and every hipEventElapsedTime on them
//      fails with hipErrorInvalidResourceHandle.
//
//      This is not capture "losing" them. A default-flag record during
//      capture is an INTERNAL capture dependency marker -- the mechanism
//      cross-stream fork/join is built out of -- and is deliberately not
//      materialised as a graph node.
//
//   2. `hipEventRecordWithFlags(ev, stream, hipEventRecordExternal)` is the
//      opt-in. hip_runtime_api.h, at the flag's own definition: "Event is
//      captured in the graph as an external event node when performing
//      stream capture." With it the same capture yields all 400 nodes and
//      every event pair reads back.
//
//   3. Agreement is good: 0.02143 ms in-graph vs 0.02107 ms conventional
//      for the same ~17 us kernel, i.e. +1.7%.
//
//   4. The 1 GiB L2-flush memset node is preserved -- it is counted in the
//      400 and, being outside each event pair, does not enter the samples.
//
// CONSEQUENCE FOR T10: timing.cc captures the whole timed loop in one pass
// with hipEventRecordExternal, and does nothing else. No explicit node
// construction, no child graphs, no post-hoc graph editing.
//
// ===================== TWO WRONG TURNS, RECORDED ========================
//
// Both were real, both were measured, and both are avoided by (2) above.
//
//   * Shared events. The first version of this probe -- and of timing.cc --
//     used N+1 events with ev[i+1] serving as iteration i's end AND
//     iteration i+1's start. Each interior event is then recorded twice and
//     the surviving timestamp is the LATER one, so every sample silently
//     became kernel_i + memset_{i+1}. It showed up as iterations 0-98
//     reading 0.362 ms while iteration 99 -- the only pair whose end event
//     is recorded once -- read 0.145 ms. Everything below uses 2N
//     independent events.
//
//   * Child graphs. Before hipEventRecordExternal was found, the workaround
//     was to capture one launch and wrap it in a hipGraphAddChildGraphNode
//     per iteration. It works, but the nesting costs ~11.5 us per iteration
//     in node dispatch, inside the timed window: +67% on a 17 us kernel.
//     Section (C) still measures this, because "hipGraph timing is
//     inaccurate for small kernels" is a conclusion someone could otherwise
//     reach from the nested shape alone and wrongly generalise.

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
constexpr size_t kFlushBytes = size_t(1) << 30;  // 1 GiB L2 flush, rev0 §5.1/D6.
constexpr int kWarmups = 3;

// Known-duration kernel: spin on clock64() so the elapsed time is governed
// by the GPU clock rather than by memory traffic, and should therefore read
// the same however it is timed. ~17 us at kSpinClocks, chosen to match
// attn_fwd at seqlen=128 -- the shape where measurement overhead matters.
constexpr long long kSpinClocks = 30000;

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
  const size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// 2N independent timing-enabled events; never a shared end/start event.
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
  // Not a hard HIP_CHECK: a failed read is the interesting outcome in (A).
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

double report(const char* label, const std::vector<double>& ms) {
  if (ms.empty()) return 0.0;
  std::vector<double> s = ms;
  std::sort(s.begin(), s.end());
  const double m = median_of(ms);
  printf("  %-44s n=%zu  min=%.6f  median=%.6f  max=%.6f ms\n", label, ms.size(), s.front(), m,
         s.back());
  return m;
}

// Captures [flush, record, kernel, record] x kIters using `flags` for the
// event records, and reports whether the events survived.
double capture_trial(const char* label, unsigned flags, void* flush_buf, int* out_dev,
                     hipStream_t stream, bool* ok) {
  *ok = false;
  EventPairs ev(kIters);
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  for (int i = 0; i < kIters; ++i) {
    HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
    HIP_CHECK(hipEventRecordWithFlags(ev.st[i], stream, flags));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, kSpinClocks, out_dev);
    HIP_CHECK(hipEventRecordWithFlags(ev.en[i], stream, flags));
  }
  hipError_t cap = hipStreamEndCapture(stream, &graph);
  printf("%s\n  hipStreamEndCapture: %s\n", label, hipGetErrorString(cap));
  if (cap != hipSuccess) return 0.0;

  size_t nodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &nodes));
  // THE DIAGNOSTIC: 4*kIters means the event nodes survived capture;
  // 2*kIters (flush + kernel only) means they were never materialised.
  printf("  hipGraphGetNodes: %zu  (expect %d if event nodes survive)\n", nodes, 4 * kIters);

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int w = 0; w < kWarmups; ++w) {
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  }
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<double> ms;
  double med = 0.0;
  *ok = ev.read(label, &ms);
  if (*ok) med = report("in-graph", ms);
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  return med;
}

}  // namespace

int main() {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  void* flush_buf = nullptr;
  HIP_CHECK(hipMalloc(&flush_buf, kFlushBytes));
  int* out_dev = nullptr;
  HIP_CHECK(hipMalloc(&out_dev, sizeof(int)));

  bool default_ok = false, external_ok = false;
  const double default_med = capture_trial(
      "=== (A) capture with hipEventRecordDefault ===", hipEventRecordDefault, flush_buf, out_dev,
      stream, &default_ok);
  static_cast<void>(default_med);
  printf("\n");
  const double external_med = capture_trial(
      "=== (B) capture with hipEventRecordExternal ===", hipEventRecordExternal, flush_buf,
      out_dev, stream, &external_ok);

  // -------------------------------------------------------------------
  // (C) The child-graph shape T10 used before (B) was found. Kept as a
  //     cautionary measurement -- see this file's header.
  // -------------------------------------------------------------------
  printf("\n=== (C) child-graph node per iteration (the superseded shape) ===\n");
  double child_med = 0.0;
  {
    hipGraph_t child = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, kSpinClocks, out_dev);
    HIP_CHECK(hipStreamEndCapture(stream, &child));

    hipMemsetParams mp = {};
    mp.dst = flush_buf; mp.value = 0; mp.elementSize = 1;
    mp.width = kFlushBytes; mp.height = 1; mp.pitch = 0;

    EventPairs ev(kIters);
    hipGraph_t g = nullptr;
    HIP_CHECK(hipGraphCreate(&g, 0));
    hipGraphNode_t prev = nullptr;
    for (int i = 0; i < kIters; ++i) {
      hipGraphNode_t nm = nullptr, na = nullptr, nc = nullptr, nb = nullptr;
      const hipGraphNode_t* dep = prev ? &prev : nullptr;
      const size_t nd = prev ? 1 : 0;
      HIP_CHECK(hipGraphAddMemsetNode(&nm, g, dep, nd, &mp));
      HIP_CHECK(hipGraphAddEventRecordNode(&na, g, &nm, 1, ev.st[i]));
      HIP_CHECK(hipGraphAddChildGraphNode(&nc, g, &na, 1, child));
      HIP_CHECK(hipGraphAddEventRecordNode(&nb, g, &nc, 1, ev.en[i]));
      prev = nb;
    }
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipGraphLaunch(exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<double> ms;
    if (ev.read("(C)", &ms)) child_med = report("child-graph node", ms);
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(g));
    HIP_CHECK(hipGraphDestroy(child));
  }

  // -------------------------------------------------------------------
  // (D) Conventional, non-graph -- also what gpu_utils.py:do_bench does.
  // -------------------------------------------------------------------
  printf("\n=== (D) conventional, non-graph (do_bench's method) ===\n");
  double conv_med = 0.0;
  {
    EventPairs ev(kIters);
    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
      hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, kSpinClocks, out_dev);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int i = 0; i < kIters; ++i) {
      HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
      HIP_CHECK(hipEventRecord(ev.st[i], stream));
      hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, kSpinClocks, out_dev);
      HIP_CHECK(hipEventRecord(ev.en[i], stream));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<double> ms;
    if (ev.read("(D)", &ms)) conv_med = report("conventional", ms);
  }

  printf("\n=== SUMMARY ===\n");
  printf("(A) hipEventRecordDefault  in capture: %s\n",
         default_ok ? "readable" : "NOT readable -- records are internal capture markers");
  printf("(B) hipEventRecordExternal in capture: %s\n",
         external_ok ? "readable  <-- this is what T10 uses" : "NOT readable");
  if (external_ok && conv_med > 0.0) {
    printf("(B) vs (D) conventional: %.6f vs %.6f ms (%+.2f%%)\n", external_med, conv_med,
           100.0 * (external_med - conv_med) / conv_med);
  }
  if (child_med > 0.0 && conv_med > 0.0) {
    printf("(C) child-graph nesting costs %+.2f%% vs conventional -- why T10 does not nest\n",
           100.0 * (child_med - conv_med) / conv_med);
  }
  printf("(D) flush node preserved: counted in (B)'s %d nodes, and outside each event pair\n",
         4 * kIters);

  HIP_CHECK(hipFree(out_dev));
  HIP_CHECK(hipFree(flush_buf));
  HIP_CHECK(hipStreamDestroy(stream));
  return external_ok ? 0 : 2;
}
