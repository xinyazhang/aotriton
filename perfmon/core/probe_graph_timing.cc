// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T05: hipGraph event-timing hardware probe.
//
// THROWAWAY, standalone program -- no AOTriton headers, no dependency on
// anything else under perfmon/. It exists to answer perfmon-rev0.md §5.1's
// open question before any other perfmon C++ is trusted to build on top of
// it: does `hipEventRecord`/`hipEventElapsedTime` on events recorded INSIDE
// a captured hipGraph return sane per-iteration timings, and do those
// timings agree with the same operation sequence timed the conventional
// (non-graph) way?
//
// Build (on a ROCm+GPU machine):
//   hipcc -O2 -std=c++17 -o probe_graph_timing perfmon/core/probe_graph_timing.cc
// Run:
//   ./probe_graph_timing
//
// THIS FILE HAS NEVER BEEN COMPILED OR RUN. This development environment has
// no ROCm/hipcc/HIP headers and no usable GPU (see perfmon-handoff0.md).
// Written strictly to the perfmon-exec0.md T05 spec and perfmon-rev0.md
// §5.1's `hipgraph_ev100` method description:
//
//   "capture N=100 iterations with L2-flush memset + paired event records
//   inside a hipGraph, instantiate, warm up, launch once, read all event-pair
//   elapsed times."
//
// Exit status is 0 if every step up to printing the summary succeeded
// (capture, instantiate, launch, event-time reads); non-zero HIP errors
// abort immediately via HIP_CHECK. The printed SUMMARY section is meant to
// be read by a human deciding whether hipgraph_ev100 is viable on the target
// ROCm/GPU combination, or whether perfmon must fall back to `batched_ev`
// (rev0 §5.1's documented fallback) instead.

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

constexpr int kIters = 100;                     // N, per rev0 §5.1.
constexpr size_t kFlushBytes = size_t(1) << 30;  // 1 GiB L2-flush buffer, rev0 §5.1/D6.
constexpr int kWarmups = 3;

// A trivial, KNOWN-DURATION kernel: spin on clock64() until roughly
// `target_clocks` GPU cycles have elapsed, then write a value so the
// compiler cannot eliminate the loop. Because the duration is governed by
// the wall/GPU clock rather than by memory traffic or launch overhead, its
// elapsed time should read back consistently regardless of whether it is
// timed via conventional events or via events recorded inside a captured
// graph -- which is exactly the property this probe needs to check.
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

double median_of(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n == 0) return 0.0;
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void print_series(const char* label, const std::vector<float>& ms) {
  printf("=== %s, per-iteration elapsed (ms) ===\n", label);
  for (size_t i = 0; i < ms.size(); ++i) {
    printf("iter %zu: %f ms\n", i, ms[i]);
  }
  printf("%s median: %f ms\n\n", label, median_of(ms));
}

}  // namespace

int main() {
  // Target ~ a few hundred microseconds of spin; the exact duration is not
  // load-bearing. What matters is that the SAME target_clocks value is used
  // in every timed section below, so the three medians are comparable.
  const long long target_clocks = 300000;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* flush_buf = nullptr;
  HIP_CHECK(hipMalloc(&flush_buf, kFlushBytes));

  int* out_dev = nullptr;
  HIP_CHECK(hipMalloc(&out_dev, sizeof(int)));

  // ---------------------------------------------------------------------
  // (A) hipgraph_ev100: capture N=100 iterations of
  //     [L2-flush memset, event record, kernel, event record], using
  //     N+1=101 timing-enabled events so consecutive events bound each
  //     iteration's kernel launch.
  // ---------------------------------------------------------------------
  std::vector<hipEvent_t> ev(kIters + 1);
  for (auto& e : ev) HIP_CHECK(hipEventCreate(&e));  // timing enabled by default

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  for (int i = 0; i < kIters; ++i) {
    HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
    HIP_CHECK(hipEventRecord(ev[i], stream));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
    HIP_CHECK(hipEventRecord(ev[i + 1], stream));
  }
  hipError_t capture_err = hipStreamEndCapture(stream, &graph);

  bool graph_ok = (capture_err == hipSuccess);
  std::vector<float> graph_ms(kIters, -1.0f);

  if (!graph_ok) {
    fprintf(stderr,
            "hipStreamEndCapture FAILED: %s -- hipgraph_ev100 is NOT viable "
            "on this ROCm/GPU combination; perfmon must use the batched_ev "
            "fallback (rev0 %%5.1).\n",
            hipGetErrorString(capture_err));
  } else {
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipGraphLaunch(graph_exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }

    HIP_CHECK(hipGraphLaunch(graph_exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int i = 0; i < kIters; ++i) {
      float ms = -1.0f;
      hipError_t e = hipEventElapsedTime(&ms, ev[i], ev[i + 1]);
      if (e != hipSuccess) {
        fprintf(stderr,
                "iter %d: hipEventElapsedTime on IN-GRAPH events FAILED: %s "
                "-- this is the exact failure mode hipgraph_ev100 must be "
                "checked for.\n",
                i, hipGetErrorString(e));
        graph_ok = false;
        break;
      }
      graph_ms[i] = ms;
    }

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
  }
  if (graph) HIP_CHECK(hipGraphDestroy(graph));

  if (graph_ok) {
    print_series("hipgraph_ev100 (with L2-flush memset)", graph_ms);
  }

  // ---------------------------------------------------------------------
  // (B) Cross-check: the SAME 100 iterations, timed the conventional
  //     (non-graph) way -- record/launch/record issued directly on the
  //     stream, no capture involved.
  // ---------------------------------------------------------------------
  std::vector<hipEvent_t> ev2(kIters + 1);
  for (auto& e : ev2) HIP_CHECK(hipEventCreate(&e));

  for (int w = 0; w < kWarmups; ++w) {
    HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<float> conv_ms(kIters, -1.0f);
  for (int i = 0; i < kIters; ++i) {
    HIP_CHECK(hipMemsetAsync(flush_buf, 0, kFlushBytes, stream));
    HIP_CHECK(hipEventRecord(ev2[i], stream));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
    HIP_CHECK(hipEventRecord(ev2[i + 1], stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  for (int i = 0; i < kIters; ++i) {
    HIP_CHECK(hipEventElapsedTime(&conv_ms[i], ev2[i], ev2[i + 1]));
  }
  print_series("conventional (non-graph)", conv_ms);

  // ---------------------------------------------------------------------
  // (C) Ablation: re-capture WITHOUT the L2-flush memset. If graph
  //     optimization silently elides the memset (or if it has no material
  //     effect on the timed kernel), the no-flush median should be close to
  //     the with-flush graph median from (A); a large gap would indicate the
  //     flush is doing real, timing-relevant work.
  // ---------------------------------------------------------------------
  std::vector<hipEvent_t> ev3(kIters + 1);
  for (auto& e : ev3) HIP_CHECK(hipEventCreate(&e));

  hipGraph_t graph_noflush = nullptr;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  for (int i = 0; i < kIters; ++i) {
    HIP_CHECK(hipEventRecord(ev3[i], stream));
    hipLaunchKernelGGL(spin_kernel, dim3(1), dim3(1), 0, stream, target_clocks, out_dev);
    HIP_CHECK(hipEventRecord(ev3[i + 1], stream));
  }
  hipError_t noflush_capture_err = hipStreamEndCapture(stream, &graph_noflush);

  std::vector<float> noflush_ms(kIters, -1.0f);
  bool noflush_ok = (noflush_capture_err == hipSuccess);
  if (!noflush_ok) {
    fprintf(stderr, "ablation capture FAILED: %s\n", hipGetErrorString(noflush_capture_err));
  } else {
    hipGraphExec_t graph_exec_noflush = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec_noflush, graph_noflush, nullptr, nullptr, 0));
    for (int w = 0; w < kWarmups; ++w) {
      HIP_CHECK(hipGraphLaunch(graph_exec_noflush, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    HIP_CHECK(hipGraphLaunch(graph_exec_noflush, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    for (int i = 0; i < kIters; ++i) {
      HIP_CHECK(hipEventElapsedTime(&noflush_ms[i], ev3[i], ev3[i + 1]));
    }
    HIP_CHECK(hipGraphExecDestroy(graph_exec_noflush));
  }
  if (graph_noflush) HIP_CHECK(hipGraphDestroy(graph_noflush));

  if (noflush_ok) {
    print_series("hipgraph_ev100 (no L2-flush, ablation)", noflush_ms);
  }

  // ---------------------------------------------------------------------
  // SUMMARY -- fill in by hand after running this on a ROCm+GPU machine.
  // ---------------------------------------------------------------------
  printf("=== SUMMARY ===\n");
  printf("(A) hipgraph_ev100 event reads: %s\n",
         graph_ok ? "OK, in-graph hipEventElapsedTime returned values (see above)"
                  : "FAILED -- fall back to batched_ev (rev0 5.1)");
  printf("(B) graph vs conventional median agreement: compare the two "
         "medians printed above (expect close agreement if hipgraph_ev100 "
         "is trustworthy)\n");
  printf("(C) flush-node ablation: compare the no-flush median to (A)'s "
         "with-flush median -- a large gap means the flush is materially "
         "changing steady-state timing, as intended\n");

  HIP_CHECK(hipFree(out_dev));
  HIP_CHECK(hipFree(flush_buf));
  HIP_CHECK(hipStreamDestroy(stream));

  return graph_ok ? 0 : 2;
}
