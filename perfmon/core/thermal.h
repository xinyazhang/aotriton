// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T08: the thermal-gate interface. Exactly one of
// thermal_stub.cc (default) / thermal_amdsmi.cc (PERFMON_ENABLE_AMDSMI=ON,
// T11's CMake option) provides this header's two functions in any given
// build. NO CALL SITE MAY BRANCH ON WHICH ONE WAS COMPILED -- that is the
// entire point of this header (rev0 §5.2).
//
// perfmon-rev0.md §5.2 explicitly rules out a sysfs fallback: a second
// temperature source is a second thing to keep correct, and a silent
// degradation path is how ungated numbers get published looking gated.
// There are exactly two implementations, never three.

#ifndef PERFMON_CORE_THERMAL_H
#define PERFMON_CORE_THERMAL_H

#include <iostream>
#include <ostream>

namespace perfmon {

// Point-in-time thermal reading.
//
// `valid == false` on the stub build (T08) -- an ungated measurement must
// be distinguishable in the data from a gated one, or the two silently
// merge. Callers (T10's timing.cc) must serialize `!valid` as
// `"thermal": null`, never as a struct of zeros: `temp_c`/`sclk_mhz`/
// `throttled` are meaningless when `valid` is false and must not be
// mistaken for real readings of 0.
struct pmon_thermal {
  double temp_c = 0.0;
  double sclk_mhz = 0.0;
  bool   throttled = false;
  // False when the platform does not report throttle state at all, in which
  // case `throttled` is meaningless and the record must serialize
  // `"throttled": null`. Not every GPU answers this: gfx942 returns
  // amd-smi's documented 0xFFFFFFFF "unsupported" sentinel for
  // amdsmi_gpu_metrics_t::throttle_status. Same reasoning as `valid` below
  // -- "we did not observe throttling" and "this GPU cannot tell us" are
  // different facts, and collapsing them is how a throttled run gets
  // published looking clean (rev0 §5.2).
  bool   throttle_known = false;
  bool   valid = false;
};

// True once it is safe to measure. On the amd-smi build, blocks, polling
// every 5s while the GPU's junction temperature stays above `threshold_c`,
// emitting one `perfmon::emit_overheating` line per poll to `out` (default
// stdout) so the exaid wire protocol's read timeout keeps getting reset
// (see protocol.h). On the stub build, returns `true` immediately without
// blocking or writing anything to `out` -- there is nothing to gate on.
bool wait_until_cool(double threshold_c, std::ostream& out = std::cout);

// Point-in-time thermal reading; see `pmon_thermal` above for the
// stub-build (`valid == false`) contract.
pmon_thermal thermal_snapshot();

}  // namespace perfmon

#endif  // PERFMON_CORE_THERMAL_H
