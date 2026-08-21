// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T08: default thermal.h implementation. Ships by
// default (PERFMON_ENABLE_AMDSMI=OFF, T11) so every configuration other
// than the gfx1201 overheating case (rev0 §5.2's stated release blocker
// for gfx1201 publication only) can be brought up without linking amd-smi.
//
// No sysfs fallback, ever (rev0 §5.2): this file's only two behaviors are
// "return true immediately" and "report nothing" -- never a second,
// less-trustworthy temperature source pretending to gate.

#include "thermal.h"

namespace perfmon {

bool wait_until_cool(double /*threshold_c*/, std::ostream& /*out*/) {
  return true;  // nothing to gate on; safe to measure unconditionally.
}

pmon_thermal thermal_snapshot() {
  // valid = false, everything else zero -- see thermal.h's pmon_thermal
  // doc comment: this must be serialized as "thermal": null, not as a
  // struct of real-looking zero readings.
  return pmon_thermal{};
}

}  // namespace perfmon
