// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon-exec0.md T08: thermal.h's amd-smi-backed implementation, built
// only when PERFMON_ENABLE_AMDSMI=ON (T11's CMake option). Mirrors
// python/tune/gpu_utils.py's own AMD-SMI scoping discipline
// (`_own_amdsmi_device`, `_pick_temp_sensor`, `wait_gpu_temperature`):
// resolve EXACTLY ONE processor handle for this process's GPU, by PCI BDF
// -- never `amdsmi_get_processor_handles()` enumerating every device,
// which would make this process appear on every GPU in `amd-smi process`
// -- and hold no reference to any other device for the lifetime of the
// process.
//
// ============================================================================
// STATUS: BUILT AND RUN (theRock ROCm 7.14 / amd-smi 26.5.0, 8x gfx942).
//
// T08 shipped this file unbuilt, on the assumption that amd-smi could not
// be linked in the container. On this node it links fine -- amd_smi ships
// in the SDK (`$ROCM_PATH/lib/cmake/amd_smi`, `libamd_smi.so.26`,
// `include/amd_smi/amdsmi.h`) -- so `-DPERFMON_ENABLE_AMDSMI=ON` builds and
// runs. The T08 escalation is therefore resolved on this platform.
//
// Two of the three API assumptions this file was written with turned out to
// be WRONG, and both were caught only by running it:
//
//   1. `amdsmi_get_temp_metric` returns WHOLE DEGREES CELSIUS, not
//      millidegrees. The original `/1000.0` turned 41 C into 0.041 C --
//      which does not look like a wrong number so much as a disabled gate,
//      since wait_until_cool() could then never exceed any real threshold.
//      Fixed in read_temp_c(); see its comment.
//   2. `amdsmi_gpu_metrics_t::throttle_status` exists and is spelled as
//      assumed, but on this hardware it reads 0xFFFFFFFF -- amdsmi.h's
//      documented "field unsupported" sentinel (and
//      `indep_throttle_status` reads UINT64_MAX). Treating nonzero as
//      "throttled" therefore reported an idle 41 C GPU as throttled. See
//      read_throttled().
//
// Assumption 3 (`amdsmi_bdf_t` member spelling) was correct as written.
//
// Verified against ground truth: `amd-smi metric -g 0 -t` reports
// HOTSPOT: 41 C while this file's JUNCTION read returns 41. EDGE returns
// AMDSMI_STATUS_NOT_SUPPORTED on gfx942, so pick_temp_sensor()'s
// JUNCTION-first order is load-bearing here, exactly as its comment says.
//
// NOT yet exercised: the actual cooldown loop in wait_until_cool() -- no
// measurement on this node has yet pushed the junction temperature above
// the 85 C threshold, so no `OVERHEATING:` line has ever been emitted.
// ============================================================================

#include "thermal.h"
#include "protocol.h"

#include <hip/hip_runtime.h>
#include <amd_smi/amdsmi.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace perfmon {

namespace {

// JUNCTION preferred over EDGE -- exactly gpu_utils.py's _pick_temp_sensor()
// rationale: gfx1151 answers junction with AMDSMI_STATUS_NOT_SUPPORTED
// (edge-only sensor), gfx942 is the mirror image (junction-only, no edge).
constexpr amdsmi_temperature_type_t kTempSensors[] = {
    AMDSMI_TEMPERATURE_TYPE_JUNCTION,
    AMDSMI_TEMPERATURE_TYPE_EDGE,
};

struct OwnedDevice {
  amdsmi_processor_handle handle = nullptr;
  amdsmi_temperature_type_t sensor = AMDSMI_TEMPERATURE_TYPE_JUNCTION;
  bool initialized = false;
};

OwnedDevice& owned_device() {
  static OwnedDevice dev;
  return dev;
}

// This process is assumed to own exactly one HIP device: local index 0
// within whatever ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES mask the
// runner's caller set (perfmon/build_subject.sh's runner is one process
// per subject per GPU, matching every other perfmon core component's
// single-GPU-per-process assumption -- rev0 D4). Resolved via HIP, not by
// enumerating AMD-SMI handles, for the same reason gpu_utils.py's
// `_bdf_of()` reads torch.cuda device properties instead of
// `amdsmi_get_processor_handles()`: it is the only mapping that survives a
// visible-devices mask.
amdsmi_bdf_t bdf_of_this_process_gpu() {
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("thermal_amdsmi: hipGetDeviceProperties failed: ") +
                              hipGetErrorString(err));
  }
  amdsmi_bdf_t bdf{};
  bdf.bdf.function_number = 0;
  bdf.bdf.device_number = static_cast<uint64_t>(prop.pciDeviceID);
  bdf.bdf.bus_number = static_cast<uint64_t>(prop.pciBusID);
  bdf.bdf.domain_number = static_cast<uint64_t>(prop.pciDomainID);
  return bdf;
}

amdsmi_temperature_type_t pick_temp_sensor(amdsmi_processor_handle handle) {
  for (auto sensor : kTempSensors) {
    int64_t temp_c = 0;
    amdsmi_status_t st = amdsmi_get_temp_metric(handle, sensor, AMDSMI_TEMP_CURRENT,
                                                 &temp_c);
    if (st == AMDSMI_STATUS_SUCCESS) {
      return sensor;
    }
  }
  // Loud on purpose, same as gpu_utils.py's _pick_temp_sensor(): silently
  // skipping the gate would let a hot GPU cook.
  throw std::runtime_error(
      "thermal_amdsmi: GPU implements neither JUNCTION nor EDGE temperature sensor");
}

OwnedDevice& ensure_device() {
  OwnedDevice& dev = owned_device();
  if (dev.initialized) {
    return dev;
  }
  amdsmi_status_t st = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (st != AMDSMI_STATUS_SUCCESS) {
    throw std::runtime_error("thermal_amdsmi: amdsmi_init failed");
  }
  amdsmi_bdf_t bdf = bdf_of_this_process_gpu();
  amdsmi_processor_handle handle = nullptr;
  st = amdsmi_get_processor_handle_from_bdf(bdf, &handle);
  if (st != AMDSMI_STATUS_SUCCESS) {
    amdsmi_shut_down();
    throw std::runtime_error("thermal_amdsmi: amdsmi_get_processor_handle_from_bdf failed");
  }
  dev.sensor = pick_temp_sensor(handle);
  dev.handle = handle;
  dev.initialized = true;
  return dev;
}

double read_temp_c(const OwnedDevice& dev) {
  int64_t temp_c = 0;
  amdsmi_status_t st =
      amdsmi_get_temp_metric(dev.handle, dev.sensor, AMDSMI_TEMP_CURRENT, &temp_c);
  if (st != AMDSMI_STATUS_SUCCESS) {
    throw std::runtime_error("thermal_amdsmi: amdsmi_get_temp_metric failed");
  }
  // WHOLE DEGREES CELSIUS, not millidegrees. amdsmi.h documents this
  // outright ("a pointer to int64_t to which the temperature is in
  // Celsius"), and it is confirmed on this install: the JUNCTION sensor
  // returns 41 while `amd-smi metric -g 0 -t` reports HOTSPOT: 41 °C.
  //
  // This previously divided by 1000. That was not a cosmetic error -- it
  // reported 41 C as 0.041 C, so wait_until_cool() could never exceed any
  // sane threshold and the thermal gate silently never engaged, which is
  // exactly the failure mode rev0 §5.2 exists to prevent.
  return static_cast<double>(temp_c);
}

double read_sclk_mhz(amdsmi_processor_handle handle) {
  amdsmi_clk_info_t info{};
  amdsmi_status_t st = amdsmi_get_clock_info(handle, AMDSMI_CLK_TYPE_GFX, &info);
  if (st != AMDSMI_STATUS_SUCCESS) {
    return 0.0;  // diagnostic-only field; do not fail the gate over it
  }
  return static_cast<double>(info.clk);
}

// Sets `*known` false when this platform does not report throttle state, so
// the caller can record `"throttled": null` rather than assert a state it
// never observed.
//
// amdsmi.h: "'throttle_status' will contain 0xFFFFFFFF and UINT64_MAX for
// uint64_t elements such as 'indep_throttle_status'" when unsupported.
// gfx942 does exactly that -- both fields read all-ones here -- so the
// original `!= 0` test reported every idle GPU as throttled.
bool read_throttled(amdsmi_processor_handle handle, bool* known) {
  *known = false;
  amdsmi_gpu_metrics_t metrics{};
  amdsmi_status_t st = amdsmi_get_gpu_metrics_info(handle, &metrics);
  if (st != AMDSMI_STATUS_SUCCESS) {
    return false;  // diagnostic-only field; do not fail the gate over it
  }
  // Prefer the newer per-cause field; fall back to the legacy one. Either
  // may be the all-ones sentinel independently.
  if (metrics.indep_throttle_status != UINT64_MAX) {
    *known = true;
    return metrics.indep_throttle_status != 0;
  }
  if (metrics.throttle_status != 0xFFFFFFFFu) {
    *known = true;
    return metrics.throttle_status != 0;
  }
  return false;
}

}  // namespace

bool wait_until_cool(double threshold_c, std::ostream& out) {
  OwnedDevice& dev = ensure_device();
  double temp = read_temp_c(dev);
  if (temp <= threshold_c) {
    return true;
  }
  const auto start = std::chrono::steady_clock::now();
  while (temp > threshold_c) {
    const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    std::ostringstream detail;
    // Spec wording (T08): "OVERHEATING: gpu=<id> temp=<c> waited=<s>".
    // gpu id is always 0 here -- see bdf_of_this_process_gpu()'s doc
    // comment on the single-GPU-per-process assumption.
    detail << "gpu=0 temp=" << temp << " waited=" << waited;
    emit_overheating(out, detail.str());
    std::this_thread::sleep_for(std::chrono::seconds(5));
    temp = read_temp_c(dev);
  }
  return true;
}

pmon_thermal thermal_snapshot() {
  OwnedDevice& dev = ensure_device();
  pmon_thermal snap;
  snap.temp_c = read_temp_c(dev);
  snap.sclk_mhz = read_sclk_mhz(dev.handle);
  snap.throttled = read_throttled(dev.handle, &snap.throttle_known);
  snap.valid = true;
  return snap;
}

}  // namespace perfmon
