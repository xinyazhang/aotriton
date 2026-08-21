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
// ASSUMPTIONS ABOUT THE AMD-SMI C API -- FLAGGED FOR HUMAN REVIEW ON A
// ROCM MACHINE BEFORE TRUSTING THIS FILE:
//
// This development environment has no /opt/rocm, no amd_smi/amdsmi.h, and
// no amd-smi install anywhere on the machine (confirmed the same way as
// T05/T06's HIP search). This file has NEVER been compiled, not even
// syntax-checked. Its function/type names are written from the public AMD
// SMI C API documentation
// (https://rocm.docs.amd.com/projects/amdsmi/.../amdsmi_8h.html) as of
// this writing, not verified against an installed header. Specific points
// most likely to need adjustment once a real header is available:
//
//   1. `amdsmi_gpu_metrics_t`'s throttle-status field: its name and
//      bit/enum shape has changed across AMD SMI releases per that
//      project's own changelog. `read_throttled()` below assumes a
//      `throttle_status` member where nonzero means throttled -- check
//      the installed header and adjust if it differs.
//   2. `amdsmi_get_temp_metric`'s CURRENT-metric unit (millidegrees C vs.
//      whole degrees C) has also varied historically. `read_temp_c()`
//      assumes millidegrees (divides by 1000) -- check the installed
//      header's doc comment for `amdsmi_get_temp_metric` before trusting
//      this on a new ROCm version.
//   3. `amdsmi_bdf_t`'s exact member names (`bdf.function_number` etc.)
//      are written per the current public documentation; if the installed
//      header spells them differently, only `bdf_of_this_process_gpu()`
//      needs to change.
//
// T08's own instruction: "thermal_amdsmi.cc will not be exercised until
// the container can link amd-smi; that is fine -- it must compile-check
// cleanly, not run." That compile-check has not happened here and cannot
// happen without a ROCm+amd-smi install; per this task's ship-the-stub-
// and-move-on directive, that is an environment problem for the user to
// resolve, not something to route around by fabricating a fake amd-smi
// header inside this repo.
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
    int64_t temp_millidegrees = 0;
    amdsmi_status_t st = amdsmi_get_temp_metric(handle, sensor, AMDSMI_TEMP_CURRENT,
                                                 &temp_millidegrees);
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
  int64_t temp_millidegrees = 0;
  amdsmi_status_t st =
      amdsmi_get_temp_metric(dev.handle, dev.sensor, AMDSMI_TEMP_CURRENT, &temp_millidegrees);
  if (st != AMDSMI_STATUS_SUCCESS) {
    throw std::runtime_error("thermal_amdsmi: amdsmi_get_temp_metric failed");
  }
  return static_cast<double>(temp_millidegrees) / 1000.0;  // see file header, assumption #2
}

double read_sclk_mhz(amdsmi_processor_handle handle) {
  amdsmi_clk_info_t info{};
  amdsmi_status_t st = amdsmi_get_clock_info(handle, AMDSMI_CLK_TYPE_GFX, &info);
  if (st != AMDSMI_STATUS_SUCCESS) {
    return 0.0;  // diagnostic-only field; do not fail the gate over it
  }
  return static_cast<double>(info.clk);
}

bool read_throttled(amdsmi_processor_handle handle) {
  amdsmi_gpu_metrics_t metrics{};
  amdsmi_status_t st = amdsmi_get_gpu_metrics_info(handle, &metrics);
  if (st != AMDSMI_STATUS_SUCCESS) {
    return false;  // diagnostic-only field; do not fail the gate over it
  }
  return metrics.throttle_status != 0;  // see file header, assumption #1
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
  snap.throttled = read_throttled(dev.handle);
  snap.valid = true;
  return snap;
}

}  // namespace perfmon
