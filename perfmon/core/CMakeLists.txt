# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# perfmon-exec0.md T11: libperfmon_core, the AOTriton-NEUTRAL half of
# perfmon's runner (rev0 D4). Built ONCE PER ROCm and reused by every
# subject's libperfmon_flash@<subject> in that column -- this file is
# therefore a standalone CMake project, never `add_subdirectory()`-ed from
# AOTriton's own top-level CMakeLists.txt, and it must never gain a
# dependency on anything under this repo's `include/` or `v3src/`. That
# property (not a build convenience) is what makes a timing comparison
# between two AOTriton versions meaningful: if this library were rebuilt
# per subject against different AOTriton headers, a measured difference
# could originate in the harness instead of in AOTriton (rev0 D4).
#
# Same CXX-compiler-is-hipcc convention as AOTriton's own top-level
# CMakeLists.txt (project(AOTriton CXX C ASM) + find_package(hip REQUIRED),
# no `enable_language(HIP)`): configure with `CXX=hipcc` (or
# `-DCMAKE_CXX_COMPILER=hipcc`) so that fill.cc's __global__ kernels
# compile, exactly like the rest of this codebase already expects.
#
# UNVERIFIED IN THIS ENVIRONMENT: no ROCm/hipcc/HIP install exists on this
# machine (see perfmon/core/probe_graph_timing.cc's T05 commit message for
# how that was confirmed), so this file has never been configured, let
# alone built. Written strictly to T11's spec; see its own Verify step
# (`ldd` on the built library shows no `libaotriton*`) for what a human
# must still check on a real machine.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PerfmonCore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- T11 spec: OFF selects thermal_stub.cc + HIP only; ON selects --------
# thermal_amdsmi.cc + amd-smi. Nothing else differs between the two builds.
option(PERFMON_ENABLE_AMDSMI "Link amd-smi for the thermal gate (default OFF: thermal_stub.cc, HIP only)" OFF)

# Same ROCM_PATH convention as AOTriton's own top-level CMakeLists.txt:
# prefer ROCM_PATH (e.g. theRock SDK from `rocm-sdk path --root`) over a
# hardcoded /opt/rocm, so this does not silently pick up a stale install.
if(DEFINED ENV{ROCM_PATH})
  set(_perfmon_rocm_path "$ENV{ROCM_PATH}")
else()
  message(WARNING "[perfmon_core] ROCM_PATH is not set, falling back to /opt/rocm. "
                   "Set ROCM_PATH to select a specific ROCm/theRock installation.")
  set(_perfmon_rocm_path "/opt/rocm")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_perfmon_rocm_path}/lib/cmake" "${_perfmon_rocm_path}")

find_package(hip REQUIRED)

if(PERFMON_ENABLE_AMDSMI)
  set(_perfmon_thermal_src thermal_amdsmi.cc)
else()
  set(_perfmon_thermal_src thermal_stub.cc)
endif()

add_library(perfmon_core SHARED
  protocol.cc
  stats.cc
  fill.cc
  timing.cc
  ${_perfmon_thermal_src}
)

target_include_directories(perfmon_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(perfmon_core PUBLIC hip::host)

if(PERFMON_ENABLE_AMDSMI)
  # Deliberately NOT a soft find_package()/find_library() with a friendly
  # fallback message: T08/T11's spec wants this to fail LOUDLY, at
  # compile/link time, with amd-smi's own missing-header/missing-library
  # error as the deliverable for the user to act on -- not a degraded mode
  # this build quietly falls back to (rev0 §5.2: no sysfs fallback, ever,
  # for exactly this reason: a silent degradation path is how ungated
  # numbers get published looking gated).
  target_include_directories(perfmon_core PRIVATE "${_perfmon_rocm_path}/include")
  target_link_libraries(perfmon_core PRIVATE amd_smi)
endif()

# --- rev0 D4: libperfmon_core must never see an AOTriton include path ----
# Explicit, configure-time check (T11's own requirement). This cannot catch
# every possible way an include path could reach the compiler (a system
# default, for instance), but it does catch the failure mode this task
# exists to prevent: someone adding `${AOTRITON_ROOT}/include` to THIS
# target's include path the way T12's adapter_v3.cc CMakeLists.txt
# legitimately does for the family library, one level up.
function(perfmon_assert_no_aotriton_includes target)
  get_target_property(_incs ${target} INCLUDE_DIRECTORIES)
  if(NOT _incs)
    return()
  endif()
  foreach(_inc IN LISTS _incs)
    string(TOLOWER "${_inc}" _inc_lower)
    if(_inc_lower MATCHES "aotriton")
      message(FATAL_ERROR
        "[perfmon_core] '${target}' has an AOTriton-looking include path on "
        "its compile line: ${_inc}. libperfmon_core must stay AOTriton-"
        "neutral (rev0 D4) -- that property is what makes timing comparable "
        "across subjects.")
    endif()
  endforeach()
endfunction()

perfmon_assert_no_aotriton_includes(perfmon_core)

# Parameterized by install prefix (plain CMAKE_INSTALL_PREFIX) so this can
# be configured once per ROCm -- e.g.
# `cmake -DCMAKE_INSTALL_PREFIX=/opt/perfmon/rocm-<ver> ...` -- and every
# subject built against that ROCm column (T13's build_subject.sh) links the
# one shared install instead of rebuilding this library per subject.
include(GNUInstallDirs)
install(TARGETS perfmon_core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES
  perfmon_abi.h
  protocol.h
  stats.h
  thermal.h
  fill.h
  timing.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/perfmon)
