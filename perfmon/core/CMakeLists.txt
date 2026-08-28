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
# Configured and built with hipcc against theRock ROCm 7.14 (gfx942), and
# T11's own Verify step passes: `ldd` on the installed libperfmon_core.so
# shows no `libaotriton*` and nothing unresolved, with no LD_LIBRARY_PATH
# set. BOTH values of PERFMON_ENABLE_AMDSMI build here -- see
# thermal_amdsmi.cc's header -- and the ON/OFF difference is exactly the
# one T11 specifies: thermal source and the amd_smi link, nothing else.

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

# ROCm's RUNTIME lib dirs, baked into this library's installed RPATH so it
# resolves libamdhip64.so with NO LD_LIBRARY_PATH -- the same requirement
# T13's Verify step places on `bin/runner`, applied one level down since
# `bin/runner` loads this library. These are NOT ${ROCM_PATH}/lib on
# theRock; see PerfmonRocmRuntime.cmake for why.
include("${CMAKE_CURRENT_SOURCE_DIR}/PerfmonRocmRuntime.cmake")
perfmon_rocm_runtime_dirs(_perfmon_rocm_rpath "${_perfmon_rocm_path}")

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

# fill.cc carries a __global__ kernel, so this library embeds a GPU code
# object and is NOT arch-neutral despite being AOTriton-neutral. Without an
# explicit target, hipcc compiles for its own default arch list, and a
# mismatch does not surface until run time as "no kernel image is available
# for execution on the device" -- long after the build looked successful.
# Name the arch instead, and say so if nobody did.
set(PERFMON_TARGET_ARCH "" CACHE STRING
    "GPU architecture to compile device code for, e.g. gfx942")
if(PERFMON_TARGET_ARCH)
  target_compile_options(perfmon_core PRIVATE --offload-arch=${PERFMON_TARGET_ARCH})
  message(STATUS "[perfmon_core] device code for ${PERFMON_TARGET_ARCH}")
else()
  message(WARNING
    "[perfmon_core] PERFMON_TARGET_ARCH is not set; fill.cc's kernel will be "
    "built for hipcc's default architectures, which may not include the GPU "
    "this library ends up running on.")
endif()

set_target_properties(perfmon_core PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_perfmon_rocm_rpath}")

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
# Tests what an include path CONTAINS, not what it is called. The original
# version matched the substring "aotriton" against the path text, which says
# nothing about whether AOTriton headers are reachable: it passed for an
# install prefix named /opt/perf/include that holds aotriton/flash.h, and
# failed for this very directory once the repo was checked out as
# `aotriton.src` -- perfmon/core's own source dir then read
# /wkdir/aotriton.src/perfmon/core and tripped the guard, which is how the
# build node first hit this.
#
# Two checks, matching the two ways AOTriton headers actually arrive:
#   1. an install prefix's include dir, recognised by aotriton/config.h;
#   2. this repo's own include/ or v3src/, by resolved path.
function(perfmon_assert_no_aotriton_includes target)
  get_target_property(_incs ${target} INCLUDE_DIRECTORIES)
  if(NOT _incs)
    return()
  endif()

  get_filename_component(_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH)
  set(_forbidden_roots "${_repo_root}/include" "${_repo_root}/v3src")

  foreach(_inc IN LISTS _incs)
    get_filename_component(_inc_real "${_inc}" REALPATH)

    # (1) Any directory holding AOTriton's own public headers.
    if(EXISTS "${_inc_real}/aotriton/config.h" OR
       EXISTS "${_inc_real}/aotriton/flash.h")
      message(FATAL_ERROR
        "[perfmon_core] '${target}' can see AOTriton headers via ${_inc} "
        "(it contains aotriton/). libperfmon_core must stay AOTriton-neutral "
        "(rev0 D4) -- that property is what makes timing comparable across "
        "subjects.")
    endif()

    # (2) This repo's own AOTriton sources, which carry no aotriton/ subdir.
    foreach(_root IN LISTS _forbidden_roots)
      if(_inc_real STREQUAL "${_root}" OR _inc_real MATCHES "^${_root}/")
        message(FATAL_ERROR
          "[perfmon_core] '${target}' includes ${_inc}, which is inside "
          "AOTriton's own ${_root}. libperfmon_core must stay "
          "AOTriton-neutral (rev0 D4).")
      endif()
    endforeach()
  endforeach()
endfunction()

perfmon_assert_no_aotriton_includes(perfmon_core)

# Parameterized by install prefix (plain CMAKE_INSTALL_PREFIX) so this can
# be configured once per ROCm -- e.g.
# `cmake -DCMAKE_INSTALL_PREFIX=/opt/perfmon/rocm-<ver> ...` -- and every
# subject built against that ROCm column (T13's build_subject.sh) links the
# one shared install instead of rebuilding this library per subject.
# TODO (long term, not scheduled): split this library in two -- the half that
# needs ROCm, and the half that does not.
#
# Today perfmon_core is ONE library that links HIP, so the whole thing is keyed
# on a ROCm version and the install prefix has to carry it
# (<install_prefix><arch>/core/rocm-<ver>, build_subject.sh). Everything in
# here that is pure CPU work -- the line protocol, stats, timing bookkeeping --
# is dragged along for the ride and rebuilt per ROCm column for no reason.
#
# Splitting it would let one build serve several ROCm versions at once, which
# is the actual goal: measuring the same AOTriton tag across ROCm versions
# currently means a separate core install per version, and anything that wants
# to FIND a core install must therefore know the version first. That is a real
# ordering problem, not a tidiness one -- an attempt to install
# launch_runner.sh into this prefix (d275246c) was reverted because the shim is
# what parses the ROCm version out of the preset, so locating the shim under
# rocm-<ver>/ required the answer the shim exists to produce.
#
# This is the same split AOTriton itself wants for libaotriton.so (a CPU-only
# part and a ROCm-dependent part), and the two should be done as one piece of
# work rather than twice with different seams.

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
