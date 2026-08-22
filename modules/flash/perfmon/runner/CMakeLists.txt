# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# perfmon-exec0.md T12: libperfmon_flash@<subject> for flash, built against
# ONE specific subject's own AOTriton install (rev0 D4: "one binary per
# AOTriton version"). Configured and built ONCE PER SUBJECT by T13's
# build_subject.sh, unlike perfmon/core/CMakeLists.txt (T11), which is built
# once per ROCm and reused unchanged across every subject in that column.
#
# Configured and built for subject `aotriton-head+rocm7.14.0` on gfx942, via
# perfmon/build_subject.sh. T13's Verify step passes: `ldd bin/runner` shows
# exactly one libaotriton, resolving under perfmon/subjects/<id>/, with NO
# LD_LIBRARY_PATH set, and `runner <<< "exit"` exits 0.
#
# --- AOTRITON_ROOT (T12 spec, verbatim) -----------------------------------
# "CMake takes AOTRITON_ROOT and compiles against that root's include/,
# linking that root's libaotriton*_v2.so. No system AOTriton." AOTRITON_ROOT
# is exactly one subject's install prefix (perfmon-rev0.md §11 file plan:
# `perfmon/subjects/<id>/aotriton/`), i.e. it has `include/aotriton/*.h`
# under it and `lib/libaotriton*_v2.so` under it (the `*` is load-bearing,
# not sloppy spec wording: v3src/CMakeLists.txt:387 sets
# `OUTPUT_NAME "aotriton${AOTRITON_NAME_SUFFIX}_v2"`, and T13 always builds
# with `AOTRITON_NAME_SUFFIX=pmon`, so the real file is
# `libaotritonpmon_v2.so`, never `libaotriton_v2.so`) -- hardcoding the
# unsuffixed name here would silently link nothing (or, on a host that also
# happens to have an unsuffixed system AOTriton on its default library
# path, silently link the WRONG one, defeating "no system AOTriton").
#
# --- PERFMON_CORE_ROOT (disclosed elaboration beyond T12's spec text) ----
# T12's spec text only mentions AOTRITON_ROOT, but adapter_v3.cc also
# `#include <perfmon/perfmon_abi.h>` / `<perfmon/fill.h>` and calls
# `perfmon::fill_uniform_scaled`/`fill_zero` -- both owned by
# perfmon/core (T11), installed under perfmon/core's own
# `CMAKE_INSTALL_PREFIX` (perfmon/core/CMakeLists.txt's own install()
# rules: `<prefix>/include/perfmon/*.h`, `<prefix>/lib/libperfmon_core.so`).
# T11's own file header states that library is "built ONCE PER ROCm and
# reused by every subject's libperfmon_flash@<subject> in that column", so
# this project needs a second, independent root pointing at THAT install,
# not at AOTRITON_ROOT. PERFMON_CORE_ROOT names it explicitly rather than
# reusing AOTRITON_ROOT for both (they are never the same directory, and
# rev0 D4's core/family split is exactly the property this keeps visible at
# the build-system level, not just in the source tree).
#
# --- `runner` executable target (T13 gap-fill, added after this file's own
# T12 commit) -----------------------------------------------------------
# T13 discovered that no task's Files list ever specified the runner
# executable's main() (see perfmon/core/main.cc's own top comment for the
# full account) -- protocol.h (T07) documents `Handlers`/`protocol_loop` as
# "a runner main() supplies", but that main() did not exist anywhere before
# this amendment. main.cc (+ its HIP-free half, entry_codec.h/.cc) is
# family-neutral source living under perfmon/core/, but the EXECUTABLE that
# results must statically link one specific family's vtable (rev0 D4: no
# dlopen), so it is built HERE, per subject, alongside `perfmon_flash`
# itself -- exactly matching T13 spec step 4's "link `libperfmon_core` +
# family lib -> perfmon/subjects/<id>/bin/runner". (The actual
# get_filename_component()/EXISTS check for perfmon/core/main.cc live below,
# after cmake_minimum_required()/project() -- CMake commands other than
# cmake_policy() are not supposed to run before cmake_minimum_required(),
# which must stay the first real command in this file.)

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PerfmonFlashAdapter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GNUInstallDirs early: CMAKE_INSTALL_LIBDIR/BINDIR are used below, in the
# runner target's INSTALL_RPATH (T13 gap-fix), not just in the install()
# calls at the bottom of this file.
include(GNUInstallDirs)

get_filename_component(_perfmon_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(_perfmon_core_src_dir "${_perfmon_repo_root}/perfmon/core")
if(NOT EXISTS "${_perfmon_core_src_dir}/main.cc")
  message(FATAL_ERROR
    "[perfmon_flash] Could not find perfmon/core/main.cc at "
    "${_perfmon_core_src_dir} -- this CMakeLists.txt assumes it is being "
    "configured from an unmoved checkout of the same repo/worktree that "
    "contains perfmon/core/ (see this file's own 'runner executable "
    "target' comment above for why the runner's main() is built from THIS "
    "checkout's sources, never from a per-tag worktree).")
endif()

if(NOT DEFINED AOTRITON_ROOT)
  message(FATAL_ERROR
    "[perfmon_flash] AOTRITON_ROOT is required, e.g.\n"
    "  cmake -DAOTRITON_ROOT=/path/to/perfmon/subjects/<id>/aotriton "
    "-DPERFMON_CORE_ROOT=/path/to/perfmon/core/install ...\n"
    "It must contain include/aotriton/*.h and lib/libaotriton*_v2.so from "
    "ONE subject's own AOTriton build (rev0 D4) -- never a system AOTriton.")
endif()
if(NOT DEFINED PERFMON_CORE_ROOT)
  message(FATAL_ERROR
    "[perfmon_flash] PERFMON_CORE_ROOT is required (see this file's own "
    "header comment for why this is a second, separate root from "
    "AOTRITON_ROOT): the install prefix T11's perfmon/core/CMakeLists.txt "
    "was installed to for this ROCm, e.g. -DPERFMON_CORE_ROOT=/opt/perfmon/rocm-<ver>.")
endif()

if(NOT EXISTS "${AOTRITON_ROOT}/include/aotriton")
  message(FATAL_ERROR
    "[perfmon_flash] AOTRITON_ROOT (${AOTRITON_ROOT}) has no include/aotriton/ "
    "-- is this a valid AOTriton install prefix?")
endif()

# Same "fail loudly on the actual failure mode this exists to prevent" style
# as perfmon/core/CMakeLists.txt's perfmon_assert_no_aotriton_includes(): a
# missing or ambiguous libaotriton*_v2.so under AOTRITON_ROOT is exactly the
# case where silently falling through to a system AOTriton (if one happens
# to be on the linker's default search path) would produce a runner that
# LOOKS built but silently measures the wrong AOTriton build.
file(GLOB _perfmon_aotriton_libs "${AOTRITON_ROOT}/lib/libaotriton*_v2.so"
                                  "${AOTRITON_ROOT}/lib64/libaotriton*_v2.so")
list(LENGTH _perfmon_aotriton_libs _perfmon_aotriton_lib_count)
if(_perfmon_aotriton_lib_count EQUAL 0)
  message(FATAL_ERROR
    "[perfmon_flash] No libaotriton*_v2.so found under ${AOTRITON_ROOT}/lib "
    "or ${AOTRITON_ROOT}/lib64. Build that subject first (T13's "
    "build_subject.sh) before configuring this project against it.")
elseif(_perfmon_aotriton_lib_count GREATER 1)
  message(FATAL_ERROR
    "[perfmon_flash] Found more than one libaotriton*_v2.so under "
    "${AOTRITON_ROOT}: ${_perfmon_aotriton_libs}. Each subject's install "
    "prefix must contain exactly one -- this looks like a stale or shared "
    "install prefix reused across subjects, which rev0 D4 forbids.")
endif()
list(GET _perfmon_aotriton_libs 0 _perfmon_aotriton_lib)

if(NOT EXISTS "${PERFMON_CORE_ROOT}/include/perfmon/perfmon_abi.h")
  message(FATAL_ERROR
    "[perfmon_flash] PERFMON_CORE_ROOT (${PERFMON_CORE_ROOT}) has no "
    "include/perfmon/perfmon_abi.h -- build and install perfmon/core (T11) "
    "for this ROCm first.")
endif()
find_library(_perfmon_core_lib
  NAMES perfmon_core
  PATHS "${PERFMON_CORE_ROOT}/lib" "${PERFMON_CORE_ROOT}/lib64"
  NO_DEFAULT_PATH)
if(NOT _perfmon_core_lib)
  message(FATAL_ERROR
    "[perfmon_flash] No libperfmon_core.so found under "
    "${PERFMON_CORE_ROOT}/lib or ${PERFMON_CORE_ROOT}/lib64.")
endif()

# Same ROCM_PATH convention as perfmon/core/CMakeLists.txt and AOTriton's
# own top-level CMakeLists.txt.
if(DEFINED ENV{ROCM_PATH})
  set(_perfmon_rocm_path "$ENV{ROCM_PATH}")
else()
  message(WARNING "[perfmon_flash] ROCM_PATH is not set, falling back to /opt/rocm. "
                   "Set ROCM_PATH to select a specific ROCm/theRock installation.")
  set(_perfmon_rocm_path "/opt/rocm")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_perfmon_rocm_path}/lib/cmake" "${_perfmon_rocm_path}")
find_package(hip REQUIRED)

# ROCm's own runtime lib dirs, for the installed RPATH below. `hip::host`
# supplies these as BUILD-tree link paths only, and BUILD_WITH_INSTALL_RPATH
# (set below) deliberately discards those -- so without naming them here the
# installed `bin/runner` cannot find `libamdhip64.so`, which is precisely
# what T13's own Verify step ("resolves with NO LD_LIBRARY_PATH set")
# checks. `rocm_sysdeps`/`host-math` are theRock's private renamed copies of
# liblzma/openblas etc.; .ci/common-vars.sh's own add_rocm_sdk_ldconfig()
# adds exactly these two to LD_LIBRARY_PATH for the same reason, and putting
# them on the RPATH is that workaround made permanent for these binaries.
set(_perfmon_rocm_rpath "${_perfmon_rocm_path}/lib")
foreach(_sysdep rocm_sysdeps host-math)
  if(IS_DIRECTORY "${_perfmon_rocm_path}/lib/${_sysdep}/lib")
    list(APPEND _perfmon_rocm_rpath "${_perfmon_rocm_path}/lib/${_sysdep}/lib")
  endif()
endforeach()

# Library name is deliberately just "perfmon_flash" (-> libperfmon_flash.so),
# not something trying to bake "@<subject>" into the filename: this whole
# CMake project is reconfigured and rebuilt fresh per subject (see this
# file's own header comment), each into that subject's own build/install
# tree (perfmon/subjects/<id>/...), so the subject identity already lives in
# the DIRECTORY, exactly like perfmon_core's own per-ROCm install prefix
# disambiguates it without a per-ROCm filename suffix (T11's
# CMakeLists.txt). T13's build_subject.sh is responsible for installing this
# into `perfmon/subjects/<id>/lib/` and rpath-linking `bin/runner` against
# it from there.
add_library(perfmon_flash SHARED adapter_v3.cc)

target_include_directories(perfmon_flash PRIVATE
  "${AOTRITON_ROOT}/include"
  "${PERFMON_CORE_ROOT}/include")

target_link_libraries(perfmon_flash PRIVATE
  hip::host
  "${_perfmon_aotriton_lib}"
  "${_perfmon_core_lib}")

# rev0 D4 / adapter_v3.cc's own header: AOTRITON_NS must resolve per-subject
# via <aotriton/config.h> (generated at THAT subject's own configure time,
# living under its own include/), never a hardcoded namespace -- nothing to
# set here, this is just the reason AOTRITON_ROOT's include/ must come
# before any other AOTriton-shaped include path could exist.

# --- RPATH (T13 gap-fix) --------------------------------------------------
# T13's own Verify step requires `ldd perfmon/subjects/<id>/bin/runner |
# grep aotriton` to resolve WITHOUT any LD_LIBRARY_PATH -- i.e. every
# dependency must be found via an RPATH baked into the binaries at install
# time. This file's original T12 commit set none: CMake's default install
# behavior does NOT carry build-tree link paths into the installed
# RPATH unless asked to, so `libperfmon_flash.so` (and, without this fix,
# `bin/runner`) would have installed with no way to find
# `libaotriton*_v2.so` or `libperfmon_core.so` short of a global/env search
# path. Fixed here rather than silently leaving T13's Verify step unable to
# pass on a real machine.
#
# `_perfmon_aotriton_lib`/`_perfmon_core_lib` are absolute paths (from
# file(GLOB ...)/find_library() above), so their directories are exactly
# AOTRITON_ROOT's and PERFMON_CORE_ROOT's own lib dirs -- baked in verbatim
# rather than relying on CMAKE_INSTALL_RPATH_USE_LINK_PATH's own path
# derivation, so this stays correct even if a future CMake version changes
# that automatic behavior.
get_filename_component(_perfmon_aotriton_libdir "${_perfmon_aotriton_lib}" DIRECTORY)
get_filename_component(_perfmon_core_libdir "${_perfmon_core_lib}" DIRECTORY)

set_target_properties(perfmon_flash PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_perfmon_aotriton_libdir};${_perfmon_core_libdir};${_perfmon_rocm_rpath}")

# --- `runner` executable (T13 gap-fill) -----------------------------------
# Links this project's own just-built `perfmon_flash` (which resolves
# `pmon_family_entry()` at link time -- no dlopen, rev0 D4) plus hip::host.
# perfmon_core's symbols (protocol_loop, measure_launch, thermal_snapshot,
# ...) come in transitively through `perfmon_flash`'s own link against
# `_perfmon_core_lib`; linking them again here directly would be redundant
# (and CMake/the linker de-duplicate it harmlessly either way) -- listed
# explicitly regardless, since `main.cc` calls those symbols directly and a
# transitive-only link is a private implementation detail of
# `perfmon_flash` that should not be relied upon.
add_executable(runner
  "${_perfmon_core_src_dir}/main.cc"
  "${_perfmon_core_src_dir}/entry_codec.cc")

target_include_directories(runner PRIVATE
  "${_perfmon_core_src_dir}"
  "${PERFMON_CORE_ROOT}/include")

target_link_libraries(runner PRIVATE
  perfmon_flash
  hip::host
  "${_perfmon_core_lib}")

# `$ORIGIN/../<libdir>` finds this same install's own `perfmon_flash.so`
# (T13 spec: "with the subject's lib dir on rpath") regardless of where the
# subject's install prefix ends up on disk; the two absolute entries find
# AOTriton's and perfmon_core's install locations, which are NOT generally
# siblings of this subject's own install prefix.
set_target_properties(runner PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR};${_perfmon_aotriton_libdir};${_perfmon_core_libdir};${_perfmon_rocm_rpath}")

install(TARGETS perfmon_flash
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS runner
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
