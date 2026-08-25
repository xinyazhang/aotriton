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
# is exactly one subject's install prefix -- which is the SUBJECT DIRECTORY
# ITSELF, not a nested aotriton/ inside it: a subject is one ordinary Linux
# install tree (bin/, lib/, include/) shared by AOTriton and perfmon's own
# artifacts. So it has `include/aotriton/*.h` under it and
# `lib/libaotriton*_v2.so` under it (the `*` is load-bearing,
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
    "  cmake -DAOTRITON_ROOT=/path/to/installed/perfmon/<arch>/<tag> "
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

# ROCm's RUNTIME lib dirs, for the installed RPATH below. `hip::host`
# supplies these as BUILD-tree link paths only, and BUILD_WITH_INSTALL_RPATH
# (set below) deliberately discards those -- so without naming them here the
# installed `bin/runner` cannot find `libamdhip64.so`, which is precisely
# what T13's own Verify step ("resolves with NO LD_LIBRARY_PATH set")
# checks. Crucially these are NOT ${ROCM_PATH}/lib on theRock, since
# ROCM_PATH is a build-time package that a measurement node need not have
# installed at all; see PerfmonRocmRuntime.cmake.
include("${_perfmon_core_src_dir}/PerfmonRocmRuntime.cmake")
perfmon_rocm_runtime_dirs(_perfmon_rocm_rpath "${_perfmon_rocm_path}")

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
# --- Which adapter -------------------------------------------------------
# One directory per AOTriton tag: <tag>/adapter.cc, with routines that do
# not depend on the drifting part of the API factored into lib/ (see
# lib/common.h for exactly where that line is drawn).
#
# NOT hardcoded to a single source. rev0 §5 calls the v3 adapter "a starting
# template per API generation; where a tag's structs diverge, that subject
# gets its own copy ... never a branch in a file other subjects also
# compile", and rev0 §4 warns "Do not assume v3 struct stability". Both were
# right, and the drift is finer than rev0's two-adapter table assumed -- the
# tag range spans four generations. Hardcoding one source is what made every
# historical tag compile HEAD's adapter and die in "no member named ..."
# errors that said nothing about the real cause.
#
# Tags whose API is identical are directory SYMLINKS rather than copies
# (0.11b -> 0.11.2b, 0.12.1b -> head, 0.13b -> head), so the two can never
# silently drift apart. When one of them does diverge, replace its symlink
# with a real directory.
if(NOT DEFINED PERFMON_FLASH_TAG)
  message(FATAL_ERROR
    "[perfmon_flash] PERFMON_FLASH_TAG is required: the AOTriton tag this "
    "subject is built from, e.g. -DPERFMON_FLASH_TAG=0.13b. It selects "
    "<tag>/adapter.cc. build_subject.sh passes it automatically.")
endif()

set(_perfmon_adapter "${CMAKE_CURRENT_SOURCE_DIR}/${PERFMON_FLASH_TAG}/adapter.cc")
if(NOT EXISTS "${_perfmon_adapter}")
  message(FATAL_ERROR
    "[perfmon_flash] no adapter for tag '${PERFMON_FLASH_TAG}': "
    "${_perfmon_adapter} does not exist.\n"
    "Every tag needs one, because the public flash API is not stable across "
    "releases -- compiling a neighbouring tag's adapter instead only yields "
    "a wall of 'no member named ...' errors. Write it against that tag's own "
    "include/aotriton/{flash,util}.h, or symlink this directory to an "
    "existing tag whose API you have verified is identical.")
endif()
message(STATUS "[perfmon_flash] tag ${PERFMON_FLASH_TAG} -> ${_perfmon_adapter}")

# lib/ is compiled fresh per subject alongside the adapter -- "shared" means
# shared SOURCE, never a shared binary. The one genuinely shared binary is
# libperfmon_core, which sees no AOTriton header at all (rev0 D4).
add_library(perfmon_flash SHARED
  "${_perfmon_adapter}"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/common.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/vtable.cc")

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
# --- $ORIGIN-relative, because the whole hierarchy relocates --------------
# The earlier version baked ABSOLUTE paths to AOTriton's and perfmon_core's
# lib dirs into both binaries. That works only on the machine that built
# them: subjects are built on the build node under its own workdir and then
# rsynced to GPU workers, whose workdir path differs -- at which point
# `ldd bin/runner` reports `libperfmon_core.so => not found`. Observed
# exactly that way.
#
# So every dependency INSIDE the synced hierarchy is expressed relative to
# $ORIGIN and the whole tree can be moved anywhere as a unit:
#
#     <prefix>/<arch>/<tag>/bin/runner              this executable
#     <prefix>/<arch>/<tag>/lib/                    perfmon_flash + aotriton
#     <prefix>/<arch>/core/rocm-<ver>/lib/          perfmon_core, shared by
#                                                   every subject in the column
#
# perfmon_core deliberately sits OUTSIDE the subject dir -- it is built once
# per ROCm and reused by every subject (rev0 D4), so it cannot live under any
# one of them -- but it is still inside <prefix>/<arch>/, which is what
# sync_workdir.sh ships. `$ORIGIN/../../core/rocm-<ver>/lib` therefore stays
# valid after relocation, which an absolute path does not.
#
# The ROCm runtime dirs stay ABSOLUTE, and that is not an oversight: they are
# genuinely outside this hierarchy and are not shipped with it (see
# PerfmonRocmRuntime.cmake -- on theRock they are not ${ROCM_PATH}/lib, and a
# measurement node need not have the build-time ROCm package at all).
get_filename_component(_perfmon_aotriton_libdir "${_perfmon_aotriton_lib}" DIRECTORY)
get_filename_component(_perfmon_core_libdir "${_perfmon_core_lib}" DIRECTORY)

get_filename_component(_perfmon_install_prefix "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
set(_perfmon_install_libdir "${_perfmon_install_prefix}/${CMAKE_INSTALL_LIBDIR}")
set(_perfmon_install_bindir "${_perfmon_install_prefix}/${CMAKE_INSTALL_BINDIR}")

# Express `target_dir` as an $ORIGIN-relative RPATH entry for a binary that
# will be installed into `anchor_dir`.
function(perfmon_origin_relative out_var anchor_dir target_dir)
  get_filename_component(_a "${anchor_dir}" ABSOLUTE)
  get_filename_component(_t "${target_dir}" ABSOLUTE)
  file(RELATIVE_PATH _rel "${_a}" "${_t}")
  if(_rel STREQUAL "")
    set(${out_var} "$ORIGIN" PARENT_SCOPE)
  else()
    set(${out_var} "$ORIGIN/${_rel}" PARENT_SCOPE)
  endif()
endfunction()

perfmon_origin_relative(_rpath_aotriton_from_lib "${_perfmon_install_libdir}" "${_perfmon_aotriton_libdir}")
perfmon_origin_relative(_rpath_core_from_lib     "${_perfmon_install_libdir}" "${_perfmon_core_libdir}")
perfmon_origin_relative(_rpath_aotriton_from_bin "${_perfmon_install_bindir}" "${_perfmon_aotriton_libdir}")
perfmon_origin_relative(_rpath_core_from_bin     "${_perfmon_install_bindir}" "${_perfmon_core_libdir}")

# With AOTriton installed into the subject prefix itself, its lib dir IS the
# subject's lib dir, so the aotriton entry collapses onto the $ORIGIN one.
# Deduplicate rather than emit it twice -- kept general, because AOTRITON_ROOT
# is still a free parameter and a caller may point it elsewhere.
set(_rpath_flash "$ORIGIN;${_rpath_aotriton_from_lib};${_rpath_core_from_lib};${_perfmon_rocm_rpath}")
set(_rpath_runner
    "$ORIGIN/../${CMAKE_INSTALL_LIBDIR};${_rpath_aotriton_from_bin};${_rpath_core_from_bin};${_perfmon_rocm_rpath}")
list(REMOVE_DUPLICATES _rpath_flash)
list(REMOVE_DUPLICATES _rpath_runner)

message(STATUS "[perfmon_flash] RUNPATH lib/libperfmon_flash.so: ${_rpath_flash}")
message(STATUS "[perfmon_flash] RUNPATH bin/runner: ${_rpath_runner}")

# DT_RUNPATH, not DT_RPATH. RUNPATH is not transitive -- it does NOT help
# resolve a dependency's own dependencies -- which is precisely why
# libperfmon_flash.so needs its own entry for perfmon_core below rather than
# relying on the runner's. Requested explicitly rather than left to the
# linker's default, since that default is a distro build-time choice.
target_link_options(perfmon_flash PRIVATE "LINKER:--enable-new-dtags")

set_target_properties(perfmon_flash PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_rpath_flash}")

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

# `$ORIGIN/../<libdir>` finds this same install's own `perfmon_flash.so` and
# (post-flattening) AOTriton too; the computed entries reach perfmon_core in
# the sibling `core/rocm-<ver>/` column. All relative -- see the RPATH block
# above for why nothing inside the hierarchy may be absolute.
target_link_options(runner PRIVATE "LINKER:--enable-new-dtags")

set_target_properties(runner PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_rpath_runner}")

install(TARGETS perfmon_flash
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS runner
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
