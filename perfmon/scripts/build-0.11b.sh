#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim-only AOTriton build for tag 0.11b ONLY. Every AOTriton release tag
# gets its own build-<tag>.sh because the .ci/ build interface drifts across
# releases (see perfmon/build_subject.sh's header, disclosure #1) -- this
# script need not, and does not try to, work for any tag but 0.11b.
#
# Usage: build-0.11b.sh <src_dir> <install_dir> <arch>
#   <src_dir>     existing `git clone --depth 1 --branch 0.11b` of AOTriton.
#                 Read-only: this script builds out-of-source in a scratch
#                 dir, never inside <src_dir>.
#   <install_dir> install prefix. On success this holds:
#                   include/aotriton/         headers
#                   lib/libaotritonpmon_v2.so*  the shim shared library
#   <arch>        AOTriton target arch, e.g. gfx942. Forwarded verbatim to
#                 -DAOTRITON_TARGET_ARCH.
#
# --- What "shim-only" means here, and why the flags below -----------------
# -DAOTRITON_NOIMAGE_MODE=ON   build the C++ runtime shim only; no GPU kernel
#                              images. At this tag NOIMAGE_MODE only skips the
#                              third_party/triton build (root CMakeLists.txt's
#                              `if(AOTRITON_NOIMAGE_MODE) ... else()` around
#                              the triton build) and, downstream in
#                              v3src/CMakeLists.txt, the HSACO kernel-compile
#                              step and the `install(DIRECTORY
#                              .../aotriton.images ...)` rule (guarded by
#                              `NOT AOTRITON_NOIMAGE_MODE`). It does NOT skip
#                              `v3python.generate`, the codegen step that
#                              produces the shim .cc files themselves -- that
#                              still runs, via the venv this configure step
#                              creates (see below).
# -DAOTRITON_NO_PYTHON=ON     skip the Python *binding* (pybind11 module).
#                              This is what lets a --depth 1 clone (no
#                              submodules) configure: with it ON, root
#                              CMakeLists.txt never runs
#                              `find_package(Python3 ... Development)` or
#                              `add_subdirectory(third_party/pybind11)`, and
#                              never `add_subdirectory(bindings)`.
# -DAOTRITON_GPU_BUILD_TIMEOUT=0   no-op with NOIMAGE_MODE=ON (nothing times
#                              out because nothing compiles GPU kernels), but
#                              it is a real, valid cache var at this tag
#                              (root CMakeLists.txt), passed for parity with
#                              every other tag's script.
# -DAOTRITON_NAME_SUFFIX=pmon  root CMakeLists.txt turns this into
#                              AOTRITON_ENABLE_SUFFIX; v3src/CMakeLists.txt
#                              renames the target's OUTPUT_NAME to
#                              "aotriton${SUFFIX}_v2", so the installed
#                              library becomes libaotritonpmon_v2.so(.0.11.0)
#                              -- matching lib/libaotriton*_v2.so and keeping
#                              this out of PyTorch's own bundled AOTriton's
#                              way.
#
# --- Arch knob: AOTRITON_TARGET_ARCH, not TARGET_GPUS ----------------------
# Root CMakeLists.txt at this tag defines AOTRITON_TARGET_ARCH as the real
# knob (a CACHE STRING list of arches to build for) and separately declares
# TARGET_GPUS as "OBSOLETE" -- if TARGET_GPUS is ever set to anything else,
# configure hits `message(FATAL_ERROR "TARGET_GPUS is OBSOLETE...")`. So this
# script sets only -DAOTRITON_TARGET_ARCH=<arch> and never touches
# TARGET_GPUS/AOTRITON_OVERRIDE_TARGET_GPUS.
#
# --- Submodules: none needed for this exact flag combination ---------------
# A --depth 1 clone has third_party/ empty. At this tag that is fine as long
# as both NOIMAGE_MODE=ON and NO_PYTHON=ON are set together:
#   - third_party/triton is only touched in the `else()` branch reached when
#     AOTRITON_NOIMAGE_MODE is OFF (root CMakeLists.txt) -- NOIMAGE_MODE=ON
#     skips it entirely.
#   - third_party/pybind11 is only touched inside `if(NOT AOTRITON_NO_PYTHON)`
#     (root CMakeLists.txt) -- NO_PYTHON=ON skips it entirely, and that
#     branch is also the only place a FATAL_ERROR would fire demanding
#     `git submodule update --init`.
# Verified by reading this tag's CMakeLists.txt in full
# (`git show 0.11b:CMakeLists.txt`); no other add_subdirectory() at this tag
# reaches into third_party/.
#
# --- Not submodule-related, but still network-touching ---------------------
# Regardless of NOIMAGE_MODE/NO_PYTHON, root CMakeLists.txt unconditionally
# creates a venv under the build dir and pip-installs this tag's
# requirements.txt (numpy, pandas, pybind11 the pip package -- unrelated to
# the third_party/pybind11 *submodule* -- wheel, setuptools, ...) into it at
# CONFIGURE time. That venv's python is what later runs `v3python.generate`
# to produce the shim .cc files. This needs network access during `cmake`,
# not just during the actual compile.
#
# --- .ci/ at this tag, and why it is NOT reused here -----------------------
# `git ls-tree --name-only 0.11b .ci/` shows: common-build.sh, common-vars.sh,
# build-release.sh, build-debug.sh, build-tune.sh, build-test.sh,
# build-for-torch.sh, dockerscript-setup-repo.sh, releasesuite-git-head.sh,
# run-ci-test.sh, run-test.sh, base.Dockerfile, rocm.Dockerfile,
# source.Dockerfile, torch-build.sh, README.md. There is NO build-shim.sh
# (that arrived in a later tag).
#
# common-build.sh DOES exist, but its `common_build()` interface does not fit
# this script's contract, read in full (`git show 0.11b:.ci/common-build.sh`):
#   function common_build() {  # exactly 3 positional args
#     if [ "$#" -ne 3 ]; then ...; fi
#     _common_build "$@" Release
#   }
#   function _common_build() {
#     target_arch="$1"; build_type="$2"; cmake_option0="$3"   # ONE cmake flag
#     bdir="build-${aotriton_major}.${aotriton_minor}-${build_type}-${target_arch}"
#     mkdir -p ${SCRIPT_DIR}/../${bdir}      # hardcoded, INSIDE the src tree
#     cd ${SCRIPT_DIR}/../${bdir}; cmake .. \
#       -DCMAKE_INSTALL_PREFIX=./install_dir \    # hardcoded, relative, not
#                                                  # this script's <install_dir>
#       -DAOTRITON_NAME_SUFFIX=123 \               # hardcoded to "123", not
#                                                   # overridable to "pmon"
#       -G Ninja; ninja install/strip
#   }
# Three mismatches, each fatal to reuse: (1) it accepts exactly one cmake
# option string, not the multi-flag set this script needs; (2) the install
# prefix is a hardcoded relative path, not this script's caller-supplied
# <install_dir>; (3) AOTRITON_NAME_SUFFIX is hardcoded to "123", and there is
# no hook to override it to "pmon". It also builds in-source
# (${SCRIPT_DIR}/../build-...), where this script prefers out-of-source.
# Rather than patching around all three from outside (there is no env-var
# escape hatch here to patch with -- that arrived in a later tag, compare
# perfmon/build_subject.sh's disclosure #1 about AOTRITON_BUILD_PATH /
# AOTRITON_INSTALL_PATH), this script does not source common-build.sh at
# all and issues the cmake invocation directly below. common-vars.sh (only
# used to compute aotriton_major/minor and an unrelated llvm_hash for
# common-build.sh) is likewise unused.
#
# --- Compiler / ROCm location ----------------------------------------------
# CXX defaults to hipcc, per this repo's convention, but is left overridable
# via the environment. ROCM_PATH is passed into CMAKE_PREFIX_PATH so
# `find_package(hip REQUIRED)` can locate TheRock's ROCm install; root
# CMakeLists.txt separately does `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")`,
# which is harmless here whether or not /opt/rocm exists on this box, since
# it is appended after (not instead of) the path this script supplies.
#
# --- Known risks NOT covered by anything above (could not verify here) ----
# - liblzma-dev + pkg-config: root CMakeLists.txt does
#   `pkg_search_module(LZMA REQUIRED liblzma)` unconditionally (kernel storage
#   v2 uses xz). Not part of this script's own contract to install; if the
#   build node's image lacks it, configure fails there.
# - python3-venv/ensurepip: the venv creation above needs it; not otherwise
#   verified present on this box.
# This script was written and statically checked (`bash -n`, and cmake option
# names re-read one by one against `git show 0.11b:CMakeLists.txt` and
# `git show 0.11b:v3src/CMakeLists.txt`) on a machine with no ROCm/hipcc, so
# it has not actually been run.

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <src_dir> <install_dir> <arch>" >&2
  exit 1
fi

SRC_DIR="$1"
INSTALL_DIR="$2"
ARCH="$3"

if [ ! -f "${SRC_DIR}/CMakeLists.txt" ]; then
  echo "[build-0.11b] ERROR: ${SRC_DIR}/CMakeLists.txt not found -- is <src_dir> an AOTriton checkout?" >&2
  exit 1
fi
SRC_DIR="$(cd "${SRC_DIR}" && pwd)"

mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Out-of-source build in a scratch dir, never inside <src_dir> (that tree is
# read-only as far as this script is concerned). Cleaned up on exit either
# way -- nothing under <install_dir> depends on it surviving.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aotriton-0.11b-build.XXXXXX")"
trap 'rm -rf "${BUILD_DIR}"' EXIT

echo "[build-0.11b] src_dir=${SRC_DIR}" >&2
echo "[build-0.11b] install_dir=${INSTALL_DIR}" >&2
echo "[build-0.11b] arch=${ARCH}" >&2
echo "[build-0.11b] build_dir=${BUILD_DIR}" >&2

# CMAKE_PREFIX_PATH is exported as an ENVIRONMENT variable, not passed as -D.
#
# Every tag in range hardcodes `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")`
# before find_package(hip). A -D cache value survives that particular APPEND,
# but it is the same variable the project is manipulating, so it is only ever
# one `set()` away from being clobbered. CMake consults $ENV{CMAKE_PREFIX_PATH}
# as an INDEPENDENT search path that project code cannot overwrite -- so ROCm
# stays findable wherever it actually lives without patching the old tag.
#
# Prepended, not replaced: an existing value in the environment is the
# caller's, and dropping it would be its own surprise.
export CMAKE_PREFIX_PATH="${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# GCC, not hipcc, builds the AOTriton library.
#
# hipcc is clang, and clang makes -Wc++11-narrowing an ERROR where GCC does
# not; this tag's own v2src/flash/attn_fwd.cc narrows int32_t to uint32_t in an
# initializer list and simply does not compile under it. That is not a bug to
# work around here: AOTriton's own CI never used hipcc for this. Neither
# .ci/build-release.sh nor .ci/common-build.sh sets CMAKE_CXX_COMPILER at any
# tag in range, so cmake picks the platform default -- g++ on Debian -- and
# that is what these releases were built and tested with.
#
# The shim needs only HIP's HOST API, which hip::host supplies to any C++
# compiler; nothing here compiles device code. (perfmon/core is the opposite
# case and does need hipcc -- it has a __global__ kernel in fill.cc.)
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${AOTRITON_CXX:-g++}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DAOTRITON_TARGET_ARCH="${ARCH}" \
  -DAOTRITON_NOIMAGE_MODE=ON \
  -DAOTRITON_NO_PYTHON=ON \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  -DAOTRITON_NAME_SUFFIX=pmon

# Build AND install in ONE `ninja install`, never `cmake --build` followed by
# `cmake --install`.
#
# This tag's own README.md says so explicitly: "do not run `ninja` separately,
# due to the limit of the current build system, `ninja install` will run the
# whole build process unconditionally." Splitting the two -- which is exactly
# what cmake --build then cmake --install does -- is the documented way to
# trip that bug. The note is present in the README of every tag in range.
ninja -C "${BUILD_DIR}" install

if [ ! -d "${INSTALL_DIR}/include/aotriton" ]; then
  echo "[build-0.11b] ERROR: build reported success but ${INSTALL_DIR}/include/aotriton is missing." >&2
  exit 1
fi
if ! compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so" >/dev/null; then
  echo "[build-0.11b] ERROR: build reported success but no ${INSTALL_DIR}/lib/libaotriton*_v2.so was installed." >&2
  exit 1
fi

echo "[build-0.11b] ok: ${INSTALL_DIR}" >&2
