#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim-only AOTriton build for tag 0.10b ONLY. Every perfmon subject script
# under perfmon/scripts/ is pinned to one tag: the `.ci/` build interface
# drifted across AOTriton releases (cmake option names come and go, some tags
# have no `.ci/` at all), so there is no single script that reuses across
# tags without silently mis-invoking an older or newer one. This file must
# work for 0.10b; it is not meant to, and need not, work for any other tag.
#
# Usage: build-0.10b.sh <src_dir> <install_dir> <arch>
#   <src_dir>     shallow clone of AOTriton at tag 0.10b
#                 (git clone --depth 1 --branch 0.10b ...). Read-only: this
#                 script never writes into it, the cmake build is entirely
#                 out-of-source (see BUILD_DIR below).
#   <install_dir> install prefix. On success contains:
#                   include/aotriton/   (headers)
#                   lib/libaotriton*_v2.so
#   <arch>        AOTriton target arch, e.g. gfx942. Forwarded verbatim as
#                 -DAOTRITON_TARGET_ARCH.
#
# --- .ci/ drift, considered not missed --------------------------------------
# `git ls-tree 0.10b .ci/` is EMPTY: this tag ships no `.ci/` directory at
# all, so there is no `common-build.sh`/`build-shim.sh` to source (contrast
# later tags, where build_subject.sh sources the tag's own
# `.ci/common-build.sh`). The cmake invocation below is therefore hand-written,
# derived directly from THIS tag's own `CMakeLists.txt` (root) and
# `v3src/CMakeLists.txt` — both read in full — rather than copied from any
# `.ci/` helper on this branch (which targets a different, later tag and uses
# option names/behavior not verified to exist at 0.10b).
#
# --- cmake options used, and why these exact names at 0.10b -----------------
# Root CMakeLists.txt at 0.10b defines:
#   AOTRITON_TARGET_ARCH          CACHE STRING, semicolon-separated arch list.
#                                  This is the arch knob at this tag -- NOT
#                                  `TARGET_GPUS`, which the same file marks
#                                  OBSOLETE and turns into a hard
#                                  message(FATAL_ERROR) the moment it is set
#                                  to anything but the literal string
#                                  "OBSOLETE". Do not pass -DTARGET_GPUS=...
#                                  here, ever.
#   AOTRITON_NOIMAGE_MODE          option(), default OFF. Skips the Triton
#                                  build (see below) and the GPU kernel
#                                  compile/package steps in v3src/CMakeLists.txt.
#   AOTRITON_NO_PYTHON             option(), default OFF. Skips the pybind11
#                                  subdirectory and the `bindings/` subdir.
#   AOTRITON_GPU_BUILD_TIMEOUT     CACHE STRING, default "8.0". Unused once
#                                  NOIMAGE_MODE=ON (no kernels get compiled),
#                                  but it exists at this tag so it is passed
#                                  anyway, matching the task's desired flags.
#   AOTRITON_NAME_SUFFIX           CACHE STRING, default "". Produces
#                                  libaotriton<SUFFIX>_v2.so and turns on
#                                  AOTRITON_ENABLE_SUFFIX internally -- this is
#                                  what keeps this .so from colliding with the
#                                  AOTriton PyTorch bundles.
#
# --- submodules / Triton: NOT needed for this configuration -----------------
# A `--depth 1` clone has no submodules initialized (.gitmodules at 0.10b
# lists third_party/triton, third_party/incbin, third_party/pybind11). Neither
# is required for a NOIMAGE_MODE + NO_PYTHON shim build:
#   * third_party/triton is only touched by the `if(NOT AOTRITON_NOIMAGE_MODE)`
#     block in the root CMakeLists.txt (builds libtriton.so into the cmake
#     venv). AOTRITON_NOIMAGE_MODE=ON skips that block entirely.
#   * third_party/pybind11 is only touched by the `if(NOT AOTRITON_NO_PYTHON)`
#     block (add_subdirectory + a FATAL_ERROR check that it was initialized).
#     AOTRITON_NO_PYTHON=ON skips that block entirely.
#   * third_party/incbin is not referenced by any .cc/.h/.cmake/CMakeLists.txt
#     at this tag (grepped in full) -- it is dead weight for this build either
#     way.
# So no `git submodule update` and no extra network fetch for submodules is
# needed; this script does not attempt one.
#
# --- what DOES still need network, even in NOIMAGE_MODE ---------------------
# The root CMakeLists.txt creates a python venv under the build dir and runs
# `pip install -r requirements.txt` UNCONDITIONALLY (this happens before the
# NOIMAGE_MODE check, not gated by it). requirements.txt at 0.10b is small
# (filelock, iniconfig, packaging, pluggy, numpy, setuptools, wheel, pybind11,
# pandas) and pulls in neither triton nor torch, but it does need PyPI
# reachable from the build node. The runtime this script targets has network,
# so this is left as-is rather than worked around.
#
# --- things plausibly fragile against ROCm 7.14 / this environment ----------
# * v3src/CMakeLists.txt does `pkg_search_module(LZMA REQUIRED liblzma)` for
#   Kernel Storage V2's xz/LZMA use. This needs pkg-config plus a liblzma
#   pkg-config file (liblzma-dev on Debian) on the build node; neither was
#   listed among this environment's confirmed packages, so this script
#   preflights it with a clear error rather than letting cmake fail deep
#   inside pkg_search_module with a less actionable message.
# * The root CMakeLists.txt does
#   `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")` unconditionally, then
#   `find_package(hip REQUIRED)`. TheRock's ROCm 7.14 here lives under
#   $ROCM_PATH inside a venv (_rocm_sdk_devel), not /opt/rocm, so this script
#   exports $ROCM_PATH in the CMAKE_PREFIX_PATH environment variable
#   CMakeLists only adds to it, it does not replace it, so both entries
#   coexist and hip is found via the real one).
# * find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED) plus a nested
#   `python3 -m venv` inside the build dir assumes the python3 cmake's
#   find_package resolves is >=3.10 and has a working venv module (needs
#   ensurepip). Not independently verified in this exact container image;
#   flagged rather than silently assumed.
#
# --- generator / compiler ----------------------------------------------------
# CXX defaults to hipcc, ROCM_PATH is read from the environment, matching the
# rest of this repo's build scripts (see .ci/build-release.sh). Neither is
# hardcoded to a specific install path.

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "build-0.10b.sh requires Bash." >&2
  exit 1
fi

if [ "$#" -ne 3 ]; then
  echo "Usage: build-0.10b.sh <src_dir> <install_dir> <arch>" >&2
  echo "  <src_dir>     shallow clone of AOTriton at tag 0.10b" >&2
  echo "  <install_dir> cmake install prefix" >&2
  echo "  <arch>        AOTriton target arch, e.g. gfx942" >&2
  exit 1
fi

SRC_DIR="$1"
INSTALL_DIR="$2"
ARCH="$3"

if [ -z "${ARCH}" ]; then
  echo "Error: empty <arch>." >&2
  exit 1
fi

if [ ! -f "${SRC_DIR}/CMakeLists.txt" ] || [ ! -f "${SRC_DIR}/v3src/CMakeLists.txt" ]; then
  echo "Error: ${SRC_DIR} does not look like an AOTriton source tree" >&2
  echo "       (missing CMakeLists.txt or v3src/CMakeLists.txt)." >&2
  exit 1
fi
SRC_DIR="$(cd "${SRC_DIR}" && pwd)"

mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

CXX="${CXX:-hipcc}"
if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "Error: C++ compiler '${CXX}' not found on PATH." >&2
  echo "       This build expects hipcc (\$CXX overrides it)." >&2
  exit 1
fi

# ROCM_PATH: read from the environment (this repo's convention, see
# .ci/build-release.sh), with a hipconfig fallback for parity with that
# script. Root CMakeLists.txt hardcodes /opt/rocm into CMAKE_PREFIX_PATH via
# list(APPEND), which does not replace whatever we pass here -- it only adds
# to it -- so pointing find_package(hip) at the real ROCm root is safe even
# though that hardcoded entry stays in the list too.
if [ -z "${ROCM_PATH:-}" ]; then
  if command -v hipconfig >/dev/null 2>&1; then
    ROCM_PATH="$(hipconfig --rocmpath 2>/dev/null || true)"
  fi
fi
if [ -z "${ROCM_PATH:-}" ]; then
  echo "Error: ROCM_PATH is not set and hipconfig --rocmpath did not find it." >&2
  echo "       This build needs a ROCm install (find_package(hip) at 0.10b)." >&2
  exit 1
fi

# pkg_search_module(LZMA REQUIRED liblzma) in v3src/CMakeLists.txt fails deep
# inside cmake configure with a much less actionable message than this.
if ! command -v pkg-config >/dev/null 2>&1 || ! pkg-config --exists liblzma; then
  echo "Error: pkg-config cannot find 'liblzma' (needed by v3src/CMakeLists.txt's" >&2
  echo "       Kernel Storage V2 xz/LZMA use). Install pkg-config and a liblzma" >&2
  echo "       development package (e.g. liblzma-dev on Debian) and retry." >&2
  exit 1
fi

echo "[build-0.10b] src_dir=${SRC_DIR}" >&2
echo "[build-0.10b] install_dir=${INSTALL_DIR}" >&2
echo "[build-0.10b] arch=${ARCH}" >&2
echo "[build-0.10b] CXX=${CXX}" >&2
echo "[build-0.10b] ROCM_PATH=${ROCM_PATH}" >&2

# Out-of-source build, entirely outside <src_dir> -- the contract only asks
# that this script may create build dirs INSIDE src_dir "if that tag's build
# system demands it"; 0.10b's cmake does not, so keep the clone untouched and
# build in a scratch dir instead. Cleaned up on success; left in place (path
# printed) on failure so a human can inspect the cmake/ninja logs.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aotriton-0.10b-${ARCH}-XXXXXX")"
cleanup() {
  local status=$?
  if [ "${status}" -eq 0 ]; then
    rm -rf "${BUILD_DIR}"
  else
    echo "[build-0.10b] FAILED -- build tree left at ${BUILD_DIR} for inspection." >&2
  fi
}
trap cleanup EXIT

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

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX}" \
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
  echo "Error: build finished but ${INSTALL_DIR}/include/aotriton is missing." >&2
  exit 1
fi
if ! compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so" >/dev/null; then
  echo "Error: build finished but no ${INSTALL_DIR}/lib/libaotriton*_v2.so was produced." >&2
  exit 1
fi

echo "[build-0.10b] ok: ${INSTALL_DIR}" >&2
