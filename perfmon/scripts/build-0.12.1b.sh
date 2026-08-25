#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim-only AOTriton build for release tag 0.12.1b ONLY. The .ci/ interface
# (common-build.sh's function signature, build-shim.sh's fixed flags, cache
# variable names) drifts across AOTriton tags, so per perfmon-exec0.md this
# gets its own frozen script instead of a shared one. Do not reuse this file
# for any other tag without re-reading that tag's own CMakeLists.txt/.ci/.
#
# Usage: build-0.12.1b.sh <src_dir> <install_dir> <arch>
#   <src_dir>     shallow clone (git clone --depth 1 --branch 0.12.1b) of
#                 https://github.com/ROCm/aotriton.git
#   <install_dir> where headers/lib get installed
#   <arch>        AOTriton target arch, e.g. gfx942
#
# --- What was verified against 0.12.1b (read via `git show 0.12.1b:...`,
#     never `git checkout`) before writing this ---
#
# * Arch cache var: `AOTRITON_TARGET_ARCH` (CMakeLists.txt:~114). NOT
#   TARGET_GPUS -- that var still exists but is hard-FATAL_ERROR'd unless left
#   at its literal default string "OBSOLETE" (CMakeLists.txt: `if(NOT
#   TARGET_GPUS STREQUAL "OBSOLETE") message(FATAL_ERROR ...)`), so this
#   script must never set it. `AOTRITON_OVERRIDE_TARGET_GPUS` is a separate,
#   unrelated further-narrowing knob this script also leaves untouched.
#
# * `.ci/` at this tag DOES have both common-build.sh and build-shim.sh, and
#   both were read in full. Deliberately NOT sourced/called here:
#     - `build-shim.sh <target arch>` hardcodes
#       `-DAOTRITON_NOIMAGE_MODE=ON -DAOTRITON_GPU_BUILD_TIMEOUT=0` plus
#       `-fuse-ld=mold` linker flags, then calls `common_build "$1" "shim"
#       "$@"` (extra args forwarded, but build-shim.sh's own CLI takes only
#       one positional arg, so it cannot itself be handed AOTRITON_NO_PYTHON
#       or a name suffix without editing it).
#     - `common_build()` (common-build.sh) forwards into `_common_build
#       Release "123" "$@"` -- note the literal `"123"`: at THIS tag the
#       name suffix is HARDCODED inside common-build.sh itself, not exposed
#       as an env-var override (no AOTRITON_NAME_SUFFIX_OVERRIDE hook exists
#       here -- that is a different tag's convention, confirmed by reading
#       this tag's common-build.sh in full). `_common_build` also hardcodes
#       both the build dir (`${SCRIPT_DIR}/../build-<ver>-shim-<arch>`,
#       i.e. INSIDE src_dir) and the install dir (`./install_dir` under that
#       build dir, relative) with no prefix override hook either.
#   Reusing them here would mean either mutating src_dir (build dir placement)
#   or relying on cmake's "last -D wins" behavior to re-override
#   AOTRITON_NAME_SUFFIX/CMAKE_INSTALL_PREFIX after common-build.sh has
#   already set them once -- fragile and exactly the kind of hidden coupling
#   the task calls out. Writing the cmake invocation directly, mirroring
#   build-shim.sh's own flags plus the two overrides it cannot express
#   (AOTRITON_NO_PYTHON, AOTRITON_NAME_SUFFIX, an arbitrary install prefix),
#   is the more faithful "reuse" here. The `-fuse-ld=mold` linker flags are
#   intentionally dropped: they are a build-speed optimization, not a
#   correctness requirement, and mold is not guaranteed present in the build
#   container described for this script.
#
# * NOIMAGE_MODE + submodules: this tag's ONLY git submodule is
#   `third_party/triton` (.gitmodules). The entire Triton build block in the
#   top-level CMakeLists.txt is wrapped in `if(AOTRITON_NOIMAGE_MODE) ...
#   skip ... else() ... build triton ... endif()`, and v3src/CMakeLists.txt's
#   external `aiter` clone (a second, non-submodule, runtime `git clone`) is
#   separately gated `if(NOT AOTRITON_NOIMAGE_MODE)`. So with
#   AOTRITON_NOIMAGE_MODE=ON, a --depth 1 clone with NO submodules
#   initialized configures cleanly -- no submodule init, no extra network
#   clone, needed.
#
# * Network still needed for one thing regardless of NOIMAGE_MODE/NO_PYTHON:
#   the top-level CMakeLists unconditionally creates a build-local venv and
#   runs `pip install -r requirements.txt` (filelock, numpy, pandas, pyyaml,
#   pybind11, etc. -- no torch) before it ever checks AOTRITON_NOIMAGE_MODE or
#   AOTRITON_NO_PYTHON. The runtime environment this script targets has
#   network available, so this is left alone rather than worked around.
#
# * AOTRITON_NO_PYTHON=ON skips `find_package(Python3 ... Development)`,
#   pybind11/libtorch detection, and `add_subdirectory(bindings)` -- exactly
#   the C++-only shim this script wants, avoiding any libtorch dependency.
#
# * Compiler: `project(AOTriton CXX C ASM)`; the shared library's sources are
#   plain .cc/.cpp (v3src/CMakeLists.txt: aux_source_directory + generated
#   shim files), so `CMAKE_CXX_COMPILER=hipcc` (this repo's convention) is
#   sufficient -- no separate HIP language toggle exists at this tag.
#
# * find_package(hip REQUIRED) is preceded by a hardcoded
#   `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")`. In this script's actual
#   runtime (TheRock ROCm 7.14 installed in a venv, not at /opt/rocm),
#   `find_package(hip)` would plausibly fail to find hip-config.cmake unless
#   ROCM_PATH is also fed into CMAKE_PREFIX_PATH -- so this script passes
#   $ROCM_PATH via the CMAKE_PREFIX_PATH environment variable
#   keeps /opt/rocm as a secondary candidate, so this is additive, not a
#   removal of the tag's own behavior.
#
# * Install layout confirmed by reading v3src/CMakeLists.txt's install()
#   rules directly (not assumed):
#     - `install(DIRECTORY .../include/aotriton DESTINATION
#       ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})` ->
#       <install_dir>/include/aotriton/ (GNUInstallDirs default
#       CMAKE_INSTALL_INCLUDEDIR is "include" on Debian, not distro-varying
#       the way lib/lib64 is).
#     - `install(TARGETS aotriton_v2 ... DESTINATION
#       ${CMAKE_INSTALL_PREFIX}/lib)` -- hardcoded "lib", NOT
#       CMAKE_INSTALL_LIBDIR, so this lands at <install_dir>/lib/ regardless
#       of any lib64 default. OUTPUT_NAME is set to
#       "aotriton${AOTRITON_NAME_SUFFIX}_v2" when AOTRITON_NAME_SUFFIX is
#       non-empty, i.e. libaotritonpmon_v2.so(.0.12.1) with this script's
#       AOTRITON_NAME_SUFFIX=pmon -- matches the contract's
#       lib/libaotriton*_v2.so glob.
#     - the kernel-images install() is itself gated
#       `if(NOT AOTRITON_NOIMAGE_MODE)`, so a shim build never produces
#       lib/aotriton.images/ -- expected and fine, this is a shim-only build.
#
# --- Things NOT verified here (could not build/test on this machine) ---
#   Whether `find_package(hip REQUIRED)` actually resolves against TheRock's
#   venv layout with only CMAKE_PREFIX_PATH=$ROCM_PATH set (vs. needing
#   hip_DIR or an additional theRock-specific CMAKE_MODULE_PATH entry), and
#   whether the build-local `python3 -m venv` step succeeds when the
#   discovered Python3 interpreter is itself inside another venv
#   (theRock's). Both are properties of the tag's/environment's build system
#   this script cannot exercise without hipcc/ROCm present.

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: build-0.12.1b.sh <src_dir> <install_dir> <arch>" >&2
  echo "  <src_dir>     shallow clone of AOTriton at tag 0.12.1b" >&2
  echo "  <install_dir> install prefix (will contain include/, lib/)" >&2
  echo "  <arch>        AOTriton target arch, e.g. gfx942" >&2
  exit 1
fi

SRC_DIR="$1"
INSTALL_DIR="$2"
ARCH="$3"

if [ ! -f "${SRC_DIR}/CMakeLists.txt" ]; then
  echo "Error: ${SRC_DIR}/CMakeLists.txt not found -- is <src_dir> really an" >&2
  echo "       AOTriton checkout?" >&2
  exit 1
fi

# Resolve to absolute paths: we are about to hand these to an out-of-source
# cmake invocation, and a relative INSTALL_DIR would end up interpreted
# relative to BUILD_DIR (cmake's cwd), not the caller's.
SRC_DIR="$(cd "${SRC_DIR}" && pwd)"
mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Out-of-source build tree, under TMPDIR rather than anywhere near src_dir or
# install_dir -- nothing in this tag's shim-build path requires an in-source
# build; build-shim.sh's own in-source placement is an artifact of its
# hardcoded relative paths, not a real requirement (see header comment).
#
# It used to be "${INSTALL_DIR}.build-...", which is a sibling of <install_dir>
# but still INSIDE the subject directory that sync_workdir.sh --workload
# perfmon rsyncs to every GPU worker. A build tree is not a product and has no
# business being shipped; keep it out of that subtree entirely.
BUILD_DIR="${AOTRITON_BUILD_PATH:-${TMPDIR:-/tmp}/aotriton-0.12.1b-shim-${ARCH}}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

AOTRITON_CXX="${AOTRITON_CXX:-g++}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

echo "[build-0.12.1b] src_dir=${SRC_DIR}" >&2
echo "[build-0.12.1b] install_dir=${INSTALL_DIR}" >&2
echo "[build-0.12.1b] build_dir=${BUILD_DIR}" >&2
echo "[build-0.12.1b] arch=${ARCH}" >&2
echo "[build-0.12.1b] CXX=${AOTRITON_CXX} ROCM_PATH=${ROCM_PATH}" >&2

# Shim-only configure: C++ runtime only, no Triton, no kernel images, no
# python bindings. Flags mirror .ci/build-shim.sh's own choice at this tag
# (NOIMAGE_MODE, GPU_BUILD_TIMEOUT=0) plus the two overrides build-shim.sh's
# fixed CLI cannot express (NO_PYTHON, NAME_SUFFIX) -- see header comment for
# why this is a direct cmake invocation rather than sourcing common-build.sh.
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
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${AOTRITON_CXX}" \
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

# Verify the contract this script promises to its caller, rather than trust
# a zero exit status from cmake alone.
# --- kernel images --------------------------------------------------------
# The shim built above is deliberately NOIMAGE: the kernels under test must be
# the ones this release actually shipped, not ones rebuilt now. They come from
# the release itself.
. "$(dirname "${BASH_SOURCE[0]}")/lib/release_asset.sh"

# Which images asset serves this arch. The grouping is release-specific --
# gfx11xx split into gfx110x and gfx115x at this release, and gfx1250
# appeared. +asan variants also exist and are never selected.
case "${ARCH%%:*}" in
    gfx90a) IMAGES_GROUP="gfx90a" ;;
    gfx942) IMAGES_GROUP="gfx942" ;;
    gfx950) IMAGES_GROUP="gfx950" ;;
    gfx1250) IMAGES_GROUP="gfx1250" ;;
    gfx1100) IMAGES_GROUP="gfx110x" ;;
    gfx1101) IMAGES_GROUP="gfx110x" ;;
    gfx1150) IMAGES_GROUP="gfx115x" ;;
    gfx1151) IMAGES_GROUP="gfx115x" ;;
    gfx1200) IMAGES_GROUP="gfx120x" ;;
    gfx1201) IMAGES_GROUP="gfx120x" ;;
    *)
      echo "Error: AOTriton 0.12.1b publishes no kernel-image asset covering" >&2
      echo "       '${ARCH}'. See https://github.com/ROCm/aotriton/releases/tag/0.12.1b" >&2
      exit 1
      ;;
esac

ASSET="aotriton-0.12.1b-images-amd-${IMAGES_GROUP}.tar.gz"
# Under BUILD_DIR, never under INSTALL_DIR: the download is scratch, and
# everything below the subject dir is rsynced to the GPU workers.
IMAGES_DL_DIR="${BUILD_DIR}/images-download"

TARBALL_NAME="$(fetch_release_asset "0.12.1b" "${ASSET}" "${IMAGES_DL_DIR}")"
install_images_from_tarball "${IMAGES_DL_DIR}/${TARBALL_NAME}" "${INSTALL_DIR}"

if [ ! -d "${INSTALL_DIR}/include/aotriton" ]; then
  echo "[build-0.12.1b] ERROR: ${INSTALL_DIR}/include/aotriton missing after install." >&2
  exit 1
fi

shopt -s nullglob
libs=("${INSTALL_DIR}"/lib/libaotriton*_v2.so)
shopt -u nullglob
if [ "${#libs[@]}" -eq 0 ]; then
  echo "[build-0.12.1b] ERROR: no ${INSTALL_DIR}/lib/libaotriton*_v2.so after install." >&2
  exit 1
fi

echo "[build-0.12.1b] ok: ${libs[*]}" >&2
