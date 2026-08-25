#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim-only AOTriton build for tag 0.11.2b ONLY. Every AOTriton release tag
# gets its own build-<tag>.sh because the `.ci/` build interface drifts across
# releases (see perfmon/build_subject.sh's header) -- this script need not,
# and does not try to, work for any tag but 0.11.2b.
#
# Usage: build-0.11.2b.sh <src_dir> <install_dir> <arch>
#   <src_dir>     existing `git clone --depth 1 --branch 0.11.2b` of AOTriton.
#                 Read-only: this script builds out-of-source in a scratch
#                 dir, never inside <src_dir>.
#   <install_dir> install prefix. On success this holds:
#                   include/aotriton/          headers
#                   lib/libaotritonpmon_v2.so*  the shim shared library
#   <arch>        AOTriton target arch, e.g. gfx942. Forwarded verbatim to
#                 -DAOTRITON_TARGET_ARCH.
#
# --- What "shim-only" means here, and why the flags below -----------------
# -DAOTRITON_NOIMAGE_MODE=ON  build the C++ runtime shim only; no GPU kernel
#                             images, no Triton. At this tag NOIMAGE_MODE
#                             skips the third_party/triton build (root
#                             CMakeLists.txt's `if(AOTRITON_NOIMAGE_MODE) ...
#                             else()` around the triton venv-install) and,
#                             downstream in v3src/CMakeLists.txt, the whole
#                             HSACO kernel-compile / kernel-storage block
#                             (`if(NOT AOTRITON_NOIMAGE_MODE) ... endif()`)
#                             plus the `install(DIRECTORY .../aotriton.images
#                             ...)` rule (same guard). It does NOT skip
#                             `v3python.generate`, the codegen step that
#                             writes the shim's own .cc/.h files -- that step
#                             sits outside the NOIMAGE_MODE guard and always
#                             runs, via the venv root CMakeLists.txt creates
#                             (see below).
# -DAOTRITON_NO_PYTHON=ON    skip the Python *binding* (pybind11 module).
#                             With this ON, root CMakeLists.txt never runs
#                             `find_package(Python3 ... Development)`, never
#                             `add_subdirectory(third_party/pybind11)`
#                             (the block that would otherwise
#                             message(FATAL_ERROR) demanding
#                             `git submodule update --init`), never probes
#                             for libtorch, and v3src/CMakeLists.txt's own
#                             `add_subdirectory(bindings)` is skipped too.
#                             This is what makes a --depth 1, non-recursive
#                             clone sufficient.
# -DAOTRITON_GPU_BUILD_TIMEOUT=0  a real CACHE STRING at this tag (root
#                             CMakeLists.txt, default "8.0" minutes, fed to
#                             the per-kernel compiler's --timeout). Inert
#                             under NOIMAGE_MODE=ON (nothing compiles a GPU
#                             kernel), passed anyway for parity with every
#                             other tag's script and in case that ever
#                             changes.
# -DAOTRITON_NAME_SUFFIX=pmon  root CMakeLists.txt turns a non-empty suffix
#                             into AOTRITON_ENABLE_SUFFIX=ON;
#                             v3src/CMakeLists.txt then does
#                             `set_target_properties(aotriton_v2 PROPERTIES
#                             OUTPUT_NAME "aotriton${AOTRITON_NAME_SUFFIX}_v2")`,
#                             so the installed library becomes
#                             libaotritonpmon_v2.so(.0.11.2) -- matching
#                             lib/libaotriton*_v2.so and keeping this build
#                             out of the way of the AOTriton PyTorch bundles.
#
# --- Arch knob: AOTRITON_TARGET_ARCH, not TARGET_GPUS ----------------------
# Root CMakeLists.txt at this tag defines AOTRITON_TARGET_ARCH as a CACHE
# STRING list of arches to build (default is every supported gfx*, e.g.
# "gfx90a;gfx942;gfx950;..."), and separately pins
# `set(TARGET_GPUS "OBSOLETE" ...)`; if TARGET_GPUS is ever set to anything
# else, configure hits `message(FATAL_ERROR "TARGET_GPUS is OBSOLETE in
# Dispatcher V3. Use AOTRITON_TARGET_ARCH or AOTRITON_OVERRIDE_TARGET_GPUS.")`.
# There is also AOTRITON_OVERRIDE_TARGET_GPUS (default ""), which further
# restricts AOTRITON_TARGET_ARCH's list in v3src/CMakeLists.txt via
# `v3python.gpu_targets`; left at its default here since a single
# -DAOTRITON_TARGET_ARCH="<arch>" already selects exactly one arch. This
# script therefore sets only -DAOTRITON_TARGET_ARCH=<arch> and never touches
# TARGET_GPUS.
#
# --- Submodules: none needed for this exact flag combination ---------------
# .gitmodules at this tag lists third_party/{triton,incbin,pybind11,aiter}.
# A --depth 1 clone leaves third_party/ empty. With NOIMAGE_MODE=ON and
# NO_PYTHON=ON together:
#   - third_party/triton is only touched in the branch reached when
#     AOTRITON_NOIMAGE_MODE is OFF (root CMakeLists.txt) -- skipped entirely.
#   - third_party/pybind11 is only add_subdirectory()'d inside
#     `if(NOT AOTRITON_NO_PYTHON)` (root CMakeLists.txt) -- skipped entirely.
#   - third_party/incbin and third_party/aiter are not referenced by any
#     CMakeLists.txt/*.cmake at this tag at all (`git grep -n
#     "incbin\|aiter" 0.11.2b -- '*.txt' '*.cmake'` is empty).
# Verified by reading root CMakeLists.txt (254 lines, in full) and
# v3src/CMakeLists.txt (in full) at this tag; no add_subdirectory() other
# than the two guarded ones above reaches into third_party/.
#
# --- Not submodule-related, but still network-touching ---------------------
# Regardless of NOIMAGE_MODE/NO_PYTHON, root CMakeLists.txt unconditionally
# creates a venv under the build dir (`${Python3_EXECUTABLE} -m venv`) and
# pip-installs this tag's requirements.txt (numpy, pandas, pyyaml, the
# pybind11 *pip package* -- unrelated to the third_party/pybind11 git
# submodule -- wheel, setuptools, ...) into it at CONFIGURE time. That
# venv's python is what v3src/CMakeLists.txt later invokes to run
# `v3python.generate`, the codegen step mentioned above. This needs a
# `python3` >= 3.10 on PATH (AOTRITON_MIN_PYTHON) and network access to PyPI
# during `cmake`, not just during the compile step -- both are available on
# this build node per spec.
#
# --- .ci/ at this tag, and why build-shim.sh/common-build.sh are NOT reused
# `git ls-tree --name-only 0.11.2b .ci/` shows: common-build.sh,
# common-vars.sh, build-shim.sh, build-tune.sh, build-test.sh,
# build-release.sh, build-debug.sh, build-for-torch.sh, build-altwheels.sh,
# include-altwheel.sh, dockerscript-setup-repo.sh, releasesuite-git-head.sh,
# run-ci-test.sh, run-test.sh, torch-build.sh, base/rocm/source.Dockerfile,
# 0.11.1b.yaml, README.md. Unlike some earlier tags, THIS tag has both
# common-build.sh and build-shim.sh -- but reading them (`git show
# 0.11.2b:.ci/common-build.sh`, `git show 0.11.2b:.ci/build-shim.sh`) in full
# turns up three reasons not to call them as-is:
#
#   1. AOTRITON_NAME_SUFFIX is hardcoded, and not to "pmon". common-build.sh:
#        function common_build() { _common_build Release "123" "$@"; }
#        function _common_build() {
#          build_type="$1"; suffix="$2"; target_arch="$3"; build_for="$4"
#          ...
#          cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir \
#            -DCMAKE_BUILD_TYPE=${build_type} \
#            -DAOTRITON_TARGET_ARCH=${target_arch} \
#            -DAOTRITON_NAME_SUFFIX=${suffix} "$@" -G Ninja
#          ninja install/strip
#        }
#      and build-shim.sh calls `common_build "$1" "shim" -DAOTRITON_NOIMAGE_MODE=ON ...`.
#      Positionally, "$1" (our arch) becomes target_arch and the literal
#      string "shim" becomes build_for (just a build-dir name component) --
#      neither reaches AOTRITON_NAME_SUFFIX, which common_build() has already
#      pinned to the literal "123" before "$@" is even seen. There is no
#      parameter to override it.
#   2. build-shim.sh never sets AOTRITON_NO_PYTHON, so root CMakeLists.txt's
#      default (`option(AOTRITON_NO_PYTHON ... OFF)`) applies: it would
#      attempt the pybind11 submodule + libtorch path this script exists to
#      avoid, and fail outright since --depth 1 leaves third_party/pybind11
#      empty.
#   3. CMAKE_INSTALL_PREFIX is hardcoded to the relative `./install_dir`
#      under a build directory _common_build creates at
#      `${SCRIPT_DIR}/../build-${major}.${minor}-shim-<arch>` -- i.e. inside
#      <src_dir> -- with no hook for this script's caller-supplied absolute
#      <install_dir>. build-shim.sh also adds
#      `-DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold"` /
#      `-DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold"`; mold is not in this
#      task's confirmed toolchain (cmake/ninja/git/build-essential), so
#      reusing that flag as-is would add a gratuitous failure mode.
#
#   Separately, common-build.sh sources common-vars.sh, which runs under
#   `set -ex` and unconditionally does
#   `native_arch=$(rocm_agent_enumerator|grep -v gfx000|head -n 1)` (and
#   again for ngpus). rocm_agent_enumerator is not something this task's
#   verified toolchain list promises, and neither value is even used by the
#   shim build -- so merely sourcing common-build.sh risks an unrelated,
#   avoidable failure (or an exit before our own cmake invocation runs) for
#   no benefit.
#
#   Given all of that, this script neither sources common-build.sh nor
#   invokes build-shim.sh; it issues the cmake configure/build/install
#   sequence directly, the same shape common_build() itself would produce
#   once corrected for suffix, NO_PYTHON, install prefix and linker flags.
#
# --- Compiler / ROCm location ----------------------------------------------
# CXX defaults to hipcc, per this repo's convention, but stays overridable
# via the environment. ROCM_PATH is passed into CMAKE_PREFIX_PATH so
# `find_package(hip REQUIRED)` can locate TheRock's ROCm install; root
# CMakeLists.txt separately does
# `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")`, which is harmless here
# whether or not /opt/rocm exists on this box, since it is appended after
# (not instead of) the path this script supplies.
#
# --- Known risks NOT covered by anything above (could not verify here) -----
# - liblzma-dev + pkg-config: root CMakeLists.txt does
#   `pkg_search_module(LZMA REQUIRED liblzma)` unconditionally (Kernel
#   Storage V2 uses xz/LZMA even in NOIMAGE_MODE, since that code path is
#   compiled into the shim regardless). Not part of this script's own
#   contract to install; if the build node's image lacks it, configure fails
#   there with a pkg-config error, not a cmake option error.
# - python3-venv/ensurepip and PyPI reachability for the unconditional
#   `pip install -r requirements.txt` described above.
# This script was written and statically checked (`bash -n`, and every cmake
# option name re-read against `git show 0.11.2b:CMakeLists.txt` and
# `git show 0.11.2b:v3src/CMakeLists.txt`) on a machine with no ROCm/hipcc,
# so it has not actually been run.

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <src_dir> <install_dir> <arch>" >&2
  exit 1
fi

SRC_DIR="$1"
INSTALL_DIR="$2"
ARCH="$3"

if [ ! -f "${SRC_DIR}/CMakeLists.txt" ]; then
  echo "[build-0.11.2b] ERROR: ${SRC_DIR}/CMakeLists.txt not found -- is <src_dir> an AOTriton checkout?" >&2
  exit 1
fi
SRC_DIR="$(cd "${SRC_DIR}" && pwd)"

mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Out-of-source build in a scratch dir, never inside <src_dir> (that tree is
# read-only as far as this script is concerned). Cleaned up on exit either
# way -- nothing under <install_dir> depends on it surviving.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aotriton-0.11.2b-build.XXXXXX")"
trap 'rm -rf "${BUILD_DIR}"' EXIT

echo "[build-0.11.2b] src_dir=${SRC_DIR}" >&2
echo "[build-0.11.2b] install_dir=${INSTALL_DIR}" >&2
echo "[build-0.11.2b] arch=${ARCH}" >&2
echo "[build-0.11.2b] build_dir=${BUILD_DIR}" >&2

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
  echo "[build-0.11.2b] ERROR: build reported success but ${INSTALL_DIR}/include/aotriton is missing." >&2
  exit 1
fi
if ! compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so" >/dev/null; then
  echo "[build-0.11.2b] ERROR: build reported success but no ${INSTALL_DIR}/lib/libaotriton*_v2.so was installed." >&2
  exit 1
fi

echo "[build-0.11.2b] ok: ${INSTALL_DIR}" >&2
