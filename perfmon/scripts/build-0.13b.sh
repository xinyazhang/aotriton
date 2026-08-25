#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim-only AOTriton build for release tag 0.13b, and ONLY that tag. The
# `.ci/` build-helper interface drifted release to release (see sibling
# perfmon/scripts/build-<tag>.sh files for how differently it drifted), so
# per-spec each tag gets its own throwaway script rather than one script
# trying to flex across all of them.
#
# Usage: build-0.13b.sh <src_dir> <install_dir> <arch>
#   <src_dir>     shallow clone of AOTriton at tag 0.13b
#                 (git clone --depth 1 --branch 0.13b). Not modified, except
#                 for an out-of-source build directory created alongside it.
#   <install_dir> destination; on success contains
#                   include/aotriton/           headers
#                   lib/libaotriton*_v2.so      shared library
#   <arch>        AOTriton target arch, e.g. gfx942
#
# --- .ci/ at 0.13b: what exists, what is used, what is not (all read via
#     `git show 0.13b:.ci/...`, not assumed from HEAD or any other tag) -----
#
# USED: .ci/build-release.sh (and the common-vars.sh it sources, only for
#   aotriton_major/minor). At this tag it is the one helper whose interface
#   already matches this task's contract with NO override tricks needed:
#     build-release.sh <noimage_mode:ON|OFF> [arch_list|ALL] [-D... extra]
#   and it already:
#     - reads AOTRITON_BUILD_PATH / AOTRITON_INSTALL_PATH env vars for the
#       cmake build dir / install prefix, both caller-controlled and both
#       free to be anywhere (exactly what an arbitrary <install_dir> needs);
#     - unconditionally passes -DAOTRITON_NO_PYTHON=ON and
#       -DAOTRITON_GPU_BUILD_TIMEOUT=0 (both wanted here);
#     - does NOT hardcode `-fuse-ld=mold` (mold is not installed in the
#       perfmon container, so its absence here is a feature, not a gap);
#     - forwards extra args verbatim to cmake, which is where
#       -DAOTRITON_NAME_SUFFIX=pmon is added below (build-release.sh itself
#       never sets a suffix, so there is nothing to override).
#   One wrinkle: at this tag .ci/build-release.sh is committed WITHOUT the
#   executable bit (100644, checked with `git ls-tree 0.13b .ci/`), so it is
#   invoked below via `bash .ci/build-release.sh`, not executed directly.
#
# NOT USED, deliberately: .ci/build-shim.sh + .ci/common-build.sh -- the
#   pairing this task's own background material (HEAD's old
#   perfmon/build_subject.sh Step 2) described. At 0.13b, common-build.sh's
#   `common_build()` hardcodes all three things this contract needs to
#   control:
#     - AOTRITON_NAME_SUFFIX="123" (need "pmon")
#     - the build dir as `<src_dir>/../build-<M.m>-<build_for>-<arch>`,
#       i.e. INSIDE the source tree, not caller-controlled
#     - the install prefix as `./install_dir` under that build dir, again
#       not caller-controlled
#   and build-shim.sh further hardcodes `-fuse-ld=mold` on top of that
#   (mold absent here). All three are fightable via cmake's last-`-D`-wins
#   argument order, but that is exactly the "hardcoded choices you need to
#   override" case where the task calls for sourcing common-build.sh
#   directly or writing the cmake invocation by hand instead -- and
#   build-release.sh needs neither, so it is used in its place.
#
# --- What 0.13b's CMakeLists.txt / v3src/CMakeLists.txt actually require
#     (both read in full at this tag) --------------------------------------
#
# 1. Arch is `AOTRITON_TARGET_ARCH` (CACHE STRING, semicolon-separated gfx
#    identifiers) at this tag. `TARGET_GPUS` also exists but only as an
#    OBSOLETE marker -- CMakeLists.txt FATAL_ERRORs if it is set to anything
#    other than the literal string "OBSOLETE" -- so it must never be passed.
#    build-release.sh already forwards its arch argument as
#    `-DAOTRITON_TARGET_ARCH=...`; this script does not touch TARGET_GPUS.
#
# 2. `third_party/triton` is a real git submodule (.gitmodules), but every
#    place CMakeLists.txt and v3src/CMakeLists.txt touch it (building the
#    venv's triton wheel; the aiter git clone; the HSACO
#    compile/AKS2/packaging blocks) is gated `if(NOT AOTRITON_NOIMAGE_MODE)`.
#    With AOTRITON_NOIMAGE_MODE=ON (passed by build-release.sh's first
#    argument, "ON", below), none of that runs, so a `--depth 1`,
#    non-recursive clone (submodule uninitialized) is sufficient -- no
#    network fetch of Triton or aiter is attempted. `modules/` (the tuning
#    database) is a plain tracked directory at this tag (`git ls-tree`
#    shows mode 040000/tree, not 160000/commit), not a submodule, so there
#    is nothing to init there either.
#
# 3. NOT skipped by NOIMAGE_MODE / NO_PYTHON: CMakeLists.txt unconditionally
#    creates a throwaway venv under the build dir and does
#    `pip install -r requirements.txt`, then `pip install <src_dir>` to make
#    this tag's own python/ codegen package (`aotriton.generate` etc.)
#    importable -- both at configure time, both needing network access to
#    PyPI (available on this build node per spec). AOTRITON_NO_PYTHON=ON
#    only skips pybind11/libtorch discovery and the `bindings/`
#    subdirectory (the Python *binding*, unrelated to the codegen package).
#    `-DPYTHON_EXECUTABLE=/usr/bin/python3.11`, which build-release.sh
#    itself hardcodes, is inert: nothing in this tag's CMake files reads the
#    legacy `PYTHON_EXECUTABLE` variable (grepped in full) -- Python
#    discovery goes through `find_package(Python3 ...)`/`Python3_EXECUTABLE`
#    instead, so that stale path not existing on this container costs
#    nothing.
#
# 4. `find_package(hip REQUIRED)` additionally does
#    `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")` UNCONDITIONALLY. That path
#    does not exist under TheRock's venv-based ROCm 7.14 install -- harmless
#    on its own (one more, unused, search path entry) but it means
#    find_package(hip) has nothing to fall back on unless ROCM_PATH is
#    handed to it too. build-release.sh already does this FOR us
#    (`-DCMAKE_PREFIX_PATH="${ROCM_PATH}"`, reading the env var, falling
#    back to `hipconfig --rocmpath` if unset) -- but this script requires
#    ROCM_PATH to already be set (below) rather than trusting that
#    hipconfig fallback, since TheRock's container may not ship a
#    traditional `hipconfig` at all. This -- ROCM_PATH resolution -- is the
#    part of this tag's CMakeLists.txt most likely to break against a
#    non-/opt/rocm layout if mishandled.
#
# 5. `pkg_search_module(LZMA REQUIRED liblzma)` (CMakeLists.txt) needs
#    pkg-config plus liblzma's .pc file (liblzma-dev on Debian) on the build
#    node. Neither is in this task's confirmed toolchain list
#    (cmake/ninja/git/build-essential) -- flagged, not worked around here,
#    since this script cannot install packages on the container's behalf.
#
# 6. `AOTRITON_NAME_SUFFIX` (CMakeLists.txt) renames both the install
#    namespace and, per v3src/CMakeLists.txt's
#    `set_target_properties(aotriton_v2 PROPERTIES OUTPUT_NAME
#    "aotriton${AOTRITON_NAME_SUFFIX}_v2")`, the output library file --
#    suffix "pmon" yields libaotritonpmon_v2.so, matching this task's
#    required `lib/libaotriton*_v2.so` glob. This is the reason the suffix
#    matters at all: it keeps this shim build from colliding with whatever
#    AOTriton PyTorch itself bundles.
#
# --- Verified but NOT exercised: there is no ROCm/hipcc on this machine.
#     Checked with `bash -n`, and by re-reading every cmake option name this
#     script (and build-release.sh) relies on against 0.13b's own
#     CMakeLists.txt / v3src/CMakeLists.txt. Not actually built.

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <src_dir> <install_dir> <arch>" >&2
  echo "  <src_dir>     shallow clone of AOTriton at tag 0.13b" >&2
  echo "  <install_dir> destination for include/ and lib/" >&2
  echo "  <arch>        AOTriton target arch, e.g. gfx942" >&2
  exit 1
fi

SRC_DIR="$1"
INSTALL_DIR="$2"
ARCH="$3"

if [ ! -f "${SRC_DIR}/.ci/build-release.sh" ]; then
  echo "Error: ${SRC_DIR}/.ci/build-release.sh not found -- is <src_dir> really an AOTriton 0.13b checkout?" >&2
  exit 1
fi

if [ -z "${ROCM_PATH:-}" ]; then
  echo "Error: ROCM_PATH is not set. .ci/build-release.sh falls back to" >&2
  echo "       'hipconfig --rocmpath' when ROCM_PATH is unset, but that is" >&2
  echo "       not trusted to exist/behave the same way under TheRock; export" >&2
  echo "       ROCM_PATH (as /etc/profile.d/perfmon.sh already does in the" >&2
  echo "       perfmon container) before calling this script." >&2
  exit 1
fi

# Out-of-source build. Overridable so a caller building several archs from
# the same clone in parallel (or re-running this script) can control /
# isolate the build directory; the default is scoped by arch so at least
# two concurrent invocations against the same src_dir don't collide.
BUILD_DIR="${AOTRITON_BUILD_PATH:-${TMPDIR:-/tmp}/aotriton-0.13b-shim-${ARCH}}"

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

echo "[build-0.13b] src_dir=${SRC_DIR}" >&2
echo "[build-0.13b] build_dir=${BUILD_DIR}" >&2
echo "[build-0.13b] install_dir=${INSTALL_DIR}" >&2
echo "[build-0.13b] arch=${ARCH}" >&2

# build-release.sh reads these two out of the environment; it sets
# everything else this task's flag list asks for
# (AOTRITON_NOIMAGE_MODE via its first arg, AOTRITON_NO_PYTHON,
# AOTRITON_GPU_BUILD_TIMEOUT=0, -DCMAKE_PREFIX_PATH=$ROCM_PATH) on its own.
# Exported as CXX, which is what cmake reads. build-release.sh sets no
# CMAKE_CXX_COMPILER of its own, so this is the only lever on the compiler it
# ends up using; exporting AOTRITON_CXX alone would be inert here.
export CXX="${AOTRITON_CXX:-g++}"
export AOTRITON_BUILD_PATH="${BUILD_DIR}"
export AOTRITON_INSTALL_PATH="${INSTALL_DIR}"

# Export ROCm in CMAKE_PREFIX_PATH as well.
#
# build-release.sh already passes -DCMAKE_PREFIX_PATH="${ROCM_PATH}", and that
# -D cannot be removed from here -- it lives in the tag's own script. But the
# cache variable it sets is the same one the project then does
# `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")` to, so it is one `set()` away
# from being clobbered. CMake also consults $ENV{CMAKE_PREFIX_PATH} as an
# INDEPENDENT search path that project code cannot overwrite, which is what
# makes this belt-and-braces rather than duplication.
#
# Prepended, not replaced: an existing value is the caller's.
export CMAKE_PREFIX_PATH="${ROCM_PATH}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

# Not executed directly: this file is committed without the executable bit
# at this tag (see header comment).
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
bash "${SRC_DIR}/.ci/build-release.sh" ON "${ARCH}" \
  -DAOTRITON_NAME_SUFFIX=pmon

if [ ! -d "${INSTALL_DIR}/include/aotriton" ]; then
  echo "[build-0.13b] ERROR: install finished but ${INSTALL_DIR}/include/aotriton is missing." >&2
  exit 1
fi

if ! compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so" > /dev/null; then
  echo "[build-0.13b] ERROR: install finished but no ${INSTALL_DIR}/lib/libaotriton*_v2.so was produced." >&2
  exit 1
fi

echo "[build-0.13b] ok: $(compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so")" >&2
