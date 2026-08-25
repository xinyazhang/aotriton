#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shim-only AOTriton build for release tag 0.9.2b, and ONLY that tag. The
# `.ci/` build-helper interface (common-build.sh, build-shim.sh, ...) drifted
# heavily release to release, so per-spec each tag gets its own throwaway
# script rather than one script trying to flex across all of them.
#
# Usage: build-0.9.2b.sh <src_dir> <install_dir> <arch>
#   <src_dir>     shallow clone of AOTriton at tag 0.9.2b
#                 (git clone --depth 1 --branch 0.9.2b). Not modified, except
#                 for an out-of-source build directory created alongside it.
#   <install_dir> destination; on success contains
#                   include/aotriton/           headers
#                   lib/libaotriton*_v2.so      shared library
#   <arch>        AOTriton target arch, e.g. gfx942
#
# --- Why this can't reuse .ci/ -----------------------------------------
# `git ls-tree 0.9.2b .ci/` is EMPTY: this tag predates the .ci/ directory
# entirely, so there is no common-build.sh / build-shim.sh to source here.
# The cmake invocation below is hand-written from THIS tag's own
# CMakeLists.txt and v2src/CMakeLists.txt (read in full), not copied from a
# later tag's .ci/ scripts or from perfmon/build_subject.sh's `head` path.
#
# --- What 0.9.2b's build system actually looks like (verified by reading
#     CMakeLists.txt / v2src/CMakeLists.txt / v2python/gpu_targets.py at this
#     tag, not assumed) ---------------------------------------------------
#
# 1. Source layout at this tag is v2src/ + v2python/ (the "v3src"/"v3python"
#    split is later history). CMakeLists.txt add_subdirectory(v2src)'s.
#
# 2. Arch is NOT `AOTRITON_TARGET_ARCH` at this tag (that option does not
#    exist yet). It is `TARGET_GPUS`, a semicolon-separated list of AMD
#    *trade names* -- CMakeLists.txt:49 literally comments
#    "Note here uses Trade names" -- e.g. MI300X, not gfx942. The mapping
#    gfx-arch -> trade name lives in v2python/gpu_targets.py
#    (AOTRITON_SUPPORTED_GPUS) and is reproduced in map_arch() below. This is
#    NOT a naming-convention violation of this project's own
#    "architecture identifier only" rule -- it is 0.9.2b's own upstream cmake
#    surface accepting only trade names; this script translates the
#    gfx-identifier interface (this task's fixed contract) to whatever that
#    old cmake actually understands, same as it would translate any other
#    version-specific option name.
#
# 3. NOIMAGE_MODE genuinely skips Triton at this tag: `third_party/triton` is
#    only touched from CMakeLists.txt under `if(NOT AOTRITON_NOIMAGE_MODE)`
#    (building the venv's triton wheel), and v2src/CMakeLists.txt's kernel
#    compile/clustering block is gated the same way. `third_party/pybind11`
#    is only add_subdirectory()'d under `if(NOT AOTRITON_NO_PYTHON)`.
#    `third_party/incbin` is not referenced by any CMakeLists.txt at this tag
#    at all. So with -DAOTRITON_NOIMAGE_MODE=ON -DAOTRITON_NO_PYTHON=ON (both
#    passed below), NO submodule under third_party/ needs to be initialized
#    -- a --depth 1, non-recursive clone (uninitialized submodules) is
#    sufficient, matching this task's <src_dir> contract.
#
# 4. NOT skipped by NOIMAGE_MODE: the top-level CMakeLists.txt
#    unconditionally (regardless of NOIMAGE_MODE/NO_PYTHON) creates its own
#    nested venv under the build dir (`python3 -m venv`) and
#    `pip install -r requirements.txt` into it, purely for the Python-side
#    shim/header generator (v2python.generate_shim, run even in shim-only
#    builds -- v2src/CMakeLists.txt's generate_shim execute_process is
#    OUTSIDE the `if(NOT AOTRITON_NOIMAGE_MODE)` block). This needs a
#    `python3` >= 3.10 on PATH (AOTRITON_MIN_PYTHON) and network access to
#    PyPI for that pip install; both are available in this build node per
#    spec.
#
# 5. `find_package(hip REQUIRED)` at the top of CMakeLists.txt additionally
#    does `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")` UNCONDITIONALLY. That
#    path does not exist under TheRock's venv-based ROCm 7.14 install --
#    harmless on its own (just one more, unused, search path entry) but it
#    means find_package(hip) has nothing to fall back on if this script
#    doesn't also hand it ${ROCM_PATH}. We pass
#    $ROCM_PATH via the CMAKE_PREFIX_PATH env var for that reason -- this
#    the part of this tag's CMakeLists.txt most likely to break against a
#    non-/opt/rocm ROCm layout if we didn't.
#
# 6. `pkg_search_module(LZMA REQUIRED liblzma)` (CMakeLists.txt) needs
#    pkg-config plus liblzma's pkg-config file (liblzma-dev on Debian) on
#    the build node. Neither is in this task's confirmed toolchain list
#    (cmake/ninja/git/build-essential) -- flagged, not worked around here,
#    since this script cannot install packages on your behalf.
#
# 7. `AOTRITON_GPU_BUILD_TIMEOUT` (CMakeLists.txt:47) exists as a plain cache
#    STRING (default "8.0", minutes) fed to the per-kernel compiler
#    `--timeout` flag; it is inert under NOIMAGE_MODE (nothing gets compiled)
#    but passing =0 costs nothing and matches the requested flag set.
#
# 8. `AOTRITON_NAME_SUFFIX` (CMakeLists.txt:63) renames both the install
#    namespace and, per v2src/CMakeLists.txt's
#    `set(AOTRITON_LIBRARY_FILE libaotriton${AOTRITON_NAME_SUFFIX}_v2.so)`,
#    the output library file -- suffix "pmon" yields libaotritonpmon_v2.so,
#    matching this task's required `lib/libaotriton*_v2.so` glob.
#
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <src_dir> <install_dir> <arch>" >&2
  echo "  <src_dir>     shallow clone of AOTriton at tag 0.9.2b" >&2
  echo "  <install_dir> destination for include/ and lib/" >&2
  echo "  <arch>        AOTriton target arch, e.g. gfx942" >&2
  exit 1
fi

SRC_DIR="$1"
INSTALL_DIR="$2"
ARCH="$3"

if [ ! -f "${SRC_DIR}/CMakeLists.txt" ]; then
  echo "Error: ${SRC_DIR}/CMakeLists.txt not found -- is <src_dir> a checkout of AOTriton?" >&2
  exit 1
fi

if [ -z "${ROCM_PATH:-}" ]; then
  echo "Error: ROCM_PATH is not set. This tag's CMakeLists.txt only searches" >&2
  echo "       /opt/rocm on its own (hardcoded, unconditional); without" >&2
  echo "       ROCM_PATH pointing find_package(hip) at TheRock's install," >&2
  echo "       configure will fail with 'Could NOT find hip'." >&2
  exit 1
fi

# TARGET_GPUS at 0.9.2b takes trade names, not gfx identifiers -- see header
# comment item 2.
#
# The mapping is READ FROM THE TAG'S OWN v2python/gpu_targets.py at run time,
# not transcribed into this file. Two reasons, and the first is the binding
# one: CLAUDE.md forbids this repo from naming products by anything but their
# architecture identifier, and says only manual editing by the user may
# introduce an alternative name. Deriving the table from the old tag's own
# source keeps those names where they already are -- in that release -- rather
# than authoring them here. Second, it cannot drift: whatever that tag
# actually accepts is what we pass, with no transcription step to get wrong.
map_arch() {
  local bare="${1%%:*}"
  local table="${SRC_DIR}/v2python/gpu_targets.py"

  if [ ! -f "${table}" ]; then
    echo "Error: ${table} not found; cannot resolve the TARGET_GPUS name for" \
         "'${1}' at this tag." >&2
    return 1
  fi

  python3 - "${table}" "${bare}" <<'PYEOF'
import ast, pathlib, re, sys

table, gfx = sys.argv[1], sys.argv[2]
text = pathlib.Path(table).read_text()

# AOTRITON_GPU_ARCH_TUNING_STRING maps the build system's own gpu name to the
# gfx identifier; invert it. literal_eval rather than importing the module, so
# nothing in that tag's python package has to be importable here.
m = re.search(r"AOTRITON_GPU_ARCH_TUNING_STRING\s*=\s*(\{.*?\})", text, re.S)
if not m:
    sys.exit(f"no AOTRITON_GPU_ARCH_TUNING_STRING in {table}")

by_gfx = {v: k for k, v in ast.literal_eval(m.group(1)).items()}
if gfx not in by_gfx:
    sys.exit(f"{gfx} is not supported at this tag; it knows: "
             + " ".join(sorted(by_gfx)))
print(by_gfx[gfx])
PYEOF
}

TARGET_GPU="$(map_arch "${ARCH}")"
echo "[build-0.9.2b] arch ${ARCH} -> TARGET_GPUS=${TARGET_GPU}" >&2

# Out-of-source build. Overridable so a caller building several archs from
# the same clone in parallel (or re-running this script) can control /
# isolate the build directory; the default is scoped by arch so at least
# two concurrent invocations against the same src_dir don't collide.
BUILD_DIR="${AOTRITON_BUILD_PATH:-${TMPDIR:-/tmp}/aotriton-0.9.2b-shim-${ARCH}}"

mkdir -p "${BUILD_DIR}" "${INSTALL_DIR}"

# Resolve to absolute paths: we are about to hand these to an out-of-source
# cmake invocation, and a relative INSTALL_DIR would end up interpreted
# relative to BUILD_DIR (cmake's cwd), not the caller's.
SRC_DIR="$(cd "${SRC_DIR}" && pwd)"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

echo "[build-0.9.2b] src_dir=${SRC_DIR}" >&2
echo "[build-0.9.2b] build_dir=${BUILD_DIR}" >&2
echo "[build-0.9.2b] install_dir=${INSTALL_DIR}" >&2

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
  -DCMAKE_CXX_COMPILER="${AOTRITON_CXX:-g++}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DAOTRITON_NOIMAGE_MODE=ON \
  -DAOTRITON_NO_PYTHON=ON \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  -DAOTRITON_NAME_SUFFIX=pmon \
  -DTARGET_GPUS="${TARGET_GPU}"

# Build AND install in ONE `ninja install`, never `cmake --build` followed by
# `cmake --install`.
#
# This tag's own README.md says so explicitly: "do not run `ninja` separately,
# due to the limit of the current build system, `ninja install` will run the
# whole build process unconditionally." Splitting the two -- which is exactly
# what cmake --build then cmake --install does -- is the documented way to
# trip that bug. The note is present in the README of every tag in range.
ninja -C "${BUILD_DIR}" install

# --- kernel images --------------------------------------------------------
# The shim built above is deliberately NOIMAGE: the kernels under test must be
# the ones this release actually shipped, not ones rebuilt now. They come from
# the release itself.
# 0.9.2b ships no SEPARATE images package: the GPU images are inside the one
# jumbo tarball, alongside the runtime, at the same aotriton/lib/aotriton.images
# path the later standalone images packages use. rocm7.0 is the newest variant
# this release offers; the choice is immaterial to the images themselves, which
# are GPU code objects rather than ROCm-linked binaries.
. "$(dirname "${BASH_SOURCE[0]}")/lib/release_asset.sh"
ASSET="aotriton-0.9.2b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz"
IMAGES_DL_DIR="${BUILD_DIR:-${INSTALL_DIR}.build}/images-download"

TARBALL_NAME="$(fetch_release_asset "0.9.2b" "${ASSET}" "${IMAGES_DL_DIR}")"
install_images_from_tarball "${IMAGES_DL_DIR}/${TARBALL_NAME}" "${INSTALL_DIR}"

if [ ! -d "${INSTALL_DIR}/include/aotriton" ]; then
  echo "[build-0.9.2b] ERROR: install finished but ${INSTALL_DIR}/include/aotriton is missing." >&2
  exit 1
fi

if ! compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so" > /dev/null; then
  echo "[build-0.9.2b] ERROR: install finished but no ${INSTALL_DIR}/lib/libaotriton*_v2.so was produced." >&2
  exit 1
fi

echo "[build-0.9.2b] ok: $(compgen -G "${INSTALL_DIR}/lib/libaotriton*_v2.so")" >&2
