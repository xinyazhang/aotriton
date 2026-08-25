#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# perfmon-exec0.md T13: builds one "subject" -- one AOTriton tag, shim-built
# for one ROCm/arch combination, plus this branch's own perfmon harness
# (libperfmon_flash@<subject> + bin/runner) linked against it.
#
# Usage: build_subject.sh <tag> <rocm> <arch>
#   <tag>   git tag to build, e.g. 0.13b. HEAD is NOT supported: it has no
#           published kernel images, so it would need a full AOTriton build
#           rather than the shim build this script performs. The working-tree
#           path still exists behind PERFMON_ALLOW_HEAD=1 -- it is the path
#           T13 actually exercised -- but no UI reaches it. The released-tag
#           path is implemented per spec but UNVERIFIED, see disclosures.
#   <rocm>  a nominal ROCm version label (perfmon-rev0.md §9:
#           `perfmon::subject::<subject_id>` keys/values, e.g. "7.14.0")
#           used ONLY to build `subject_id` and this script's default
#           PERFMON_CORE_ROOT guess -- it does NOT select a ROCm install by
#           itself. The actual toolchain is located exactly the way every
#           other script in this repo already does it: via the `ROCM_PATH`
#           environment variable (perfmon/core/CMakeLists.txt's and
#           .ci/common-vars.sh's own convention), which the CALLER must set
#           before invoking this script. Inventing a `<rocm>` -> `ROCM_PATH`
#           naming convention that nothing in this repo defines would be
#           guessing at infrastructure this task was not asked to build.
#   <arch>  GPU arch, e.g. "gfx942" -- forwarded verbatim as
#           AOTRITON_TARGET_ARCH.
#
# Produces <workdir>/installed/perfmon/<arch>/<tag>/ containing a
# `subject_id` file (aotriton-<tag>+rocm<rocm>, perfmon-rev0.md §9/§11) and:
#   aotriton/            AOTRITON_ROOT: shim-built include/+lib/, plus
#                         (released tags only) fetched lib/aotriton.images/
#   bin/runner            the executable T13's own Verify step checks
#   lib/libperfmon_flash.so
#
# RUN AND VERIFIED for `head 7.14.0 gfx942` on an 8x gfx942 node with
# theRock ROCm 7.14: T13's Verify step passes (see the runner CMakeLists).
# The <tag> != head path is still UNRUN, as is the `gh release download`
# images fetch in step 3 -- disclosure #5 below stands unchanged.
#
# Disclosure #4 (below) was CONFIRMED in practice: the shim build produces
# no lib/aotriton.images/ even for `head`, so T14 could only run after a
# separate full (non-shim) gfx942 build's images were copied in by hand:
#   .ci/build-test.sh gfx942 <prebuilt-triton-wheel>
#   cp -r build-0.14-test-gfx942/install_dir/lib/aotriton.images \
#         perfmon/subjects/aotriton-head+rocm7.14.0/aotriton/lib/
# Originally verified only via `bash -n` (syntax check) and manual re-reading
# against .ci/build-shim.sh, .ci/common-build.sh, v3src/CMakeLists.txt's
# install() rules, and .ci/runc-manylinux-build-tar.sh's tarball-naming
# convention (all read in full while writing this). See
# perfmon-handoff0.md for the exact commands a human must run on a
# ROCm+GPU machine to actually exercise this end to end.
#
# --- Disclosed design decisions / gaps (none dictated by T13's spec text
#     word-for-word; each is a first-cut choice or a found spec problem) ---
#
# 1. Step 2 ("Shim-only AOTriton build ... read .ci/build-shim.sh and reuse
#    it") is implemented by SOURCING that tag's own .ci/common-build.sh and
#    calling its `common_build` function directly with build-shim.sh's own
#    flags, PLUS `-DAOTRITON_NO_PYTHON=ON` (a real, pre-existing CMake
#    option -- CMakeLists.txt:113 -- not invented here). build-shim.sh's own
#    CLI takes only `<target arch>` and cannot express the extra flag or a
#    per-subject install prefix, so its fixed single-invocation script
#    itself cannot be called as a subprocess; sourcing the function it
#    itself calls is the closest thing to "reuse it, don't write a new cmake
#    invocation" without duplicating cmake's actual argument list by hand.
#    `AOTRITON_NAME_SUFFIX_OVERRIDE=pmon`, `AOTRITON_BUILD_PATH`, and
#    `AOTRITON_INSTALL_PATH` are all pre-existing env-var hooks
#    `common-build.sh` itself already defines (read in full: lines 9-13,
#    24-27) -- not new surface added by this script.
#
# 2. Sourced from the SUBJECT'S OWN tag's worktree (`${SRC_DIR}/.ci/...`),
#    never this repo's own .ci/ -- an old tag's common-build.sh/CMakeLists
#    may differ from this branch's, and the shim build must reflect that
#    tag's own build system, not this branch's.
#
# 3. libperfmon_flash + bin/runner (step 4), by contrast, is ALWAYS
#    configured from THIS repo's own `modules/flash/perfmon/runner/` and
#    `perfmon/core/` -- never from a per-tag worktree. Old AOTriton release
#    tags predate `modules/flash/perfmon/` entirely (it is new code on this
#    branch), so there is nothing to build it FROM in an old tag's tree; the
#    harness source is one thing, built repeatedly against many AOTriton
#    roots, exactly as documented in `modules/flash/perfmon/runner/
#    CMakeLists.txt`'s own "runner executable target" comment. A concrete,
#    KNOWN, DISCLOSED consequence: if a released tag's AOTriton headers/API
#    have drifted from what `adapter_v3.cc` (a "starting template per API
#    generation", perfmon-rev0.md D4) expects, that subject's build will
#    fail to compile until it gets its own `overrides/<tag>.cc` -- explicitly
#    out of scope for T13, whose own title is "Build the first subject:
#    head" (the one case where this cannot happen, since `head` IS this
#    branch). Non-head tags are therefore an implemented-but-unexercised
#    path, more so than the rest of this already-unverified script.
#
# 4. Step 3's `tag == head` case: spec text says "For head, the local build
#    already has them [images]" -- but step 2's shim build always passes
#    `AOTRITON_NOIMAGE_MODE=ON`, and v3src/CMakeLists.txt gates its OWN
#    `install(DIRECTORY .../aotriton.images ...)` on `NOT
#    AOTRITON_NOIMAGE_MODE` (read in full: line 456-459) -- so a shim build,
#    by construction, NEVER has `lib/aotriton.images/` populated, for `head`
#    or any other tag. This is a genuine inconsistency between two bullets
#    of T13's own spec text (item 2's unconditional NOIMAGE_MODE=ON vs. item
#    3's "head already has them"), not a per-task planning gap this script
#    can quietly paper over. Resolution taken here: warn loudly (not fail)
#    when `head`'s images directory is absent after the shim build, since
#    T13's own Verify step (`ldd` + `runner <<< "exit"`) does not exercise
#    `enumerate`/`measure` and therefore does not need images to pass;
#    T14 (out of this task's scope) is exactly where a missing-images subject
#    would be caught for real. Failing hard here would block T13's own
#    stated primary target ("head") over a condition its own Verify step
#    does not test.
#
# 5. Step 3's released-tag path: nothing in this repository's committed CI
#    configuration (`.ci/`, `.github/` -- both grepped in full) actually
#    publishes `aotriton-<sha>-images-<arch>.tar.gz` to GitHub Releases;
#    `.ci/runc-manylinux-build-tar.sh` (lines 116-122) only shows how that
#    exact filename is PRODUCED locally inside a release build container,
#    written to a bind-mounted `/output`, with no further step in this repo
#    that uploads it anywhere `gh release download` could reach. Rather than
#    inventing an unconfirmed publishing pipeline and hardcoding a `gh`
#    invocation as if it were known to work, this script tries `gh release
#    download` as a first-cut, documented BEST EFFORT (disclosed as
#    unverified, since this environment has no network and no `gh`), and
#    provides `PERFMON_IMAGES_TARBALL=<path>` as an explicit escape hatch for
#    a manually-obtained tarball -- so a human on a real machine is never
#    blocked by this script's guess being wrong, only by silence about it.
#
# 6. `PERFMON_CORE_ROOT`: T13's fixed `<tag> <rocm> <arch>` CLI has no slot
#    for it (a genuine gap already disclosed in `modules/flash/perfmon/
#    runner/CMakeLists.txt`'s own header comment). Read as the
#    `PERFMON_CORE_ROOT` env var if set; otherwise defaults to
#    `/opt/perfmon/rocm-<rocm>`, the exact example path
#    `perfmon/core/CMakeLists.txt`'s own header comment already uses for
#    "where T11 installs this ROCm's libperfmon_core". Fails loudly (not
#    silently) if that root turns out not to actually contain a
#    perfmon_core install.

set -euo pipefail

if [ -z "${BASH_VERSION:-}" ]; then
  echo "build_subject.sh requires Bash." >&2
  exit 1
fi

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: build_subject.sh <tag> <rocm> <arch> [workdir]" >&2
  echo '  <tag>:  git tag to build, e.g. 0.13b' >&2
  echo '  <rocm>: nominal ROCm version label (e.g. 7.14.0) -- see this' >&2
  echo '          script'"'"'s own header comment for why this is NOT how' >&2
  echo '          the ROCm toolchain itself is located (set ROCM_PATH)' >&2
  echo '  <arch>: GPU arch, e.g. gfx942 (-> AOTRITON_TARGET_ARCH)' >&2
  echo '  [workdir]: project workdir; the subject installs under' >&2
  echo '          <workdir>/installed/perfmon/<arch>/<tag>/. Defaults to' >&2
  echo '          $PERFMON_WORKDIR.' >&2
  exit 1
fi

TAG="$1"
ROCM="$2"
ARCH="$3"

# HEAD is not a supported subject. A released tag is shim-built and paired
# with that release's prebuilt kernel images; HEAD has no published images, so
# it would need a full AOTriton build (Triton, every kernel) before it could be
# measured -- a different and far more expensive operation than this script
# performs. Refuse rather than half-produce a subject with no images.
#
# PERFMON_ALLOW_HEAD=1 keeps the working-tree path reachable for the GPU
# session that has been exercising it by hand; no UI path sets it.
case "${TAG}" in
  head|HEAD)
    if [ "${PERFMON_ALLOW_HEAD:-0}" != "1" ]; then
      echo "Error: '${TAG}' is not a supported subject." >&2
      echo "       Building from HEAD needs a full AOTriton build (no released" >&2
      echo "       kernel images exist for it), which this script does not do." >&2
      echo "       Pass a released git tag instead, e.g. 0.13b." >&2
      echo "       (Set PERFMON_ALLOW_HEAD=1 to force the working-tree path.)" >&2
      exit 1
    fi
    TAG="head"   # internal spelling of the working-tree path below
    echo "[build_subject] PERFMON_ALLOW_HEAD=1: using the working tree" >&2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Subjects install into the WORKDIR's installed/ tree, not into the checkout.
# installed/ is the one directory the deploy machinery knows how to ship
# per-arch (installed/<arch>, installed/test/<arch>, installed/database), so
# putting subjects anywhere else would leave them unsyncable -- and a build
# artifact living inside the git checkout is wrong regardless.
PERFMON_WORKDIR="${4:-${PERFMON_WORKDIR:-}}"
if [ -z "${PERFMON_WORKDIR}" ]; then
  echo "Error: no workdir given. Pass it as the 4th argument or set" >&2
  echo "       PERFMON_WORKDIR; the subject installs under" >&2
  echo "       <workdir>/installed/perfmon/<arch>/<tag>/." >&2
  exit 1
fi
if [ ! -d "${PERFMON_WORKDIR}" ]; then
  echo "Error: workdir '${PERFMON_WORKDIR}' does not exist." >&2
  exit 1
fi

# The ROCm is NOT a path segment: one workdir pins one ROCm
# (perfmon::default_rocm), so <arch>/<tag> is already unique within it. It is
# recorded inside the subject instead, so a directory built against a ROCm that
# has since been changed is still identifiable rather than silently assumed
# current.
SUBJECT_ID="aotriton-${TAG}+rocm${ROCM}"
SUBJECT_DIR="${PERFMON_WORKDIR}/installed/perfmon/${ARCH}/${TAG}"
AOTRITON_ROOT="${SUBJECT_DIR}/aotriton"

echo "[build_subject] subject_id=${SUBJECT_ID}" >&2
echo "[build_subject] subject_dir=${SUBJECT_DIR}" >&2
mkdir -p "${SUBJECT_DIR}"
printf '%s\n' "${SUBJECT_ID}" > "${SUBJECT_DIR}/subject_id"

# --- Step 1: source tree (T13 spec item 1) --------------------------------
if [ "${TAG}" == "head" ]; then
  SRC_DIR="${REPO_ROOT}"
  echo "[build_subject] tag=head -> using working tree ${SRC_DIR}" >&2
else
  WORKTREE_DIR="${SUBJECT_DIR}/src"
  if [ -f "${WORKTREE_DIR}/.git" ] || [ -d "${WORKTREE_DIR}/.git" ]; then
    echo "[build_subject] reusing existing worktree ${WORKTREE_DIR}" >&2
  else
    echo "[build_subject] git worktree add ${WORKTREE_DIR} ${TAG}" >&2
    git -C "${REPO_ROOT}" worktree add --detach "${WORKTREE_DIR}" "${TAG}"
  fi
  SRC_DIR="${WORKTREE_DIR}"
fi

# --- Step 2: shim-only AOTriton build (T13 spec item 2) -------------------
# Sourced from THIS tag's own .ci/, not this repo's -- see disclosure #2.
export AOTRITON_NAME_SUFFIX_OVERRIDE=pmon
export AOTRITON_BUILD_PATH="${SUBJECT_DIR}/build-aotriton"
export AOTRITON_INSTALL_PATH="${AOTRITON_ROOT}"

echo "[build_subject] sourcing ${SRC_DIR}/.ci/common-build.sh" >&2
# shellcheck source=/dev/null
. "${SRC_DIR}/.ci/common-build.sh"

# Exactly build-shim.sh's own flags (disclosure #1) plus -DAOTRITON_NO_PYTHON=ON.
common_build "${ARCH}" "shim" \
  -DAOTRITON_NOIMAGE_MODE=ON \
  -DAOTRITON_GPU_BUILD_TIMEOUT=0 \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold" \
  -DAOTRITON_NO_PYTHON=ON

if [ ! -d "${AOTRITON_ROOT}/include/aotriton" ]; then
  echo "[build_subject] ERROR: shim build finished but ${AOTRITON_ROOT}/include/aotriton is missing." >&2
  exit 1
fi

# --- Step 3: kernel images (T13 spec item 3) ------------------------------
if [ "${TAG}" == "head" ]; then
  if [ -d "${AOTRITON_ROOT}/lib/aotriton.images" ]; then
    echo "[build_subject] ${AOTRITON_ROOT}/lib/aotriton.images already present" >&2
  else
    echo "[build_subject] WARNING: no lib/aotriton.images under ${AOTRITON_ROOT}." \
         "T13's own spec text says \"For head, the local build already has" \
         "them\", but a NOIMAGE_MODE=ON shim build (step 2, always run" \
         "regardless of tag) never populates that directory --" \
         "v3src/CMakeLists.txt gates its own images install() on" \
         "'NOT AOTRITON_NOIMAGE_MODE'. This is a disclosed spec" \
         "inconsistency (see this script's own header comment, item 4)," \
         "not a bug in this script. Continuing: T13's own Verify step does" \
         "not exercise enumerate/measure, so it does not need images." >&2
  fi
else
  echo "[build_subject] fetching kernel images for released tag ${TAG}" >&2
  GIT_SHA="$(git -C "${SRC_DIR}" rev-parse --short=12 "${TAG}")"
  # Matches .ci/runc-manylinux-build-tar.sh's own tarball naming
  # (tarbase=aotriton-${GIT_SHORT}${asan_suffix}-images, one file per arch
  # under aotriton/lib/aotriton.images/) -- '*' absorbs an optional
  # '+asan' suffix this script does not otherwise select.
  IMAGES_PATTERN="aotriton-${GIT_SHA}*-images-${ARCH}.tar.gz"
  IMAGES_DL_DIR="${SUBJECT_DIR}/.images-download"
  rm -rf "${IMAGES_DL_DIR}"
  mkdir -p "${IMAGES_DL_DIR}"

  if [ -n "${PERFMON_IMAGES_TARBALL:-}" ]; then
    echo "[build_subject] using PERFMON_IMAGES_TARBALL override: ${PERFMON_IMAGES_TARBALL}" >&2
    cp "${PERFMON_IMAGES_TARBALL}" "${IMAGES_DL_DIR}/"
  elif command -v gh >/dev/null 2>&1; then
    # See this script's header comment, disclosure #5: this publishing step
    # is NOT confirmed to exist anywhere in this repo's own CI config; this
    # is a documented best-effort guess, not a verified integration.
    echo "[build_subject] gh release download ${TAG} -p '${IMAGES_PATTERN}'" >&2
    if ! gh release download "${TAG}" -p "${IMAGES_PATTERN}" -D "${IMAGES_DL_DIR}" --clobber; then
      echo "[build_subject] ERROR: 'gh release download' did not find an asset" \
           "matching '${IMAGES_PATTERN}' on release '${TAG}'. This repo has no" \
           "confirmed step that publishes that tarball to GitHub Releases" \
           "(see this script's header comment, disclosure #5) -- if you have" \
           "the tarball some other way, re-run with" \
           "PERFMON_IMAGES_TARBALL=/path/to/it.tar.gz set." >&2
      exit 1
    fi
  else
    echo "[build_subject] ERROR: 'gh' CLI not found and PERFMON_IMAGES_TARBALL" \
         "is not set -- cannot fetch images for released tag '${TAG}'." \
         "Set PERFMON_IMAGES_TARBALL=/path/to/aotriton-<sha>-images-${ARCH}.tar.gz" \
         "or install the GitHub CLI." >&2
    exit 1
  fi

  TARBALL="$(find "${IMAGES_DL_DIR}" -maxdepth 1 -name '*.tar.gz' | head -n 1)"
  if [ -z "${TARBALL}" ]; then
    echo "[build_subject] ERROR: no .tar.gz found in ${IMAGES_DL_DIR} after fetch." >&2
    exit 1
  fi
  # Tarball root is "aotriton/" (runc-manylinux-build-tar.sh tars from
  # inside AOTRITON_INSTALL_PREFIX, whose child is "aotriton/"), so
  # extracting at SUBJECT_DIR lands it at
  # ${SUBJECT_DIR}/aotriton/lib/aotriton.images/<arch>/ -- exactly beside
  # the shim-built libaotriton*_v2.so already sitting in
  # ${AOTRITON_ROOT}/lib/ (T13 spec item 3: "beside the built .so").
  tar xzf "${TARBALL}" -C "${SUBJECT_DIR}"
  rm -rf "${IMAGES_DL_DIR}"

  if [ ! -d "${AOTRITON_ROOT}/lib/aotriton.images" ]; then
    echo "[build_subject] ERROR: extracted ${TARBALL} but" \
         "${AOTRITON_ROOT}/lib/aotriton.images still does not exist --" \
         "unexpected tarball layout." >&2
    exit 1
  fi
fi

# --- Step 4: libperfmon_flash@<subject> + bin/runner (T13 spec item 4) ---
PERFMON_CORE_ROOT="${PERFMON_CORE_ROOT:-/opt/perfmon/rocm-${ROCM}}"
if [ ! -f "${PERFMON_CORE_ROOT}/include/perfmon/perfmon_abi.h" ]; then
  echo "[build_subject] ERROR: PERFMON_CORE_ROOT (${PERFMON_CORE_ROOT}) has no" \
       "include/perfmon/perfmon_abi.h -- build and install perfmon/core" \
       "(T11) for rocm ${ROCM} first, or set PERFMON_CORE_ROOT to point at" \
       "an existing install (see this script's header comment, item 6)." >&2
  exit 1
fi

FLASH_BUILD_DIR="${SUBJECT_DIR}/build-flash"
echo "[build_subject] configuring modules/flash/perfmon/runner against" \
     "AOTRITON_ROOT=${AOTRITON_ROOT} PERFMON_CORE_ROOT=${PERFMON_CORE_ROOT}" >&2
cmake -S "${REPO_ROOT}/modules/flash/perfmon/runner" -B "${FLASH_BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-hipcc}" \
  -DCMAKE_INSTALL_PREFIX="${SUBJECT_DIR}" \
  -DAOTRITON_ROOT="${AOTRITON_ROOT}" \
  -DPERFMON_CORE_ROOT="${PERFMON_CORE_ROOT}"
cmake --build "${FLASH_BUILD_DIR}"
cmake --install "${FLASH_BUILD_DIR}"

echo "[build_subject] done: ${SUBJECT_DIR}/bin/runner" >&2
echo "[build_subject] verify with:" >&2
echo "  ldd ${SUBJECT_DIR}/bin/runner | grep aotriton" >&2
echo "  ${SUBJECT_DIR}/bin/runner <<< exit; echo \$?" >&2
