#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# perfmon-exec0.md T13: builds "subjects" -- AOTriton tags, shim-built for one
# ROCm/arch combination, plus this branch's own perfmon harness
# (libperfmon_flash@<subject> + bin/runner) linked against each.
#
# Runs INSIDE the perfmon image, on the build node: it needs the ROCm toolchain
# that image carries. .tune/single/build_perfmon.sh is what puts it there; this
# script is not meant to be invoked on the webui server, which has no ROCm.
#
# Usage: build_subject.sh <arch> <install_prefix> <tag> [<tag>...]
#   Several tags in one invocation, because the (ROCm, arch)-scoped work --
#   building the shared libperfmon_core -- is done once for the batch rather
#   than repeated per tag. Each tag still builds in
#   its own subshell, so one failure does not abort the rest; the exit status
#   is nonzero if any failed.
#
#   <tag>   git tag to build, e.g. 0.13b. HEAD is NOT supported: it has no
#           published kernel images, so it would need a full AOTriton build
#           rather than the shim build this script performs. The working-tree
#           path still exists behind PERFMON_ALLOW_HEAD=1 -- it is the path
#           T13 actually exercised -- but no UI reaches it. The released-tag
#           path is implemented per spec but UNVERIFIED, see disclosures.
#   The ROCm version is PROBED, not passed: this script cannot locate a ROCm
#   install from a version string -- it uses whatever ROCM_PATH/hipcc the
#   environment provides -- so accepting one could only let the label disagree
#   with the toolchain actually used, and that label goes into subject_id.
#   <arch>            GPU arch, e.g. "gfx942" -- forwarded verbatim as
#                     AOTRITON_TARGET_ARCH.
#   <install_prefix>  Where subjects go: each lands in
#                     <install_prefix><arch>/<tag>/, the trailing slash carried
#                     by the prefix as autotools does. This script knows
#                     nothing of "workdir" -- that is a tuner concept, and
#                     perfmon is not the tuner.
#
# Produces <install_prefix><arch>/<tag>/ containing a
# `subject_id` file (aotriton-<tag>+rocm<rocm>, perfmon-rev0.md §9/§11) and:
#   aotriton/            AOTRITON_ROOT: shim-built include/+lib/, plus
#                         (released tags only) fetched lib/aotriton.images/
#   bin/runner            the executable T13's own Verify step checks
#   lib/libperfmon_flash.so
#
# RUN AND VERIFIED for `head 7.14.0 gfx942` on an 8x gfx942 node with
# theRock ROCm 7.14: T13's Verify step passes (see the runner CMakeLists).
# The <tag> != head path now builds the shim as far as step 3; the images
# fetch there is still UNRUN -- disclosure #5 below stands unchanged.
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
# 1. Step 2 (the shim-only AOTriton build) is delegated to a PER-TAG script,
#    perfmon/scripts/build-<tag>.sh, rather than driven from here.
#
#    An earlier version sourced ${SRC_DIR}/.ci/common-build.sh and called
#    its `common_build` function, on the assumption that every tag ships
#    that file with a compatible interface. That assumption is false, and
#    not marginally so: 0.9.2b and 0.10b have NO .ci/ directory whatsoever,
#    0.11b has common-build.sh but no build-shim.sh, and the signatures of
#    the ones that do exist drift between releases. A single call site
#    cannot track that, and every attempt to make it do so would encode the
#    newest tag's interface and silently break the older ones.
#
#    So each tag owns its build script, free to reuse that tag's own .ci/
#    helpers where they fit and to invoke cmake directly where they do not.
#    The contract between them is deliberately narrow:
#        build-<tag>.sh <src_dir> <install_dir> <arch>
#    with success meaning <install_dir>/include/aotriton/ and
#    lib/libaotriton*_v2.so exist -- both verified here afterwards, since a
#    per-tag script returning 0 without producing them is exactly the
#    failure this indirection could otherwise hide.
#
# 2. Those scripts read the SUBJECT'S OWN tag's source (`${SRC_DIR}`), never
#    this repo's -- an old tag's build system may differ from this branch's,
#    and the shim build must reflect that tag's, not ours.
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
# 5. Kernel images are fetched by perfmon/scripts/build-<tag>.sh, not here,
#    because their asset naming drifts as hard as the build system does:
#    0.9.2b and 0.10b publish no SEPARATE images package -- the images are
#    inside their one jumbo tarball -- 0.11b/0.11.2b publish gfx11xx as one group, and
#    0.12.1b/0.13b split that into gfx110x/gfx115x and add gfx1250. Each tag's
#    script names its own asset, verified against that release's actual asset
#    list; the download and extraction mechanics they share live in
#    scripts/lib/release_asset.sh. PERFMON_IMAGES_TARBALL=<path> still
#    overrides the download for a tarball obtained some other way.


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

# Defaults for the two knobs below. ORIGIN follows .ci/releasesuite-git-head.sh's
# --origin convention; SRC_PREFIX defaults somewhere throwaway so the script is
# usable standalone, with the caller pointing it at the workdir's scratch/.
ORIGIN="https://github.com/ROCm/aotriton.git"
SRC_PREFIX="${TMPDIR:-/tmp}/perfmon-src/"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --origin)     ORIGIN="$2"; shift 2 ;;
    --src_prefix) SRC_PREFIX="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Error: unrecognized option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done

if [ "$#" -lt 3 ]; then
  echo "Usage: build_subject.sh [--origin <url>] [--src_prefix <dir>] \\" >&2
  echo "                        <arch> <install_prefix> <tag> [<tag>...]" >&2
  echo '  <arch>:           GPU arch, e.g. gfx942 (-> AOTRITON_TARGET_ARCH)' >&2
  echo '  <install_prefix>: where subjects go; each lands in' >&2
  echo '                    <install_prefix><arch>/<tag>/. Carry the trailing' >&2
  echo '                    slash yourself, autotools-style.' >&2
  echo '  <tag>...:         one or more git tags, e.g. 0.13b 0.12.1b' >&2
  echo '  --origin:         git URL to clone release tags from' >&2
  echo "                    (default: ${ORIGIN})" >&2
  echo '  --src_prefix:     where shallow clones go; each tag lands in' >&2
  echo '                    <src_prefix><tag>/. Trailing slash included.' >&2
  echo "                    (default: ${SRC_PREFIX})" >&2
  exit 1
fi

ARCH="$1"
INSTALL_PREFIX="$2"
shift 2
TAGS=("$@")

# HEAD is not a supported subject. A released tag is shim-built and paired
# with that release's prebuilt kernel images; HEAD has no published images, so
# it would need a full AOTriton build (Triton, every kernel) before it could be
# measured -- a different and far more expensive operation than this script
# performs. Refuse rather than half-produce a subject with no images.
#
# PERFMON_ALLOW_HEAD=1 keeps the working-tree path reachable for the GPU
# session that has been exercising it by hand; no UI path sets it.
for _t in "${TAGS[@]}"; do
  # `core` is not a tag, it is the slot libperfmon_core occupies at
  # <install_prefix><arch>/core/. The layout is only safe while no AOTriton
  # tag is called that, so enforce it rather than trusting it.
  if [ "$_t" = "core" ]; then
    echo "Error: 'core' is reserved -- it names libperfmon_core's directory" >&2
    echo "       at <install_prefix><arch>/core/, not an AOTriton tag." >&2
    exit 1
  fi
  case "$_t" in
    head|HEAD)
      if [ "${PERFMON_ALLOW_HEAD:-0}" != "1" ]; then
        echo "Error: '${_t}' is not a supported subject." >&2
        echo "       Building from HEAD needs a full AOTriton build (no released" >&2
        echo "       kernel images exist for it), which this script does not do." >&2
        echo "       Pass released git tags instead, e.g. 0.13b." >&2
        echo "       (Set PERFMON_ALLOW_HEAD=1 to force the working-tree path.)" >&2
        exit 1
      fi
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# The ROCm version is PROBED from the toolchain actually in use, never taken
# as input. This script cannot locate a ROCm install from a version string --
# it uses whatever ROCM_PATH/hipcc the environment provides -- so accepting one
# would only let the label disagree with the thing being used, and that label
# ends up in subject_id, i.e. in what the published numbers claim they were
# measured against.
probe_rocm_version() {
  local v=""
  if command -v rocm-sdk >/dev/null 2>&1; then
    v="$(rocm-sdk version 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$v" ] && command -v hipconfig >/dev/null 2>&1; then
    v="$(hipconfig --version 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$v" ] && [ -n "${ROCM_PATH:-}" ] && [ -f "${ROCM_PATH}/.info/version" ]; then
    v="$(head -1 "${ROCM_PATH}/.info/version" || true)"
  fi
  v="$(printf '%s' "$v" | tr -d '[:space:]')"
  printf '%s' "${v%%-*}"          # 6.2.41134-abcdef -> 6.2.41134
}

ROCM="$(probe_rocm_version)"
if [ -z "${ROCM}" ]; then
  echo "Error: could not determine the ROCm version." >&2
  echo "       Tried rocm-sdk, hipconfig and \$ROCM_PATH/.info/version. This" >&2
  echo "       script must run where the ROCm toolchain is -- inside the" >&2
  echo "       perfmon image, not on the WebUI server." >&2
  exit 1
fi
echo "[build_subject] probed ROCm ${ROCM}" >&2

# Subjects install under the caller's prefix. This script has no idea what a
# "workdir" is -- that is a tuner concept, and perfmon is not the tuner. The
# caller (.tune/single/build_perfmon.sh) passes
# <workdir>/installed/perfmon/ and this puts <prefix><arch>/<tag>/ beneath it,
# with the trailing slash carried by the prefix as autotools does.
if [ -z "${INSTALL_PREFIX}" ]; then
  echo "Error: empty install prefix." >&2
  exit 1
fi

build_one_subject() {
  local TAG="$1"

  # The ROCm is NOT a path segment: one prefix serves one ROCm, so
  # <arch>/<tag> is already unique beneath it. The probed version is recorded
  # inside the subject instead, so a directory built against a ROCm that has
  # since changed stays identifiable rather than silently assumed current.
  SUBJECT_ID="aotriton-${TAG}+rocm${ROCM}"
  SUBJECT_DIR="${INSTALL_PREFIX}${ARCH}/${TAG}"
  AOTRITON_ROOT="${SUBJECT_DIR}/aotriton"

  echo "[build_subject] subject_id=${SUBJECT_ID}" >&2
  echo "[build_subject] subject_dir=${SUBJECT_DIR}" >&2
  mkdir -p "${SUBJECT_DIR}"
  printf '%s\n' "${SUBJECT_ID}" > "${SUBJECT_DIR}/subject_id"

  # --- Step 1: source tree ------------------------------------------------
  # A shallow clone of the tag from upstream, NOT a worktree of the local
  # checkout. The local aotriton.src is a depth-limited, single-branch clone
  # synced from the dev node, so it carries no release tags -- and teaching the
  # dev node to fetch them just to ship them was solving the wrong problem:
  # upstream is where release tags authoritatively live, and the build node can
  # reach it directly.
  #
  # Sources land under ${SRC_PREFIX}, which the caller points at the workdir's
  # scratch/ -- excluded from sync_workdir, so a full source tree per tag never
  # gets rsynced to GPU workers alongside the runner.
  if [ "${TAG}" == "head" ]; then
    SRC_DIR="${REPO_ROOT}"
    echo "[build_subject] tag=head -> using working tree ${SRC_DIR}" >&2
  else
    SRC_DIR="${SRC_PREFIX}${TAG}"
    if [ -e "${SRC_DIR}/.git" ]; then
      echo "[build_subject] reusing existing clone ${SRC_DIR}" >&2
    else
      echo "[build_subject] git clone --depth 1 --branch ${TAG} ${ORIGIN}" >&2
      mkdir -p "$(dirname "${SRC_DIR}")"
      # Clone into a temporary sibling and move it into place, so an
      # interrupted clone cannot leave a half-populated directory that the
      # `-e .git` check above would later mistake for a good one.
      rm -rf "${SRC_DIR}.partial"
      git clone --depth 1 --branch "${TAG}" "${ORIGIN}" "${SRC_DIR}.partial"
      mv "${SRC_DIR}.partial" "${SRC_DIR}"
    fi
  fi

  # --- Step 2: shim-only AOTriton build -------------------------------------
  # Delegated to a PER-TAG script, perfmon/scripts/build-<tag>.sh.
  #
  # The previous version sourced ${SRC_DIR}/.ci/common-build.sh and called
  # common_build(), which assumed every tag ships that file with HEAD's
  # interface. It does not: 0.9.2b and 0.10b have no .ci/ directory at all
  # ("No such file or directory"), 0.11b has common-build.sh but no
  # build-shim.sh, and the signatures drift across the ones that do exist. One
  # caller cannot track that; each tag gets its own script, free to reuse that
  # tag's own .ci/ helpers where they fit and to invoke cmake directly where
  # they do not.
  #
  # Contract, identical for every tag:
  #     build-<tag>.sh <src_dir> <install_dir> <arch>
  # producing <install_dir>/include/aotriton/ and lib/libaotriton*_v2.so.
  TAG_BUILD="${SCRIPT_DIR}/scripts/build-${TAG}.sh"
  if [ ! -x "${TAG_BUILD}" ]; then
    echo "[build_subject] ERROR: no build script for tag '${TAG}'." \
         "Expected ${TAG_BUILD}. Each tag needs its own, because the .ci/" \
         "build interface is not stable across releases -- see the scripts" \
         "that do exist in $(dirname "${TAG_BUILD}")." >&2
    exit 1
  fi

  echo "[build_subject] ${TAG_BUILD} ${SRC_DIR} ${AOTRITON_ROOT} ${ARCH}" >&2
  "${TAG_BUILD}" "${SRC_DIR}" "${AOTRITON_ROOT}" "${ARCH}"

  if [ ! -d "${AOTRITON_ROOT}/include/aotriton" ]; then
    echo "[build_subject] ERROR: ${TAG_BUILD} returned success but" \
         "${AOTRITON_ROOT}/include/aotriton is missing." >&2
    exit 1
  fi
  if ! compgen -G "${AOTRITON_ROOT}/lib/libaotriton*_v2.so" >/dev/null; then
    echo "[build_subject] ERROR: ${TAG_BUILD} returned success but no" \
         "${AOTRITON_ROOT}/lib/libaotriton*_v2.so was installed." >&2
    exit 1
  fi

  # Kernel images are fetched by the per-tag script above, not here.
  #
  # They have to be: the asset naming drifts as hard as the build system does.
  # 0.9.2b and 0.10b publish no SEPARATE images package -- the images are
  # inside their one jumbo tarball; 0.11b/0.11.2b publish gfx11xx as one group;
  # 0.12.1b/0.13b split that into gfx110x and gfx115x and add gfx1250. There
  # is no git sha in any of those names -- the version of this step that lived
  # here matched on one, and so could never have found anything.
  if [ ! -d "${AOTRITON_ROOT}/lib/aotriton.images" ]; then
    echo "[build_subject] ERROR: ${TAG_BUILD} left no" \
         "${AOTRITON_ROOT}/lib/aotriton.images -- the subject would have no" \
         "kernels to measure." >&2
    exit 1
  fi

  # --- Step 4: libperfmon_flash@<subject> + bin/runner (T13 spec item 4) ---

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
}

# ROCm-scoped, not per-tag: check it ONCE, before any subject is built. Failing
# after the first tag's AOTriton build would waste the expensive part of the run
# to report a precondition that was already false at the start.
# libperfmon_core lives INSIDE the arch subtree, beside that arch's subjects:
#
#     <install_prefix><arch>/core/rocm-<version>/
#     <install_prefix><arch>/<tag>/
#
# It is AOTriton-neutral -- that is what makes a timing comparison between two
# AOTriton versions meaningful (rev0 D4) -- but it is NOT arch-neutral:
# fill.cc carries a __global__ kernel, so the library embeds a GPU code
# object.
#
# `core` simply occupies the <tag> slot, which is safe because no AOTriton tag
# is named that (asserted below rather than assumed). Arch has to lead because
# that is the unit that gets deployed: sync_workdir.sh --workload perfmon
# ships installed/perfmon/<arch>/, so a core parked beside the arch dirs
# rather than inside one would never reach the worker, and the runner would
# arrive without the libperfmon_core.so its RPATH points at. ROCm is the inner
# key so one build is shared by every subject in that column.
#
# The original default, /opt/perfmon/rocm-<ver>, was a path nothing in this
# repo ever writes, so it could only fail -- which is what the build node
# reported.
PERFMON_CORE_ROOT="${PERFMON_CORE_ROOT:-${INSTALL_PREFIX}${ARCH}/core/rocm-${ROCM}}"

# Build it if it is not there: leaving it a manual prerequisite meant a
# prerequisite nothing performed.
if [ ! -f "${PERFMON_CORE_ROOT}/include/perfmon/perfmon_abi.h" ]; then
  echo "[build_subject] libperfmon_core not found at ${PERFMON_CORE_ROOT}" >&2
  echo "[build_subject] building perfmon/core for rocm ${ROCM} / ${ARCH}" \
       "(once per ROCm+arch)" >&2
  # Build tree goes to TMPDIR, not under the prefix: everything beneath
  # <install_prefix> is rsynced to GPU workers, and shipping a CMake build
  # tree there would be pure waste. perfmon/core is five translation units,
  # so re-configuring on a rebuild costs seconds.
  CORE_BUILD_DIR="${TMPDIR:-/tmp}/perfmon-core-rocm${ROCM}-${ARCH}"
  cmake -S "${SCRIPT_DIR}/core" -B "${CORE_BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${CXX:-hipcc}" \
    -DPERFMON_TARGET_ARCH="${ARCH}" \
    -DCMAKE_INSTALL_PREFIX="${PERFMON_CORE_ROOT}"
  cmake --build "${CORE_BUILD_DIR}"
  cmake --install "${CORE_BUILD_DIR}"
fi

if [ ! -f "${PERFMON_CORE_ROOT}/include/perfmon/perfmon_abi.h" ]; then
  echo "[build_subject] ERROR: still no ${PERFMON_CORE_ROOT}/include/perfmon/perfmon_abi.h" \
       "after building perfmon/core -- check the cmake output above." >&2
  exit 1
fi

FAILED=()
for TAG in "${TAGS[@]}"; do
  echo "[build_subject] === ${TAG} (${ARCH}, rocm ${ROCM}) ===" >&2
  # Subshell so one tag's failure does not abort the batch under `set -e`, and
  # so the per-subject variables cannot leak into the next iteration.
  if ( build_one_subject "${TAG}" ); then
    echo "[build_subject] ok: ${TAG}" >&2
  else
    echo "[build_subject] FAILED: ${TAG}" >&2
    FAILED+=("${TAG}")
  fi
done

if [ "${#FAILED[@]}" -ne 0 ]; then
  echo "[build_subject] ${#FAILED[@]} of ${#TAGS[@]} subject(s) failed: ${FAILED[*]}" >&2
  exit 1
fi
echo "[build_subject] all ${#TAGS[@]} subject(s) built for ${ARCH}" >&2
