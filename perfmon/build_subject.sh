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
#   with the toolchain actually used, and that label goes into the
#   subject's directory name.
#   <arch>            GPU arch, e.g. "gfx942" -- forwarded verbatim as
#                     AOTRITON_TARGET_ARCH.
#   <install_prefix>  Where subjects go: each lands in
#                     <install_prefix><arch>/rocm<ver>+aotriton<tag>/ -- the
#                     preset, which is what resolves it. Trailing slash carried
#                     by the prefix as autotools does. This script knows
#                     nothing of "workdir" -- that is a tuner concept, and
#                     perfmon is not the tuner.
#
# Produces <install_prefix><arch>/<preset>/ -- ONE ordinary Linux install tree,
# not a nest of them. AOTriton and perfmon's own artifacts share it:
#
#   bin/runner                    the executable T13's Verify step checks
#   lib/libperfmon_flash.so       this subject's adapter
#   lib/libaotritonpmon_v2.so*    the shim-built AOTriton
#   lib/aotriton.images/          that release's fetched kernel images
#   include/aotriton/             the shim's headers
#
# and beside it, shared by every subject in the column:
#
#   <install_prefix><arch>/core/rocm-<ver>/lib/libperfmon_core.so
#
# bin/runner reaches all of it through $ORIGIN-relative RUNPATHs, so the
# whole hierarchy can be rsynced to a GPU worker at a different path and
# still resolve -- see the RPATH block in the runner's CMakeLists.txt.
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
#         perfmon/subjects/aotriton-head+rocm7.14.0/lib/
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
#    CMakeLists.txt`'s own "runner executable target" comment.
#
#    That the API drifts per tag is now HANDLED rather than merely disclosed:
#    the harness has one adapter directory per tag,
#    `modules/flash/perfmon/runner/<tag>/adapter.cc`, selected by the
#    PERFMON_FLASH_TAG passed below, with API-independent routines in
#    `runner/lib/`. (An earlier note here predicted a single adapter_v3.cc
#    plus per-tag `overrides/<tag>.cc`; the real drift turned out to be
#    structural enough -- four generations across six tags -- that whole
#    per-tag files are what it took.) A tag with no adapter directory now
#    fails with a one-line reason instead of a compiler dump.
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

# Defaults for the three knobs below. ORIGIN follows
# .ci/releasesuite-git-head.sh's --origin convention; SRC_PREFIX and
# BUILD_PREFIX default somewhere throwaway so the script is usable standalone,
# with the caller pointing them at the workdir's scratch/.
#
# BUILD_PREFIX is deliberately NOT under <install_prefix>. Everything beneath
# the subject dir is a build PRODUCT that sync_workdir.sh --workload perfmon
# rsyncs to the GPU workers (with --delete); a cmake build tree there would be
# shipped to every worker for nothing and would make the subject dir no longer
# describe just the subject. The per-tag AOTriton shim builds already keep
# their trees out of the way this way; this is the runner build catching up.
ORIGIN="https://github.com/ROCm/aotriton.git"
SRC_PREFIX="${TMPDIR:-/tmp}/perfmon-src/"
BUILD_PREFIX="${TMPDIR:-/tmp}/perfmon-build/"
# Shared by every tag and arch, unlike the two prefixes above: an asset name
# is unique across releases and the images are arch-GROUP-specific at most, so
# one download serves the fleet. Not a prefix for the same reason -- nothing is
# appended to it.
CACHE_DIR="${TMPDIR:-/tmp}/perfmon-cache"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --origin)       ORIGIN="$2"; shift 2 ;;
    --src_prefix)   SRC_PREFIX="$2"; shift 2 ;;
    --build_prefix) BUILD_PREFIX="$2"; shift 2 ;;
    --cache_dir) CACHE_DIR="$2"; shift 2 ;;
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
  echo '                    <install_prefix><arch>/rocm<ver>+aotriton<tag>/.' >&2
  echo '                    Carry the trailing' >&2
  echo '                    slash yourself, autotools-style.' >&2
  echo '  <tag>...:         one or more git tags, e.g. 0.13b 0.12.1b' >&2
  echo '  --origin:         git URL to clone release tags from' >&2
  echo "                    (default: ${ORIGIN})" >&2
  echo '  --src_prefix:     where shallow clones go; each tag lands in' >&2
  echo '                    <src_prefix><tag>/. Trailing slash included.' >&2
  echo "                    (default: ${SRC_PREFIX})" >&2
  echo '  --build_prefix:   where cmake build trees go; each subject builds in' >&2
  echo '                    <build_prefix><arch>/<tag>/. Never under' >&2
  echo '                    <install_prefix>: that subtree is rsynced to the' >&2
  echo '                    GPU workers and should hold products only.' >&2
  echo "                    (default: ${BUILD_PREFIX})" >&2
  echo '  --cache_dir:      where downloaded release tarballs are kept and' >&2
  echo '                    reused. Shared across tags and arches; nothing is' >&2
  echo '                    appended, so no trailing slash is needed.' >&2
  echo "                    (default: ${CACHE_DIR})" >&2
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
# ends up in the subject's directory name, i.e. in what the published
# numbers claim they were
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
# <workdir>/installed/perfmon/ and this puts <prefix><arch>/<preset>/ beneath it,
# with the trailing slash carried by the prefix as autotools does.
if [ -z "${INSTALL_PREFIX}" ]; then
  echo "Error: empty install prefix." >&2
  exit 1
fi

build_one_subject() {
  local TAG="$1"

  # The directory IS the preset. A subject is a
  # (ROCm, AOTriton tag) pair, and `rocm<ver>+aotriton<tag>` is how that pair
  # is spelled everywhere else -- python/tune/perfmon/presets.py builds it,
  # task_config carries it, and launch_runner.sh resolves
  # <root>/<arch>/<preset>/bin/runner from it.
  #
  # This used to be <arch>/<tag>, on the reasoning that one prefix serves one
  # ROCm so the tag alone is unique beneath it. True, and beside the point: the
  # shim was still looking up the preset, so every launch failed with
  #     launch_runner.sh: no runner for preset 'rocm7.14.0+aotriton0.10b'
  #     Expected: .../gfx942/rocm7.14.0+aotriton0.10b/bin/runner
  # while the subject sat next to it under a different name. Naming the
  # directory after the thing that looks it up removes the translation step
  # rather than fixing it in one direction.
  #
  # ${ROCM} here is the version PROBED during this build, not a configured
  # one, so the directory name records what the subject was actually built
  # against. That provenance used to live in a separate subject_id file; the
  # path carries it now, and nothing else did.
  PRESET="rocm${ROCM}+aotriton${TAG}"
  SUBJECT_DIR="${INSTALL_PREFIX}${ARCH}/${PRESET}"

  # AOTriton installs into the subject prefix ITSELF, not a nested
  # <subject>/aotriton/. A subject is one self-contained install tree with
  # the ordinary Linux layout:
  #
  #     <subject>/bin/runner
  #     <subject>/lib/libperfmon_flash.so
  #     <subject>/lib/libaotritonpmon_v2.so*
  #     <subject>/lib/aotriton.images/
  #     <subject>/include/aotriton/
  #
  # The earlier nesting put AOTriton's include/ and lib/ one level down
  # while perfmon's own bin/ and lib/ sat at the top, so the tree had two
  # unrelated lib/ dirs and matched no convention. Nothing needs them
  # separated: the names do not collide (libaotritonpmon_v2 vs
  # libperfmon_flash), and merging them is what lets bin/runner reach every
  # in-subject dependency through a single `$ORIGIN/../lib`.
  AOTRITON_ROOT="${SUBJECT_DIR}"

  echo "[build_subject] subject_dir=${SUBJECT_DIR}" >&2
  mkdir -p "${SUBJECT_DIR}"

  # Sweep out layouts this script no longer produces. Without this, a subject
  # dir built by an older revision keeps its nested aotriton/ tree beside the
  # new flat include/ and lib/, and the result is worse than either -- two
  # copies of the same headers, and a `lib/` that looks right while a stale
  # `aotriton/lib/` still holds the library ldd actually resolved last time.
  #
  # Only paths the CURRENT code never creates are listed, so this can never
  # delete live output: aotriton/ is the old nesting, src/ was an in-subject
  # clone before --src_prefix existed, and build-flash/ and .images-download/
  # were build scratch before --build_prefix moved it out.
  for _stale in aotriton src build-flash .images-download; do
    if [ -e "${SUBJECT_DIR}/${_stale}" ]; then
      echo "[build_subject] removing stale ${SUBJECT_DIR}/${_stale}" \
           "(no longer produced by this script)" >&2
      rm -rf "${SUBJECT_DIR:?}/${_stale}"
    fi
  done

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
  # Passed by environment, not as a fourth positional: the
  # `build-<tag>.sh <src_dir> <install_dir> <arch>` contract is implemented
  # six times over, and release_asset.sh is the only thing that reads it.
  PERFMON_CACHE_DIR="${CACHE_DIR}" \
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

  # The flash adapter is selected by TAG: modules/flash/perfmon/runner/<tag>/
  # adapter.cc, with API-independent routines factored into runner/lib/.
  #
  # rev0 §4's table assumed exactly two adapters -- v2 for 0.9.x, one v3 for
  # everything from 0.10b -- while the same section warned "Do not assume v3
  # struct stability". The warning won: the public flash API drifts WITHIN
  # the v3 line, so the tag range spans FOUR generations. Read from each
  # tag's own include/aotriton/{flash,util}.h:
  #
  #   0.9.2b            v2 only. No params struct at all; the entry points
  #                     are long positional v2::flash free functions, and
  #                     backends cannot be forced.
  #   0.10b             v3 params exist, but attn_options is EMPTY (no
  #                     force_backend_index), there is no LazyTensor, and
  #                     attn_bwd_params has no DQ_ACC.
  #   0.11b, 0.11.2b    LazyTensor, force_backend_index and DQ_ACC all
  #                     exist, but LazyTensor is cookie-style --
  #                     acquire(void*), no `eager` member.
  #   0.12.1b, 0.13b    LazyTensor gains `eager`; acquire/dispose take
  #   (and head)        LazyTensor<Rank>* self.
  #
  # Tags sharing a generation are directory symlinks, so they cannot drift
  # apart unnoticed. A tag with no directory fails HERE rather than
  # compiling a neighbour's adapter against the wrong headers and burying
  # the reason in fifty "no member named ..." diagnostics.
  FLASH_ADAPTER_DIR="${REPO_ROOT}/modules/flash/perfmon/runner/${TAG}"
  if [ ! -f "${FLASH_ADAPTER_DIR}/adapter.cc" ]; then
    echo "[build_subject] ERROR: no flash adapter for tag '${TAG}'." \
         "Expected ${FLASH_ADAPTER_DIR}/adapter.cc. The public flash API is" \
         "not stable across releases, so every tag needs its own adapter;" \
         "write it against that tag's own include/aotriton/{flash,util}.h," \
         "or symlink that directory to a tag whose API you have verified is" \
         "identical." >&2
    exit 1
  fi

  FLASH_BUILD_DIR="${BUILD_PREFIX}${ARCH}/${TAG}/flash"
  mkdir -p "${FLASH_BUILD_DIR}"
  echo "[build_subject] configuring modules/flash/perfmon/runner against" \
       "AOTRITON_ROOT=${AOTRITON_ROOT} PERFMON_CORE_ROOT=${PERFMON_CORE_ROOT}" \
       "adapter=${TAG}/adapter.cc" >&2
  cmake -S "${REPO_ROOT}/modules/flash/perfmon/runner" -B "${FLASH_BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${CXX:-hipcc}" \
    -DCMAKE_INSTALL_PREFIX="${SUBJECT_DIR}" \
    -DAOTRITON_ROOT="${AOTRITON_ROOT}" \
    -DPERFMON_CORE_ROOT="${PERFMON_CORE_ROOT}" \
    -DPERFMON_FLASH_TAG="${TAG}"
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
# TODO (long term, not scheduled): the rocm-${ROCM} level here exists only
# because perfmon_core links HIP as one indivisible library. Once it is split
# into ROCm-dependent and CPU-only halves (see perfmon/core/CMakeLists.txt),
# one install can serve several ROCm versions at once and this key goes away --
# along with the requirement that anything looking for a core install must
# first know which ROCm version it wants.
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
