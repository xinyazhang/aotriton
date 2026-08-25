#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build perfmon subjects on the build node.
# Usage: build_perfmon.sh <workdir> <arch> <tag> [<tag>...] [--follow]
#
# Subjects are built where the ROCm toolchain is, not where the WebUI runs.
# perfmon/build_subject.sh needs hipcc, cmake and a ROCm install; the server
# hosting the WebUI generally has none of them. So this script does what
# remotebld --test does for the testing build: ssh to the build node and run
# the builder inside a container, tsp-tracked so it survives a disconnect.
#
# The container is the per-arch perfmon image
# (<CELERY_WORKER_IMAGE>-perfmon_<arch>, built by build_image.sh --workload
# perfmon), and the remote workdir is bind-mounted at /wkdir, so subjects land
# in <remote workdir>/installed/perfmon/<arch>/<tag>/ -- inside the tree
# sync_workdir.sh --workload perfmon knows how to ship. build_subject.sh is
# handed that path as a plain install prefix; it has no notion of a workdir.
#
# All tags for one arch go in a single invocation: build_subject.sh validates
# the shared, ROCm-scoped libperfmon_core once for the batch instead of once
# per tag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="${1:-}"
ARCH="${2:-}"
shift 2 || true

FOLLOW=""
TAGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --follow) FOLLOW="true"; shift ;;
    -*) echo "Error: unrecognized argument: $1" >&2; exit 1 ;;
    *) TAGS+=("$1"); shift ;;
  esac
done

if [ -z "$WORKDIR" ] || [ -z "$ARCH" ] || [ "${#TAGS[@]}" -eq 0 ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <arch> <tag> [<tag>...] [--follow]

  Build perfmon subjects for <arch> on the configured build node, inside the
  per-arch perfmon image, via tsp.

  --follow  Tail the build output in real-time (blocks until done).
EOF
  exit 1
fi

if [ ! -d "$WORKDIR" ]; then
  echo "Error: workdir does not exist: $WORKDIR" >&2
  exit 1
fi

load_config "$WORKDIR"
ABSWORKDIR=$(realpath "$WORKDIR")

# The build node is required, not optional. Falling back to a local build --
# as remotebld does for the tuning build -- would mean running hipcc on the
# WebUI server, which is exactly what this script exists to avoid.
BUILD_NODE_ENABLE=$(sqlite3 "$ABSWORKDIR/workers.db" \
  "SELECT COALESCE(value,'') FROM config WHERE key='buildnode::enable'" 2>/dev/null || true)
BUILD_NODE_HOST=$(sqlite3 "$ABSWORKDIR/workers.db" \
  "SELECT COALESCE(value,'') FROM config WHERE key='buildnode::hostname'" 2>/dev/null || true)

if [ "$BUILD_NODE_ENABLE" != "1" ] || [ -z "$BUILD_NODE_HOST" ]; then
  echo "Error: no build node is enabled in $ABSWORKDIR/workers.db." >&2
  echo "       Perfmon subjects need the ROCm toolchain, which lives in the" >&2
  echo "       perfmon image on the build node -- configure and enable one on" >&2
  echo "       the Builds tab." >&2
  exit 1
fi

ROCM=$(sqlite3 "$ABSWORKDIR/workers.db" \
  "SELECT COALESCE(value,'') FROM config WHERE key='perfmon::default_rocm'" 2>/dev/null || true)
if [ -z "$ROCM" ]; then
  echo "Error: perfmon::default_rocm is not set in $ABSWORKDIR/workers.db." >&2
  echo "       Set it on the WebUI's PerfmonConfig tab." >&2
  exit 1
fi

REMOTE_WORKDIR="$(get_buildnode_workdir "$ABSWORKDIR")"
PERFMON_IMAGE="${CELERY_WORKER_IMAGE}-perfmon_${ARCH}"

echo "Build node:   $BUILD_NODE_HOST ($REMOTE_WORKDIR)"
echo "Image:        $PERFMON_IMAGE"
echo "Image ROCm:   $ROCM (subjects label themselves from the probed toolchain)"
echo "Arch:         $ARCH"
echo "Tags:         ${TAGS[*]}"

# --user keeps artifacts owned by the invoking uid:gid, matching how the
# perfmon image builds its venv -- otherwise installed/perfmon/ comes back
# root-owned and the next build cannot overwrite it. It is written \$(id -u)
# so the substitution runs on the REMOTE, not here.
#
# Follow and no-follow are two different ssh invocations, exactly as
# build_image.sh and remotebld do it: the heredoc form queues and then tails
# unconditionally, the one-liner form just queues. An earlier version passed a
# FOLLOW flag into a single payload and branched on it inside; the branch never
# fired, and the job was queued with its output going nowhere. Matching the
# proven shape removes the failure mode rather than debugging it.
DOCKER_RUN="docker run --rm --network=host --user \$(id -u):\$(id -g)"
DOCKER_RUN="$DOCKER_RUN --mount type=bind,source=$REMOTE_WORKDIR,target=/wkdir"
DOCKER_RUN="$DOCKER_RUN $PERFMON_IMAGE"
DOCKER_RUN="$DOCKER_RUN bash /wkdir/aotriton.src/perfmon/build_subject.sh"
DOCKER_RUN="$DOCKER_RUN $ARCH /wkdir/installed/perfmon/ ${TAGS[*]}"

if [ -n "$FOLLOW" ]; then
  # Use tsp -t to tail/follow output in real-time
  # shellcheck disable=SC2029
  ssh "$BUILD_NODE_HOST" bash -s "$DOCKER_RUN" <<'EOF'
DOCKER_RUN="$1"

jobid=$(tsp $DOCKER_RUN)
echo "Job ID: $jobid"
if [ "$(tsp -s "$jobid")" = "queued" ]; then
  echo "Waiting for tsp job $jobid to start..."
  while [ "$(tsp -s "$jobid")" = "queued" ]; do sleep 5; done
fi
tsp -t $jobid
EOF
else
  # shellcheck disable=SC2029
  ssh -n "$BUILD_NODE_HOST" "tsp $DOCKER_RUN"
fi
