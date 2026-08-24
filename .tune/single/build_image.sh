#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Docker image on one host
# Usage: build_image.sh <workdir> <hostname> [--perfmon] [--follow]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2 || true
FOLLOW=""
PERFMON=0

# Flags are order-independent; both are optional.
for arg in "$@"; do
  case "$arg" in
    --follow)  FOLLOW="true" ;;
    --perfmon) PERFMON=1 ;;
    *) echo "Error: unrecognized argument: $arg" >&2; exit 1 ;;
  esac
done

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [--perfmon] [--follow]" >&2
  echo "" >&2
  echo "  Submit a Docker image build job via tsp on <hostname>." >&2
  echo "  --perfmon Build the perfmon measurement image from" >&2
  echo "            perfmon.image.build/Dockerfile.<arch> instead of the tuning" >&2
  echo "            worker image. Generate it with: prepwkdir <workdir> --perfmon" >&2
  echo "  --follow  Tail the build output in real-time (blocks until done)." >&2
  echo "  Without --follow, the job runs in background; check with tsp on the host." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get workdir_override for this hostname. Check the worker-registry table
# first (a registered tuning worker acting as its own build target); a
# build node is explicitly NOT required to be a registered worker (see
# remotebld/get_buildnode_workdir), so fall back to the separate
# buildnode::* config when the hostname matches the configured build node
# instead -- otherwise this would silently resolve to DEFAULT_WORKDIR, which
# may not even exist on that host.
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

if [ -z "$workdir_override" ]; then
  BUILDNODE_HOSTNAME=$(sqlite3 "$WORKDIR/workers.db" \
    "SELECT COALESCE(value,'') FROM config WHERE key='buildnode::hostname'" 2>/dev/null || true)
  if [ -n "$BUILDNODE_HOSTNAME" ] && [ "$HOSTNAME" = "$BUILDNODE_HOSTNAME" ]; then
    workdir_override="$(get_buildnode_workdir "$WORKDIR")"
  fi
fi

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# Which Dockerfile to build. The tuning worker image and the perfmon image are
# NOT interchangeable: create_dockerfile.sh assumes a base that already ships
# python3 and ROCm (ubuntu + distro ROCm), while the perfmon image starts from a
# bare debian and installs TheRock into a venv itself. Building the former on a
# debian base fails at `python3: not found`, which is what this flag exists to
# avoid.
if [ "$PERFMON" -eq 1 ]; then
  if [ -z "$arch" ]; then
    echo "Error: --perfmon needs the target host's architecture, but $HOSTNAME" >&2
    echo "       is not registered as a worker in $WORKDIR/workers.db." >&2
    exit 1
  fi
  DOCKERFILE_REL="perfmon.image.build/Dockerfile.$arch"
  if [ ! -f "$WORKDIR/$DOCKERFILE_REL" ]; then
    echo "Error: $WORKDIR/$DOCKERFILE_REL does not exist." >&2
    echo "       Generate it first:  prepwkdir $WORKDIR --perfmon" >&2
    echo "       then sync the workdir to $HOSTNAME before building." >&2
    exit 1
  fi
  # Tag per-arch. The perfmon image bakes an arch-specific ROCm payload, so a
  # fleet with more than one arch would otherwise have its second build
  # silently overwrite the first under a single CELERY_WORKER_IMAGE tag --
  # leaving an image whose name says nothing about which GPU it can serve.
  IMAGE_TAG="${CELERY_WORKER_IMAGE}-${arch}"
  echo "Perfmon build: $DOCKERFILE_REL -> $IMAGE_TAG"
else
  DOCKERFILE_REL="image.build/Dockerfile"
  IMAGE_TAG="$CELERY_WORKER_IMAGE"
fi

# Certain nodes need --network=host to access internet
if [ -n "$FOLLOW" ]; then
  # Use tsp -t to tail/follow output in real-time
  ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$IMAGE_TAG" "$DOCKERFILE_REL" <<'EOF'
WORKER_WORKDIR="$1"
IMAGE_TAG="$2"
DOCKERFILE_REL="$3"

jobid=$(tsp docker build --network=host -f $WORKER_WORKDIR/$DOCKERFILE_REL -t $IMAGE_TAG $WORKER_WORKDIR)
echo "Job ID: $jobid"
if [ "$(tsp -s "$jobid")" = "queued" ]; then
  echo "Waiting for tsp job $jobid to start..."
  while [ "$(tsp -s "$jobid")" = "queued" ]; do sleep 5; done
fi
tsp -t $jobid
EOF
else
  ssh -n "$HOSTNAME" "tsp docker build --network=host -f $WORKER_WORKDIR/$DOCKERFILE_REL -t $IMAGE_TAG $WORKER_WORKDIR"
fi
