#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build Docker image on one host
# Usage: build_image.sh <workdir> <hostname> [--workload <name>] [--follow]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2 || true
FOLLOW=""
WORKLOAD="kernel"

# Flags are order-independent; both are optional.
while [ "$#" -gt 0 ]; do
  case "$1" in
    --follow)   FOLLOW="true"; shift ;;
    --workload)
      if [ -z "$2" ]; then
        echo "Error: --workload needs a value (kernel|op|perfmon)" >&2
        exit 1
      fi
      WORKLOAD="$2"; shift 2 ;;
    *) echo "Error: unrecognized argument: $1" >&2; exit 1 ;;
  esac
done

case "$WORKLOAD" in
  kernel|op|perfmon) ;;
  *) echo "Error: unknown workload '$WORKLOAD' (expected kernel|op|perfmon)" >&2; exit 1 ;;
esac

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [--workload <name>] [--follow]" >&2
  echo "" >&2
  echo "  Submit a Docker image build job via tsp on <hostname>." >&2
  echo "  --workload <name>  kernel (default) | op | perfmon." >&2
  echo "                     kernel and op share the tuning worker image;" >&2
  echo "                     perfmon builds its own, per-arch, image. Generate" >&2
  echo "                     it with: prepwkdir <workdir> --workload perfmon" >&2
  echo "  --follow           Tail the build output in real-time (blocks)." >&2
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
if [ "$WORKLOAD" = "perfmon" ]; then
  if [ -z "$arch" ]; then
    echo "Error: --workload perfmon needs the target host's architecture, but" >&2
    echo "       $HOSTNAME is not registered as a worker in $WORKDIR/workers.db." >&2
    exit 1
  fi

  # Generate the Dockerfile NOW, from this server's state, and ship it with the
  # build. What the perfmon image contains is decided entirely by its
  # Dockerfile -- it has zero COPY/ADD, so the build context contributes
  # nothing -- and that Dockerfile is generated here from workers.db
  # (perfmon::default_rocm) plus config.rc.
  #
  # Building from a previously synced copy would make the result depend on when
  # the workdir was last deployed: change the ROCm on the PerfmonConfig tab,
  # press build, and you would silently get an image built to the old value
  # with no indication anything was stale. The server is the only authority on
  # what this image is, so it must not be possible to build a different one.
  DOCKERFILE_REL="image.build/Dockerfile.$WORKLOAD.$arch"
  if ! bash "$TUNE_ROOT/lib/create_perfmon_dockerfile.sh" "$WORKDIR" "$arch"; then
    echo "Error: could not generate $DOCKERFILE_REL" >&2
    exit 1
  fi

  # Push it, then build with image.build/ as the context: it holds only
  # generated Dockerfiles, so the upload is negligible, whereas the full
  # workdir would be sent for a build that reads none of it.
  BUILD_CONTEXT="$WORKER_WORKDIR/image.build"
  ssh "$HOSTNAME" "mkdir -p '$BUILD_CONTEXT' && cat > '$WORKER_WORKDIR/$DOCKERFILE_REL'" \
    < "$WORKDIR/$DOCKERFILE_REL"

  # Tag by workload AND arch. Workload because two images that serve different
  # DAGs must not share a name; arch because the perfmon image bakes an
  # arch-specific ROCm payload, so on a multi-arch fleet a shared tag would let
  # the second build silently overwrite the first, leaving an image whose name
  # says nothing about which GPU it can serve.
  IMAGE_TAG="${CELERY_WORKER_IMAGE}-${WORKLOAD}_${arch}"
  echo "Workload $WORKLOAD: $DOCKERFILE_REL -> $IMAGE_TAG (generated and pushed just now)"
else
  # kernel and op share the tuning worker image: same DAG, same container. The
  # workload axis is finer than the image axis, so it does not appear here.
  #
  # This one is NOT regenerated-and-pushed: it COPYs config.rc, image.scripts
  # and files from aotriton.src, so it genuinely needs the synced workdir as
  # its build context and cannot be made independent of the deploy.
  DOCKERFILE_REL="image.build/Dockerfile"
  BUILD_CONTEXT="$WORKER_WORKDIR"
  IMAGE_TAG="$CELERY_WORKER_IMAGE"
fi

# Certain nodes need --network=host to access internet
if [ -n "$FOLLOW" ]; then
  # Use tsp -t to tail/follow output in real-time
  ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$IMAGE_TAG" "$DOCKERFILE_REL" "$BUILD_CONTEXT" <<'EOF'
WORKER_WORKDIR="$1"
IMAGE_TAG="$2"
DOCKERFILE_REL="$3"
BUILD_CONTEXT="$4"

jobid=$(tsp docker build --network=host -f $WORKER_WORKDIR/$DOCKERFILE_REL -t $IMAGE_TAG $BUILD_CONTEXT)
echo "Job ID: $jobid"
if [ "$(tsp -s "$jobid")" = "queued" ]; then
  echo "Waiting for tsp job $jobid to start..."
  while [ "$(tsp -s "$jobid")" = "queued" ]; do sleep 5; done
fi
tsp -t $jobid
EOF
else
  ssh -n "$HOSTNAME" "tsp docker build --network=host -f $WORKER_WORKDIR/$DOCKERFILE_REL -t $IMAGE_TAG $BUILD_CONTEXT"
fi
