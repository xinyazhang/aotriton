#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start worker on one host
# Usage: start_worker.sh <workdir> <hostname> [-- <extra args>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/db_query.sh"

WORKDIR="$1"
HOSTNAME="$2"
shift 2

# Collect extra args after '--'
EXTRA_ARGS=()
if [ "$1" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
fi

if [ -z "$WORKDIR" ] || [ -z "$HOSTNAME" ]; then
  echo "Usage: $0 <workdir> <hostname> [-- <extra args>]" >&2
  echo "" >&2
  echo "  Start a worker container on <hostname> via SSH." >&2
  echo "" >&2
  echo "  WARNING: This script does NOT read GPU selection from the workers DB." >&2
  echo "  GPU assignment must be passed explicitly via extra args (e.g. -- --multi_gpu 0 1)." >&2
  echo "  Use wkctl start to apply GPU selection automatically." >&2
  exit 1
fi

load_config "$WORKDIR"

# Get arch and workdir_override for this hostname
WORKER_INFO=$(get_worker_by_hostname "$WORKDIR" "$HOSTNAME")
IFS='|' read -r arch workdir_override <<< "$WORKER_INFO"

WORKER_WORKDIR="${workdir_override:-$DEFAULT_WORKDIR}"

# Add --hostname to extra args
EXTRA_ARGS=(--hostname "$HOSTNAME" "${EXTRA_ARGS[@]}")

# Resolve the workload HERE, not on the remote side, because it selects the
# container image and the image name is what gets shipped over ssh.
#
# `--workload` is canonical; `--tuning_mode` is a deprecated alias kept
# working because callers still emit it (.tune/bin/wkctl, and the webui's
# Start button). Same deprecation `.tune/bin/deploy` already documents.
#
# NOTE these are two different axes that got conflated. The node-kind
# workload is kernel|op|perfmon. `--tuning_mode` downstream means something
# else entirely -- kernel|op, selecting which LUT surface a PG reader filters
# for -- and `perfmon` is not a legal value of it. So a `--tuning_mode
# perfmon` from an old caller is translated into a workload here and never
# forwarded under that name; forwarding it verbatim is what made
# pg_reader_worker reject it (its argparse has choices=['kernel','op']).
WORKLOAD="kernel"
REMAINING_ARGS=()
set -- "${EXTRA_ARGS[@]}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload|--tuning_mode)
      WORKLOAD="$2"
      shift 2
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done
EXTRA_ARGS=("${REMAINING_ARGS[@]}" --workload "$WORKLOAD")

case "$WORKLOAD" in
  kernel|op|perfmon) ;;
  *)
    echo "Error: unknown workload '$WORKLOAD' (expected kernel|op|perfmon)" >&2
    exit 1
    ;;
esac

# perfmon runs in its OWN image: a debian:13 + TheRock ROCm container built by
# `build_image.sh --workload perfmon`, which tags it
# "${CELERY_WORKER_IMAGE}-perfmon_<arch>" (build_image.sh's own IMAGE_TAG).
# Using the plain CELERY_WORKER_IMAGE for a perfmon worker is what produced
# `pull access denied for aotrtion, repository does not exist` -- docker went
# looking for an image nobody ever built under that name.
if [ "$WORKLOAD" = "perfmon" ]; then
  WORKER_IMAGE="${CELERY_WORKER_IMAGE}-perfmon_${arch}"
else
  WORKER_IMAGE="$CELERY_WORKER_IMAGE"
fi

ssh "$HOSTNAME" bash -s "$WORKER_WORKDIR" "$arch" "$WORKER_IMAGE" "$WORKLOAD" "${EXTRA_ARGS[@]}" <<'EOF'
WORKER_WORKDIR="$1"
ARCH="$2"
WORKER_IMAGE="$3"
WORKLOAD="$4"
shift 4
EXTRA_ARGS=("$@")

# PYTHONPATH selects which compiled pyaotriton the container imports.
#
# perfmon deliberately gets the same entry as kernel rather than one of its
# own: it drives native `bin/runner` executables over the exaid protocol and
# imports no pyaotriton, and installed/perfmon/<arch>/ holds subjects, not a
# Python package. A nonexistent PYTHONPATH entry is ignored by Python, so
# this is inert for perfmon -- named explicitly so the next reader does not
# have to re-derive that it was left alone on purpose.
if [ "$WORKLOAD" = "op" ]; then
  WORKER_PYTHONPATH="/wkdir/installed/test/$ARCH/lib"
else
  WORKER_PYTHONPATH="/wkdir/installed/$ARCH/lib"
fi

RUNFILE="$WORKER_WORKDIR/run/worker.containerid"

mkdir -p "$WORKER_WORKDIR/run"

if [ -f "$RUNFILE" ]; then
  echo "Worker already running or stale run file exists. Run stop first." >&2
  exit 1
fi

set -x
WORKER_CONTAINER_ID=$(docker run -d \
  --init \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --network=host \
  -e PYTHONPATH=$WORKER_PYTHONPATH \
  -e PYTHONPYCACHEPREFIX=/wkdir/run/pycache \
  --mount type=bind,source=$(realpath $WORKER_WORKDIR),target=/wkdir \
  "$WORKER_IMAGE" \
  bash -c "source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd /wkdir/aotriton.src && bash .tune/remote/worker_service.sh start /wkdir $ARCH ${EXTRA_ARGS[*]} && exec sleep infinity")

if [ -z "$WORKER_CONTAINER_ID" ]; then
  echo "Failed to start container" >&2
  exit 1
fi

echo "$WORKER_CONTAINER_ID" > "$RUNFILE"
echo "Started container: $WORKER_CONTAINER_ID"
EOF
