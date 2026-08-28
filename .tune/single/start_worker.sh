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
if [ "${1:-}" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
elif [ "$#" -gt 0 ]; then
  # Refuse rather than ignore. Silently dropping these is what let a caller
  # that forgot the separator start a perfmon worker from the tuning image:
  # the workload flag never arrived, WORKLOAD fell back to its default, and
  # nothing said so.
  echo "Error: extra arguments must follow a literal '--'; got: $*" >&2
  exit 1
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

# Create the run subdirectories HERE, on the host, as the ssh user -- not
# inside the container, where worker_service.sh would otherwise be the first
# to make them. It failed there with
#     mkdir: cannot create directory '/wkdir/run/pids': Permission denied
# because the container is not the workdir's owner (see the --user block
# below). Making them out here means their ownership is never in doubt, and
# it is the same reason `run/` itself has always been created here.
mkdir -p "$WORKER_WORKDIR/run" \
         "$WORKER_WORKDIR/run/pids" \
         "$WORKER_WORKDIR/run/logs" \
         "$WORKER_WORKDIR/run/pycache"

if [ -f "$RUNFILE" ]; then
  echo "Worker already running or stale run file exists. Run stop first." >&2
  exit 1
fi

# --- who the container runs as ------------------------------------------
#
# perfmon runs as the INVOKING uid:gid; tuning keeps the image's default
# (root). They differ on purpose.
#
# The perfmon image is built around a non-root user: its venv is owned by
# the build uid:gid and every RUN step goes through `su`, because -- in that
# Dockerfile's own words -- "a root-written file inside a user-owned venv
# makes the next non-root pip install fail with EACCES inside
# site-packages". Running that image as root at runtime contradicts the way
# it was built. It is also what produced the mkdir failure above: a root
# process is not the workdir's owner, and on a host that squashes container
# root (NFS root_squash, or docker userns-remap) it cannot write into a
# directory the user owns.
#
# .tune/single/build_perfmon.sh already passes --user for the same image,
# with the same reasoning recorded there ("otherwise installed/perfmon/
# comes back root-owned and the next build cannot overwrite it").
#
# Tuning is deliberately NOT changed here. It has no user-owned venv to
# protect, it works as root today, and switching it would put GPU device
# access (/dev/kfd, /dev/dri) behind group membership that has never been
# tested for it -- an untested change to a path in production use. The two
# should converge once someone can verify a non-root tuning worker on real
# hardware.
DOCKER_USER_ARGS=()
if [ "$WORKLOAD" = "perfmon" ]; then
  DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")
  # A non-root process needs the render group to open /dev/kfd and
  # /dev/dri/renderD*; root did not. Added by GID, and only if the host has
  # such a group, since --group-add on a name that does not exist is a hard
  # docker error. `video` is already added unconditionally below.
  render_gid=$(getent group render 2>/dev/null | cut -d: -f3)
  if [ -n "$render_gid" ]; then
    DOCKER_USER_ARGS+=(--group-add "$render_gid")
  fi
fi

# --- making `import aotriton` work inside the container -----------------
#
# worker_service.sh launches `python -m aotriton.tune.localq.*`, so the
# package must be importable or the broker dies at once with
#     Error while finding module specification for
#     'aotriton.tune.localq.broker_main' (ModuleNotFoundError: No module
#     named 'aotriton')
#
# setup.py maps it with package_dir {'aotriton': 'python'} -- the importable
# name and the directory name differ -- so PYTHONPATH cannot express it and
# only an install will do.
#
# Done HERE, at container launch, and deliberately NOT baked into the image:
# aotriton.src arrives as a bind mount, so it does not exist when the image is
# built, and the perfmon image's build context is image.build/ alone (kept
# COPY-free so the image depends only on the server, never on when the workdir
# was last deployed). Installing at launch is also what keeps the container
# tracking the mounted checkout rather than a copy frozen at build time.
#
# Run UNCONDITIONALLY, not guarded by an `import aotriton` check.
#
# An editable install tracks edits to modules it already knows about, but not
# new ones: `find_packages()` runs at install time and its result is baked
# into the generated finder, so a package that appears in a later deploy is
# invisible to an install made before it existed. python/tune/dispatch/ and
# python/tune/perfmon/ are both recent examples. A guard would therefore keep
# a container pinned to whatever the checkout looked like the first time it
# started, which is exactly the staleness the bind mount exists to avoid.
#
# The cost is one pip run per container launch, on an already-populated venv.
#
# --no-deps is load-bearing: the perfmon venv is deliberately torch-free
# (create_perfmon_dockerfile.sh), and resolving this project's dependencies is
# exactly how torch would come back.
#
# This works only because the venv is owned by the uid the container runs as
# -- the image bakes the REMOTE host's uid (build_image.sh) and --user passes
# the same one below. With the server's uid baked in, as it was, pip would
# fail with EACCES inside site-packages.
#
# perfmon only. The tuning image installs its requirements at build time and
# works today; adding a pip step there would be an untested change to a path
# in production use.
PKG_SETUP=""
if [ "$WORKLOAD" = "perfmon" ]; then
  PKG_SETUP="python -m pip install -q -e . --no-deps && "
fi

set -x
WORKER_CONTAINER_ID=$(docker run -d \
  --init \
  "${DOCKER_USER_ARGS[@]}" \
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
  bash -c "source /wkdir/config.rc && source \$(dirname \$CELERY_WORKER_PYTHON)/activate && cd /wkdir/aotriton.src && ${PKG_SETUP}bash .tune/remote/worker_service.sh start /wkdir $ARCH ${EXTRA_ARGS[*]} && exec sleep infinity")

if [ -z "$WORKER_CONTAINER_ID" ]; then
  echo "Failed to start container" >&2
  exit 1
fi

echo "$WORKER_CONTAINER_ID" > "$RUNFILE"
echo "Started container: $WORKER_CONTAINER_ID"
EOF
