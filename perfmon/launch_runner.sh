#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Launcher shim for a subject's `bin/runner` (perfmon rev2 R05).
#
# ExaidPerfmonWorker spawns THIS, not the C++ binary directly. The shim
# resolves which subject to run, sets up the environment around it, and
# `exec`s the binary.
#
# Usage: launch_runner.sh --preset <tag> --module <family> --gpu <id>
#
# --- Why a shim, and why it must exec ------------------------------------
#
# T13/T14 established that the runner needs an environment built around it:
# ROCm on the library path, the venv active under TheRock, a GPU mask, and a
# working directory. None of that belongs in the C++ runner, and none of it
# belongs in exaid.py, which is DAG-neutral and must not learn where AOTriton
# subjects live. A shim is also where "run this inside a nested container"
# would later go without touching exaid.
#
# GPU selection lives here specifically. rev2 R08 was withdrawn -- the runner
# gains NO `--gpu` flag -- because setting HIP_VISIBLE_DEVICES around it is
# the right layer: the runner keeps its existing assumption of device 0 of
# whatever mask it inherits (thermal_amdsmi.cc's bdf_of_this_process_gpu()).
#
# `exec` is load-bearing, not style: it makes the C++ process the DIRECT
# child of the exaid proxy, so the proxy's crash detection, pid tracking and
# EOF/exit shutdown all work unchanged. A shim that forked and waited would
# break every one of those.

set -euo pipefail

PRESET=""
MODULE=""
GPU=""

while [ $# -gt 0 ]; do
  case "$1" in
    --preset) PRESET="$2"; shift 2 ;;
    --module) MODULE="$2"; shift 2 ;;
    --gpu)    GPU="$2";    shift 2 ;;
    *) echo "launch_runner.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$PRESET" ] || [ -z "$MODULE" ] || [ -z "$GPU" ]; then
  echo "Usage: $0 --preset <tag> --module <family> --gpu <id>" >&2
  exit 2
fi

# --- Where subjects live -------------------------------------------------
# The worker runs inside the perfmon container with the workdir bind-mounted
# at /wkdir, so the default is right there; both knobs stay overridable so
# the shim is runnable by hand outside a container.
SUBJECT_ROOT="${PERFMON_SUBJECT_ROOT:-/wkdir/installed/perfmon}"

# The arch is a property of the node, not of the task. Prefer an explicit
# environment value; fall back to asking the GPU. Never guess from the
# subject tree: a node whose tree happens to hold one arch would then
# "resolve" correctly right up until a second arch is synced to it.
if [ -n "${PERFMON_ARCH:-}" ]; then
  ARCH="$PERFMON_ARCH"
elif command -v rocm_agent_enumerator >/dev/null 2>&1; then
  ARCH="$(rocm_agent_enumerator | grep -v gfx000 | head -n 1)"
else
  echo "launch_runner.sh: cannot determine the GPU arch. Set PERFMON_ARCH," >&2
  echo "                  or make rocm_agent_enumerator available on PATH." >&2
  exit 1
fi

SUBJECT_DIR="${SUBJECT_ROOT}/${ARCH}/${PRESET}"

# One runner per subject today, because the executable statically links one
# family's vtable (rev0 D4: no dlopen). --module is carried so that a subject
# shipping several family runners can select among them without changing this
# contract; it is validated against the subject's own id below.
RUNNER="${SUBJECT_DIR}/bin/runner"

if [ ! -x "$RUNNER" ]; then
  echo "launch_runner.sh: no runner for preset '${PRESET}' on ${ARCH}." >&2
  echo "                  Expected: ${RUNNER}" >&2
  if [ -d "${SUBJECT_ROOT}/${ARCH}" ]; then
    echo "                  Subjects present for ${ARCH}:" >&2
    find "${SUBJECT_ROOT}/${ARCH}" -maxdepth 1 -mindepth 1 -type d -printf '                    %f\n' >&2 2>/dev/null || true
  else
    echo "                  ${SUBJECT_ROOT}/${ARCH} does not exist -- has this node been" >&2
    echo "                  synced with 'sync_workdir.sh --workload perfmon'?" >&2
  fi
  exit 1
fi

# --- GPU mask ------------------------------------------------------------
# HIP_VISIBLE_DEVICES only. The two variables are NOT two spellings of one
# setting, which is what this used to assume: ROCR_VISIBLE_DEVICES filters
# first, at the ROCr layer, and HIP_VISIBLE_DEVICES then indexes into whatever
# ROCr left visible. Setting both to N asks for "device N of the one-device
# list containing device N", which is out of range for every N except 0:
#     runner: hipStreamCreate failed: no ROCm-capable device is detected
# It worked on gpu 0 and on nothing else.
#
# ROCR_VISIBLE_DEVICES is actively cleared rather than merely left alone. An
# inherited one would re-filter underneath this and silently change which
# physical GPU `--gpu N` means -- the same compounding, just harder to see.
#
# The runner then uses device 0 of what it inherits, per R08.
export HIP_VISIBLE_DEVICES="$GPU"
unset ROCR_VISIBLE_DEVICES

# --- ROCm and the venv ---------------------------------------------------
# Mirrors /etc/profile.d/perfmon.sh in the perfmon image, and for the same
# reason: rocm-sdk is the authority on where its own tree lives, so ROCM_PATH
# is asked at use time rather than read from anything recorded earlier. The
# `-z` guard means "this environment is not already set up", not "cache it".
#
# R05's verify is precisely that this works from a CLEAN environment -- no
# ROCM_PATH, no LD_LIBRARY_PATH pre-set -- which is why none of it may assume
# a login shell ran first.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -n "${PERFMON_VENV:-}" ] && [ -f "${PERFMON_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${PERFMON_VENV}/bin/activate"
fi

if [ -z "${ROCM_PATH:-}" ] && command -v rocm-sdk >/dev/null 2>&1; then
  ROCM_PATH="$(rocm-sdk path --root)"
  export ROCM_PATH
fi

if [ -n "${ROCM_PATH:-}" ]; then
  export PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
  export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# The subject's own libraries are found by $ORIGIN-relative RUNPATHs baked
# into bin/runner, so LD_LIBRARY_PATH is deliberately NOT extended with
# ${SUBJECT_DIR}/lib. Adding it would let a stray copy on that path win over
# the subject's own, which is the failure the RUNPATH work exists to prevent.

# cd into the subject so any relative path the runner resolves is its own.
cd "$SUBJECT_DIR"

if [ -n "${PERFMON_LAUNCH_DEBUG:-}" ]; then
  echo "launch_runner: preset=${PRESET} module=${MODULE} gpu=${GPU}" >&2
  echo "launch_runner: arch=${ARCH} subject=${SUBJECT_DIR}" >&2
  echo "launch_runner: ROCM_PATH=${ROCM_PATH:-<unset>}" >&2
  echo "launch_runner: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" \
       "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}" >&2
fi

exec "$RUNNER"
