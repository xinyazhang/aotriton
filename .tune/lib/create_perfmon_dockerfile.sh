#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate the Dockerfile for the perfmon measurement image.
# Usage: create_perfmon_dockerfile.sh <workdir> <arch> [rocm_version]
#
# This is NOT a variant of create_dockerfile.sh. That script targets an image
# that ALREADY HAS ROCm (ubuntu 24.04 + distro ROCm) and only layers a venv and
# some wheels on top. Here the base is a plain debian:13 with no ROCm at all,
# and we install TheRock ROCm ourselves, from the pip wheel index, into a venv.
# Steps follow ROCm/legacy-rocm-build, branch docs/7.14.0, docs/install:
#   * debian 13 -> python3.13 (100-prerequisites.rst)
#   * python3.13 -m venv .venv                                (200-install.rst)
#   * python -m pip install --index-url <whl-multi-arch> \
#         "rocm[libraries,devel,device-<arch>]==<ver>"        (200-install.rst)
#   * rocm-sdk init      -- only valid because we take `devel` (300-post-install.rst)
#
# Two constraints shape the rest of this file:
#
#   1. The venv is owned by the INVOKING USER's uid:gid, not root. Build
#      artifacts (perfmon/subjects/<id>/) are produced inside the container and
#      read on the host; a root-owned venv would emit root-owned artifacts that
#      the host user cannot rewrite, and bind-mounted source would be written
#      as root too.
#   2. Because of (1), every command that touches the venv runs AS THAT USER --
#      hence the `su` wrapper below. Running pip as root into a user-owned venv
#      leaves a mix of root- and user-owned files, which breaks the next
#      non-root `pip install` with confusing EACCES deep inside site-packages.
#
# No torch: the perfmon runner is torch-free by construction, and omitting it
# is what guarantees torch's bundled libaotriton can never be loaded in the
# measurement process (perfmon-rev0.md D4, §4).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

. "$TUNE_ROOT/lib/config_load.sh"
. "$TUNE_ROOT/lib/sqlite3_compat.sh"

WORKDIR="$1"
ARCH="$2"
ROCM_VERSION="$3"

if [ -z "$WORKDIR" ] || [ -z "$ARCH" ]; then
  cat >&2 <<EOF
Usage: $0 <workdir> <arch> [rocm_version]

  <workdir>       Project working directory (holds config.rc and workers.db)
  <arch>          GPU architecture, e.g. gfx942. Selects the ROCm wheel's
                  device extra (rocm[...,device-<arch>]), so it is required --
                  device-all would pull every architecture's payload.
  [rocm_version]  Exact ROCm version, e.g. 7.14.0. Defaults to
                  \`perfmon::default_rocm\` in workers.db.
EOF
  exit 1
fi

load_config "$WORKDIR"

# ROCm version: argument wins, else workers.db. Never defaulted -- it is half
# of a subject's identity ((AOTriton tag, ROCm) pair), so guessing it would
# mislabel whatever gets built.
if [ -z "$ROCM_VERSION" ]; then
  ROCM_VERSION=$(sqlite3 "$WORKDIR/workers.db" \
    "SELECT COALESCE(value,'') FROM config WHERE key='perfmon::default_rocm'" 2>/dev/null || true)
fi

if [ -z "$ROCM_VERSION" ]; then
  echo "Error: no ROCm version given and perfmon::default_rocm is not set in" >&2
  echo "       $WORKDIR/workers.db. Set it to an exact version (e.g. 7.14.0)," >&2
  echo "       never a truncated one (7.14), or pass it as the third argument." >&2
  exit 1
fi

# The DEF's dependencies (requirements-def.txt) are installed into the venv at
# image build: they are common to every DEF workload, unlike the aotriton
# package itself, which start_worker.sh installs at container launch so it
# tracks the bind-mounted checkout.
#
# The files are COPYied into the image and handed to `pip install -r`. pip
# already resolves the `-r requirements.txt` include that requirements-def.txt
# opens with; an earlier version of this script inlined a flattened package
# list into the Dockerfile instead, which reimplemented that resolution in
# shell to avoid transferring two extra files. Copying the files is smaller and
# keeps pip's semantics.
AOTRITON_ROOT="$(cd "$TUNE_ROOT/.." && pwd)"

if [ ! -f "$AOTRITON_ROOT/requirements-def.txt" ]; then
  echo "Error: $AOTRITON_ROOT/requirements-def.txt not found." >&2
  echo "       It names what python/tune/{pq,localq,exaid} import, and this" >&2
  echo "       image has no other source for them." >&2
  exit 1
fi

# Base image. Falls back to the worker base only because a fresh config.rc
# already sets it to debian:13; anything else should be set explicitly.
PERFMON_IMAGE_BASE="${PERFMON_IMAGE_BASE:-${CELERY_WORKER_IMAGE_BASE:-debian:13}}"

# debian:13 ships python3.13 as `python3`; the venv module is a separate
# package. Keep this table honest rather than assuming a version -- an older
# base silently building against a different interpreter is exactly the kind of
# drift that only shows up as an ABI error much later.
case "$PERFMON_IMAGE_BASE" in
  debian:13|debian:13-*|debian:trixie*)
    PY=python3.13
    PY_APT="python3.13 python3.13-venv"
    ;;
  debian:12|debian:12-*|debian:bookworm*)
    PY=python3.11
    PY_APT="python3.11 python3.11-venv"
    ;;
  *)
    echo "Error: unsupported perfmon base image '$PERFMON_IMAGE_BASE'." >&2
    echo "       Add its python version to the case block in $0 -- do not" >&2
    echo "       guess, the ROCm wheels are built per interpreter version." >&2
    exit 1
    ;;
esac

# Run the venv as the invoking user so artifacts land host-writable.
BUILD_UID="${PERFMON_BUILD_UID:-$(id -u)}"
BUILD_GID="${PERFMON_BUILD_GID:-$(id -g)}"
BUILD_USER="${PERFMON_BUILD_USER:-builder}"

if [ "$BUILD_UID" = "0" ]; then
  echo "Warning: building as uid 0; artifacts will be root-owned on the host." >&2
fi

# The venv location comes from config.rc, derived from CELERY_WORKER_PYTHON
# the same way create_dockerfile.sh derives it (`/venv/bin/python` -> `/venv`).
# config.rc is the single place the image's python lives, so hardcoding a path
# here would silently diverge from it: the workers would look for their
# interpreter at CELERY_WORKER_PYTHON while the image had built one elsewhere.
if [ -z "$CELERY_WORKER_PYTHON" ]; then
  echo "Error: CELERY_WORKER_PYTHON not set in $WORKDIR/config.rc" >&2
  echo "       It names the venv's interpreter (e.g. /venv/bin/python) and is" >&2
  echo "       what fixes the venv location for this image." >&2
  exit 1
fi
VENV="$(dirname "$(dirname "$CELERY_WORKER_PYTHON")")"

if [ "$VENV" = "/" ] || [ "$VENV" = "." ]; then
  echo "Error: CELERY_WORKER_PYTHON='$CELERY_WORKER_PYTHON' does not look like" >&2
  echo "       <venv>/bin/python -- refusing to treat '$VENV' as the venv root." >&2
  exit 1
fi

# Where the checkout gets bind-mounted at run time. Like ${VENV} it sits
# directly under /, so it must be created and chown'd by root at build time
# (see the root layer in the Dockerfile below).
WORKMOUNT="${PERFMON_WORKMOUNT:-/work}"

PIP_INDEX="https://repo.amd.com/rocm/whl-multi-arch/"
ROCM_SPEC="rocm[libraries,devel,device-${ARCH}]==${ROCM_VERSION}"

# Named <workload>.<arch>, beside the tuning worker's Dockerfile in the same
# image.build/ directory -- one directory to generate, sync and reason about.
#
# The workload segment is what separates images that serve different DAGs; the
# arch segment is needed because this image bakes an arch-specific ROCm payload
# (rocm[...,device-<arch>]), so a shared name would let a second arch silently
# overwrite the first and leave the wrong wheels in the image. The tuning
# worker's own `Dockerfile` carries neither segment because it is one image for
# every arch and it predates this scheme.
WORKLOAD="perfmon"
IMAGE_BUILD_DIR="$WORKDIR/image.build"
DOCKERFILE="$IMAGE_BUILD_DIR/Dockerfile.${WORKLOAD}.${ARCH}"
mkdir -p "$IMAGE_BUILD_DIR"

# Stage the requirements files into the context, in their own subdirectory so
# they arrive together and `-r` includes still resolve beside each other.
# Rebuilt from scratch each run so a file deleted upstream does not survive here
# and keep satisfying an include that should have started failing.
#
# All of requirements*.txt, not just requirements-def.txt: the include graph is
# pip's to walk, and guessing which files it will reach is how the missing one
# turns into a docker build failure.
REQ_CONTEXT_DIR="$IMAGE_BUILD_DIR/requirements"
rm -rf "$REQ_CONTEXT_DIR"
mkdir -p "$REQ_CONTEXT_DIR"
cp "$AOTRITON_ROOT"/requirements*.txt "$REQ_CONTEXT_DIR/"

cat > "$DOCKERFILE" <<EOF
# Auto-generated by .tune/lib/create_perfmon_dockerfile.sh
# WARNING: This file is overwritten on every run. Do not edit by hand.
#
#   base   ${PERFMON_IMAGE_BASE}
#   python ${PY}
#   ROCm   ${ROCM_VERSION} (TheRock wheels, ${ARCH})
#   venv   ${VENV}, owned by ${BUILD_UID}:${BUILD_GID}
FROM ${PERFMON_IMAGE_BASE}

# Base image has no ROCm and no compiler. git/cmake/ninja are for building the
# perfmon runner and the AOTriton shim; ca-certificates is needed before pip
# can reach an https index at all.
#
# pkg-config and liblzma-dev are for the AOTriton shim builds, not for perfmon
# itself: every release tag from 0.9.2b through 0.13b has
# \`pkg_search_module(LZMA REQUIRED liblzma)\` in its top-level CMakeLists.txt,
# so without them each subject build dies at configure time.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      ca-certificates curl git cmake ninja-build build-essential \\
      pkg-config liblzma-dev \\
      ${PY_APT} \\
 && rm -rf /var/lib/apt/lists/*

# Recreate the invoking user inside the image so the venv and every artifact it
# produces are owned by the same uid:gid on the host.
#
# Every later layer says \`su ${BUILD_USER}\`, so this step must guarantee that
# NAME exists and owns that uid -- not merely that the uid exists. A base image
# that already ships an account at this uid (ubuntu has \`ubuntu\` at 1000;
# debian does not) would otherwise leave every subsequent su failing with
# "user ${BUILD_USER} does not exist". Rename it instead of skipping.
#
# This layer runs as ROOT, deliberately, and it is the only place that may.
# Both ${VENV} and ${WORKMOUNT} sit directly under /, which is root-owned, so
# the build user cannot create either one -- they have to be made here and
# handed over. Doing it in one root layer is also why no later \`su\` layer ever
# needs to mkdir: it only ever writes inside directories it already owns.
RUN set -eux; \\
    if ! getent group ${BUILD_GID} >/dev/null; then groupadd -g ${BUILD_GID} ${BUILD_USER}; fi; \\
    existing=\$(getent passwd ${BUILD_UID} | cut -d: -f1 || true); \\
    if [ -z "\$existing" ]; then \\
      useradd -m -u ${BUILD_UID} -g ${BUILD_GID} -s /bin/bash ${BUILD_USER}; \\
    elif [ "\$existing" != "${BUILD_USER}" ]; then \\
      usermod -l ${BUILD_USER} -s /bin/bash "\$existing"; \\
    fi; \\
    getent passwd ${BUILD_USER}; \\
    install -d -o ${BUILD_UID} -g ${BUILD_GID} ${VENV} ${WORKMOUNT}

# Everything below runs as the build user. \`su\` (not USER) so the image's
# default user stays root for any later layer that needs it, and so the venv is
# never touched by root -- a root-written file inside a user-owned venv makes
# the next non-root pip install fail with EACCES inside site-packages.
RUN su -s /bin/bash ${BUILD_USER} -c '\\
      set -eux; \\
      ${PY} -m venv ${VENV}; \\
      . ${VENV}/bin/activate; \\
      python -m pip install --upgrade pip'

# TheRock ROCm from the pip wheel index. \`devel\` is required: it carries the
# headers and the compiler that the AOTriton shim build needs, and it is what
# makes \`rocm-sdk init\` below meaningful (300-post-install.rst says to run
# init only when devel is installed).
RUN su -s /bin/bash ${BUILD_USER} -c '\\
      set -eux; \\
      . ${VENV}/bin/activate; \\
      python -m pip install --index-url ${PIP_INDEX} "${ROCM_SPEC}"; \\
      rocm-sdk init; \\
      echo "Resolved ROCM_PATH=\$(rocm-sdk path --root)"'

# The DEF's own dependencies (see requirements-def.txt). Installed into the
# venv at BUILD time because they are common to every DEF workload, not
# specific to perfmon -- unlike the aotriton package itself, which
# start_worker.sh installs at container launch so it tracks the bind-mounted
# checkout.
#
# Placed after the ROCm layer so that changing this list re-runs pip over a
# handful of wheels rather than a multi-gigabyte ROCm download.
COPY requirements/ /tmp/requirements/
RUN su -s /bin/bash ${BUILD_USER} -c '\\
      set -eux; \\
      . ${VENV}/bin/activate; \\
      python -m pip install -r /tmp/requirements/requirements-def.txt'

# Auto-activate the venv and derive ROCm's location for every \`docker run\`.
ENV VIRTUAL_ENV=${VENV}
ENV PATH="${VENV}/bin:\${PATH}"
ENV BASH_ENV=/etc/profile.d/perfmon.sh

# ROCM_PATH is asked of \`rocm-sdk path --root\` at use time, never read from a
# file recorded at build time. rocm-sdk is the authority on where its own tree
# lives; a cached copy is a second source of truth that goes stale the moment
# the venv is rebuilt, relocated, or the wheels are upgraded in place -- and it
# goes stale silently, pointing at a directory that still exists.
#
# The \`-z ROCM_PATH\` guard is not a cache. It means "this environment is
# already configured", and it is load-bearing for two distinct cases:
#
#   1. Nesting. BASH_ENV fires for every non-interactive bash and the su
#      payloads above nest, so without the guard each level would spawn
#      another rocm-sdk and prepend another copy of ROCM_PATH to PATH.
#   2. Override. A caller can point the image at a different ROCm tree --
#      \`docker run -e ROCM_PATH=...\` -- and this profile will respect it
#      instead of overwriting it with the venv's own. That is deliberate;
#      do not "simplify" the guard away.
#
# Note the override is all-or-nothing: setting ROCM_PATH skips the PATH and
# LD_LIBRARY_PATH exports too, so a caller who overrides it owns those as well.
# That is the same contract as case (1), where the ancestor shell already set
# all three.
RUN set -eux; \\
    mkdir -p /etc/profile.d; \\
    printf '%s\\n' \\
      '# Auto-generated. Sourced via BASH_ENV for every non-interactive bash.' \\
      '[ -n "\${VIRTUAL_ENV:-}" ] || . ${VENV}/bin/activate' \\
      'if [ -z "\${ROCM_PATH:-}" ]; then' \\
      '    ROCM_PATH="\$(rocm-sdk path --root)"' \\
      '    export ROCM_PATH' \\
      '    export PATH="\${ROCM_PATH}/bin:\${ROCM_PATH}/llvm/bin:\${PATH}"' \\
      '    export LD_LIBRARY_PATH="\${ROCM_PATH}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"' \\
      'fi' \\
      > /etc/profile.d/perfmon.sh

# Sanity check, as the build user, through the same profile a later build will
# use. Failing here beats failing three layers into an AOTriton shim build.
#
# Asserts only \`hipconfig\`, which is a stable part of every ROCm layout. The
# compiler is LISTED, not asserted: TheRock's wheel layout is not verified here,
# and hard-failing the whole image on a guessed binary name would turn a naming
# difference into a build outage. Read the listing when a shim build later
# cannot find its compiler.
RUN su -s /bin/bash ${BUILD_USER} -c 'set -eux; \\
      . /etc/profile.d/perfmon.sh; \\
      hipconfig --version; \\
      echo "--- \${ROCM_PATH}/llvm/bin ---"; \\
      ls "\${ROCM_PATH}/llvm/bin" 2>/dev/null | head -40 || echo "(no llvm/bin)"'

# ${WORKMOUNT} was created and chown'd in the root layer above, NOT left to
# WORKDIR. A WORKDIR that has to create its directory does so with ownership
# that varies by builder and version, and here it would be created after
# USER -- landing root-owned in exactly the case where the build user needs to
# write to it. Creating it explicitly makes the ownership a property of the
# image rather than of whoever built it.
USER ${BUILD_UID}:${BUILD_GID}
WORKDIR ${WORKMOUNT}
EOF

echo "Generated perfmon Dockerfile at: $DOCKERFILE"
echo "  base:  $PERFMON_IMAGE_BASE ($PY)"
echo "  arch:  $ARCH"
echo "  ROCm:  $ROCM_VERSION"
echo "  venv:  $VENV (root-created, owned by ${BUILD_UID}:${BUILD_GID})"
echo "  work:  $WORKMOUNT (root-created, owned by ${BUILD_UID}:${BUILD_GID})"
