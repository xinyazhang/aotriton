# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared mechanics for pulling a release asset and installing the kernel
# images out of it. Source this from a build-<tag>.sh.
#
# WHAT IS SHARED AND WHAT IS NOT
#
# Downloading a URL and untarring it does not drift between releases, so it
# lives here. WHICH asset to ask for drifts a great deal, so it does not --
# each build-<tag>.sh names its own asset. Verified against the actual
# release pages:
#
#   0.9.2b, 0.10b   NO images asset exists at all. Only
#                   aotriton-<tag>-manylinux_2_28_x86_64-rocm<ver>-shared.tar.gz,
#                   which for these two releases is a full runtime that has to
#                   carry lib/aotriton.images/ -- there is nowhere else for the
#                   kernels of those releases to come from.
#   0.11b, 0.11.2b  aotriton-<tag>-images-amd-<group>.tar.gz, with gfx11xx as
#                   ONE group.
#   0.12.1b, 0.13b  same shape, but gfx11xx was split into gfx110x and gfx115x,
#                   and gfx1250 appeared. +asan variants also exist and must
#                   never be selected.
#
# Note there is no git sha in any of these names -- an earlier version of this
# code matched on one, which would never have found anything.

# fetch_release_asset <tag> <asset_name> <dest_dir>
#
# PERFMON_IMAGES_TARBALL overrides the download entirely, for a tarball
# obtained by other means. PERFMON_RELEASE_REPO overrides the repository, for
# a mirror.
fetch_release_asset() {
  local tag="$1" asset="$2" dest="$3"
  local repo="${PERFMON_RELEASE_REPO:-ROCm/aotriton}"

  mkdir -p "${dest}"

  if [ -n "${PERFMON_IMAGES_TARBALL:-}" ]; then
    echo "[release_asset] using PERFMON_IMAGES_TARBALL: ${PERFMON_IMAGES_TARBALL}" >&2
    cp "${PERFMON_IMAGES_TARBALL}" "${dest}/"
    basename "${PERFMON_IMAGES_TARBALL}"
    return 0
  fi

  # curl, not the gh CLI: one HTTP GET does not justify that dependency, and
  # it is not in the perfmon image. The URL is deterministic, so no API query
  # is needed either. GITHUB_TOKEN is honoured only for the rate limit.
  local url="https://github.com/${repo}/releases/download/${tag}/${asset}"
  local auth=()
  [ -n "${GITHUB_TOKEN:-}" ] && auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}")

  echo "[release_asset] curl ${url}" >&2
  if ! curl -fL "${auth[@]}" -o "${dest}/${asset}" "${url}"; then
    echo "[release_asset] ERROR: could not download ${url}" >&2
    echo "[release_asset]        If that asset name is wrong for this release," >&2
    echo "[release_asset]        check https://github.com/${repo}/releases/tag/${tag}" >&2
    echo "[release_asset]        and fix this tag's build script. To bypass the" >&2
    echo "[release_asset]        download, set PERFMON_IMAGES_TARBALL=/path/to.tar.gz" >&2
    return 1
  fi
  printf '%s' "${asset}"
}

# install_images_from_tarball <tarball> <install_dir>
#
# Extracts just lib/aotriton.images/ and lands it at
# <install_dir>/lib/aotriton.images/. Every one of these tarballs is rooted at
# "aotriton/", so that prefix is stripped.
install_images_from_tarball() {
  local tarball="$1" install_dir="$2"

  # Fail on a tarball that does not contain what this whole step exists for,
  # rather than "succeeding" and leaving a subject that cannot measure
  # anything. The runner loads images by dladdr, relative to the library, so a
  # missing directory would surface much later as a kernel-not-found at
  # measurement time.
  if ! tar tzf "${tarball}" | grep -q 'aotriton/lib/aotriton\.images/'; then
    echo "[release_asset] ERROR: ${tarball} contains no aotriton/lib/aotriton.images/." >&2
    echo "[release_asset]        Contents begin:" >&2
    tar tzf "${tarball}" | head -10 | sed 's/^/[release_asset]          /' >&2
    return 1
  fi

  mkdir -p "${install_dir}/lib"
  tar xzf "${tarball}" -C "${install_dir}/lib" \
      --strip-components=2 'aotriton/lib/aotriton.images'
  echo "[release_asset] installed ${install_dir}/lib/aotriton.images" >&2
}
