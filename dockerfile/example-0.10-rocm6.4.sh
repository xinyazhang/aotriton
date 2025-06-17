#!/bin/bash

export DOCKER_IMAGE=aotriton:manylinux_2_28-buildenv-rocm6.4
export AMDGPU_INSTALLER='https://repo.radeon.com/amdgpu-install/6.4.1/rhel/8.10/amdgpu-install-6.4.60401-1.el8.noarch.rpm'
# Set call to amdgpu-repo with AMDGPU_INSTALLER_SELECT_VERSION. Example:
# export AMDGPU_INSTALLER_SELECT_VERSION='amdgpu-repo --rocm-build=rocm-build-name/99 --amdgpu-build=999999'
# This is not needed for released ROCM build

export NOIMAGE_MODE=OFF
export TRITON_LLVM_HASH=3c709802
export AOTRITON_GIT_URL='https://github.com/ROCm/aotriton.git'
# export AOTRITON_GIT_URL='https://github.com/xinyazhang/aotriton.git'

# Cannot build all arches at once since this will need > 128 GiB tmpfs space.
# bash build.sh input tmpfs output 8e7b6dacbf "gfx90a;gfx942;gfx950;gfx1100;gfx1201;gfx1101;gfx1150;gfx1151;gfx1200"

# Build with shards
export DELETE_ONCE_COMPLETE_OPTION="--rm"  # Delete the container to reclaim tmpfs space
export AOTRITON_TARBALL_SHARD=0
bash build.sh input tmpfs output 8e7b6dacbf "gfx90a;gfx942;gfx950"  # MI
export AOTRITON_TARBALL_SHARD=1
# But be aware arches packed together should be build at once
# The clustering rules can be found at AOTRITON_ARCH_TO_PACK in v3python/gpu_targets.py
# We may automate the arch selection for sharding later, but not now
bash build.sh input tmpfs output 8e7b6dacbf "gfx1201;gfx1200"  # Navi48/44
export AOTRITON_TARBALL_SHARD=2
bash build.sh input tmpfs output 8e7b6dacbf "gfx1100;gfx1101;gfx1150;gfx1151"  # Navi3x and Strix Halo
