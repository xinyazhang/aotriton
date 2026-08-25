// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The one exported symbol of libperfmon_flash@<subject> (perfmon_abi.h:
// `const pmon_family_vtable* pmon_family_entry(void)`), wired to the five
// entry points the subject's own <tag>/adapter.cc defines.
//
// This is in lib/ rather than restated in every adapter because it depends
// on nothing that drifts: perfmon_abi.h is owned by perfmon/core and is the
// stable C boundary by construction (rev0 D4 -- no AOTriton type ever
// crosses it). Every AOTriton-version-dependent decision lives on the other
// side of these five function pointers.

#include "common.h"

namespace {
const pmon_family_vtable kFlashVtable = {
    pmon_flash_enumerate_backends,
    pmon_flash_prepare,
    pmon_flash_launch,
    pmon_flash_describe,
    pmon_flash_release,
};
}  // namespace

extern "C" const pmon_family_vtable* pmon_family_entry(void) {
  return &kFlashVtable;
}
