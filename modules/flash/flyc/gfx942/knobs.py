# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knob classes of the gfx942 attention kernels (forward, dQ, dK/dV).

Plain data and flydsl-free: the code generator imports them to record the
resolved knobs. The policy that fills them is in the `tuning_*` modules.

Each kernel has several bodies (narrow, wide, ...), all compiled as the one
`@flyc.kernel` def named below, in the entry file named below. `body` says
which one a build uses; the entry file's primary builder maps it to a source.
"""

from dataclasses import replace


class _PolicyGridAxisOrder:
    """gfx942's grid order is per functional, so `GRID_AXIS_ORDER` is a policy
    field here: `with_flyc_fields` keeps a resolved one, and falls back to
    `FLYC_GRID_AXIS_ORDER` only when it is unset."""

    def with_flyc_fields(self):
        order = self.GRID_AXIS_ORDER
        knobs = super().with_flyc_fields()
        return knobs if order is None else replace(knobs, GRID_AXIS_ORDER=order)
