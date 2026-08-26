# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared dispatch driver package (dispatch-perfmon-exec.md D07).

`driver.py` holds the workload-neutral part of "generate, dedup, confirm and
insert" that `dispatch_tasks.py` used to do entirely on its own, and that
perfmon dispatch (D10) would otherwise have had to copy. See `driver.py`'s
module docstring and `dispatch-perfmon.md` §5 for the survey behind this
split.
"""
