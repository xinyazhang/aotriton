# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Message handlers, one module per DAG class plus a DAG-neutral base
(perfmon rev2 R03).

    base.py          MessageHandler and the class-agnostic handlers
    tune_kernel.py   the tuning DAG
    perf_measure.py  the measurement DAG (Stage P; empty for now)

This was one 595-line module that described exactly one DAG while living
inside the framework package. Its own header carried the TODO that became
this task.

Re-exported here so existing imports keep working; new code should import
from the specific module, so that which DAG a worker registers for stays
visible at the import line.
"""

from .base import (
    MessageHandler,
    GracefulCancelRunningTaskHandler,
    MarkTaskFailedHandler,
)
from .tune_kernel import (
    TuneKernelHandler,
    PreprocessHandler,
    ProbeHandler,
    TuneImplHandler,
    WriteImplResultHandler,
    PostprocessHandler,
)

__all__ = [
    'MessageHandler',
    'GracefulCancelRunningTaskHandler',
    'MarkTaskFailedHandler',
    'TuneKernelHandler',
    'PreprocessHandler',
    'ProbeHandler',
    'TuneImplHandler',
    'WriteImplResultHandler',
    'PostprocessHandler',
]
