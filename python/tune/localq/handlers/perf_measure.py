# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Handlers for the `perf_measure` DAG -- perfmon rev2 Stage P.

DELIBERATELY EMPTY. This module exists so that R03's package split is
complete and the second DAG has an obvious home, not because the DAG is
implemented: rev2 sequences the framework work (Stage R) strictly before the
`perf_measure` DAG itself (Stage P), so that every R task is testable with
tuning alone, before a perf_measure row is ever written.

When Stage P lands, the handlers go here and follow the same two rules
`tune_kernel.py` follows:

  * Message classes are namespaced with this DAG's name --
    `perf_measure/measure`, not `measure`. Namespacing is the entire reason
    R03 renamed the tuning classes; a bare `probe` or `postprocess` here
    would collide with tune_kernel's in the worker's handler registry.

  * The DAG-START message keeps the bare name `perf_measure`, matching the
    task_queue.class value the PG reader forwards verbatim
    (pg_reader_worker.py sends `'class': task['class']`).

Registration is a list entry in gpu_worker_socket.py / cpu_worker.py, not an
edit to any shared file.
"""
