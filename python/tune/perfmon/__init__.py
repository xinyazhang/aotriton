# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
perfmon: family-agnostic core of the AOTriton performance-monitoring
framework (perfmon-rev0.md).

This package deliberately stays torch-free and pyaotriton-free at import
time -- see pdesc.py's module docstring for why (dispatch_tasks-style CLIs
instantiate every family's PerfDesc() to build argparse subparsers).

Family-specific parts (entry-space generators, TFLOPS formulas) live outside
this package, under modules/<family>/perfmon/, loaded by
`aotriton.tune.registry.load_family_perfmon` -- mirroring the existing
tune/visperf split (D8).
"""
