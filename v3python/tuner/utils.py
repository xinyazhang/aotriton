# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

TUNER_ROOT = Path(__file__).resolve().parent

def get_sql(name):
    with open((CODEGEN_ROOT / name).with_suffix('.sql'), 'r') as f:
        return f.read()
