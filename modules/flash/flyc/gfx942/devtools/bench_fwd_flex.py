# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TFLOPS of a forward body on flex_gemm against the original, timed interleaved
(the power cap makes back-to-back runs of one kernel unfair to the other):
narrow fwd_flex.py vs fwd.py, wide fwd_wide_flex.py vs fwd_wide.py, xwide
fwd_xwide_flex.py vs fwd_xwide.py.

    AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL tree> python bench_fwd_flex.py \\
        --body wide --head_dim 256 --seqlen 8192 --causal --reps 10
"""

import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.environ['AOTRITON_FLYDSL_KERNEL_ROOT'])

import importlib  # noqa: E402

import torch  # noqa: E402

# body -> (original module, builder), (flex module, builder)
BODIES = {'narrow': (('gfx942.fwd', 'build_fmha_fwd_gfx942'),
                     ('gfx942.fwd_flex', 'build_fmha_fwd_gfx942_flex')),
          'wide': (('gfx942.fwd_wide', 'build_fmha_fwd_gfx942_wide'),
                   ('gfx942.fwd_wide_flex', 'build_fmha_fwd_gfx942_wide_flex')),
          'xwide': (('gfx942.fwd_xwide', 'build_fmha_fwd_gfx942_xwide'),
                    ('gfx942.fwd_xwide_flex', 'build_fmha_fwd_gfx942_xwide_flex'))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--body', default='narrow', choices=list(BODIES))
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--heads', type=int, default=32)
    p.add_argument('--seqlen', type=int, default=8192)
    p.add_argument('--head_dim', type=int, default=128)
    p.add_argument('--causal', action='store_true')
    p.add_argument('--dtype', default='bf16', choices=['bf16', 'f16'])
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--reps', type=int, default=10)
    a = p.parse_args()
    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float16
    torch.manual_seed(0)
    q, k, v = (torch.randn(a.batch, a.seqlen, a.heads, a.head_dim, dtype=dt, device='cuda') for _ in range(3))
    o = torch.empty_like(q)
    lse = torch.empty(a.batch * a.heads, a.seqlen, dtype=torch.float32, device='cuda')
    ct = 2 if a.causal else 0
    launches = {}
    for mod, fn in BODIES[a.body]:
        build = getattr(importlib.import_module(mod), fn)
        launches[mod.split('.')[-1] + '.py'] = build(a.head_dim, causal_type=ct, dtype_str=a.dtype)
    names = list(launches)

    def run(launch):
        launch(q.transpose(1, 2),
               k.transpose(1, 2),
               v.transpose(1, 2),
               o.transpose(1, 2),
               a.batch,
               a.seqlen,
               lse=lse,
               stream=torch.cuda.current_stream())

    outs = {}
    for name, launch in launches.items():
        run(launch)
        torch.cuda.synchronize()
        outs[name] = o.clone()
    assert torch.equal(outs[names[0]], outs[names[1]]), 'outputs differ'
    pairs = a.seqlen * a.seqlen
    if a.causal:
        pairs = a.seqlen * (a.seqlen + 1) // 2
    flops = 4 * a.batch * a.heads * pairs * a.head_dim
    tf = {name: [] for name in launches}
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for rep in range(a.reps):
        for name in (names if rep % 2 == 0 else names[::-1]):
            for _ in range(3):
                run(launches[name])
            e0.record()
            for _ in range(a.iters):
                run(launches[name])
            e1.record()
            torch.cuda.synchronize()
            tf[name].append(flops / (e0.elapsed_time(e1) / a.iters) / 1e9)
    for name in names:
        print(f'B{a.batch} H{a.heads} S{a.seqlen} D{a.head_dim} causal={a.causal} {a.dtype} {name:18s} '
              f'median {statistics.median(tf[name]):.1f} TFLOPS  (min {min(tf[name]):.1f}, max {max(tf[name]):.1f})')


if __name__ == '__main__':
    main()
