# `modules/flash/flyc/`

A bare directory of Python files, modelled on `modules/flash/kernel/`: no
`__init__.py`, not a package, flat sibling imports between the files. Anything
that wants to import from here puts this directory on `sys.path` first (the
same contract `python/compile.py:60` uses for the Triton kernel dir) — see
`python/flyc_compile.py`.

## What is here

The FlyDSL gfx1201 SDPA forward kernel and its host-side ABI layer, vendored
from FlyDSL's `kernels/attention/parity/`. See `UPSTREAM.md` for the exact
commit, the file list, and the import rewrites applied on top of the verbatim
copy — re-apply that table on every re-sync.

`flyc_polyfill.py` is the one file here that is *authored*, not vendored: a
`six`-style shim supplying helpers that exist only on the
`xinyazhang/sdpa-gfx1201-feature` branch and are not yet upstream in flydsl.
It disappears function-by-function as each one lands upstream.

## Cross-compiling without a GPU

This kernel is compiled ahead-of-time, for gfx1201, with **no GPU present and
no kernel ever launched** — see `python/flyc_bootstrap.py` and
`python/flyc_compile.py`. Two non-obvious things make that work:

- **`ROCM_PATH` must point at the directory containing `llvm/bin/ld.lld` at
  exactly that relative path** — for this environment,
  `<site-packages>/_rocm_sdk_core/lib`, *not* the more natural-looking
  `.../lib/llvm`. Getting this wrong produces `error: lld invocation failed`
  with no further diagnostic; `flyc_bootstrap.resolve_rocm_path()` exists
  specifically to avoid rediscovering this.
- **`flyc.compile()` cannot be used for a cross-compile.** It always ends in
  `_get_func_exe()`, which builds an `ExecutionEngine` and needs HIP. The
  driver calls the traced `JitFunction` directly instead, under
  `COMPILE_ONLY=1`, and reads the compiled artifact back out of
  `jf._last_compiled`.

See `PLAN.md` and `PLAN-PHASE1.md` in this directory for the full design and
task list.
