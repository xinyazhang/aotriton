# Phase 1 — build gfx1201 hsacos from the FlyDSL forward kernel

Executable task plan. Design rationale lives in `PLAN.md` (Parts 1-6) and `SURVEY.md`;
this file is the ordered work list. Read `PLAN.md` §Phasing before starting.

**Goal.** Emit enough build rules to compile hsacos from flyc kernels, and package them.
`modules/flash/flyc/` holds the vendored FlyDSL forward kernel, and `ninja install`
produces a packaged GPU image — `aotriton.images/<arch>/flash/flyc_attn_fwd.zip` — containing
one gfx1201 HSA code object per flash `attn_fwd` functional. Nothing is dispatched and no C++
shim exists; that is Phase 2.

## Rules of engagement

- **Never launch a GPU kernel.** There is no gfx1201 device here and GPU access is blocked
  deliberately. Every step is compile-only. `COMPILE_ONLY=1` is what makes that safe.
- **Do not `pip install`.** The environment is behind a firewall. If something is missing,
  stop and ask the user to install it.
- **Do not modify `/home/xinyazha/dockerhome/meff/FlyDSL/`.** It is a separate repository.
  Vendored copies are edited in place under `modules/flash/flyc/`.
- **Commit at every logical step, and keep commits small** — the point is reviewability. One
  commit per coherent change: a `git mv` set with its import fixes, a vendored-file drop, one
  new module, one gate made to pass. Never one commit per task. Each commit must leave the
  tree importable, and its message should say what was verified. Work on the current branch.
  Do not push.

## Verified environment facts

These were measured, not inferred. Treat them as given; do not re-derive.

| Fact | Value |
|---|---|
| Python | `/home/xinyazha/.venvs/nogpu/bin/python` (3.13) |
| Installed | flydsl 0.3.1 (bundles LLVM/MLIR 24.0git incl. the AMDGPU backend), numpy, rocm-sdk 7.14 |
| **Not** installed | torch — and it must stay that way (`CMakeLists.txt:142`) |
| `ROCM_PATH` must be | `<site-packages>/_rocm_sdk_core/lib` — the dir whose `llvm/bin/ld.lld` exists |
| Env for cross-compile | `ARCH=gfx1201`, `FLYDSL_GPU_ARCH=gfx1201`, `COMPILE_ONLY=1`, `FLYDSL_RUNTIME_ENABLE_CACHE=0` |
| Expected artifact | ~13.5 KB ELF, `EM_AMDGPU`, `Flags: 0x4e, gfx1201`, symbol `flash_attn_func_aiw_kernel_0` |
| Build cost | 1.1 s (hd 64) / 1.7 s (hd 128) / 2.7 s (hd 256) |
| Required env | `AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL source root>`; no fallback exists. CMake sets it to the `third_party/flydsl-kernel.txt` clone; set it by hand for a direct `flyc_compile` run |

Traps that already cost time — do not rediscover them:

- `ROCM_PATH=.../_rocm_sdk_core/lib/llvm` looks right and **fails**. MLIR appends
  `llvm/bin/ld.lld` to it, and its documented `PATH` fallback does not work.
  The only symptom is `error: lld invocation failed` with no lld output.
- `flyc.compile()` cannot be used: it ends in `_get_func_exe()`, which builds an
  ExecutionEngine and needs HIP. Call the `JitFunction` directly instead.
- The `ARCH` and `COMPILE_ONLY` env vars are **unprefixed** (`flydsl/utils/env.py:233,238`),
  not `FLYDSL_ARCH` / `FLYDSL_COMPILE_ONLY`.

## Task 0 — prelude

Three preconditions for the real work: where the shared FlyDSL helpers come from (0a), the
ones that exist only on the feature branch (0b), and reshaping `ir/` so flyc has somewhere to
live (0c). None produces an hsaco; all have to be settled first.

### 0a. RESOLVED: the shared kernel helpers come from a pinned source clone

**Settled, not blocked.** The wheel still does not ship them, but nothing waits on that
now. Two independent FlyDSL pins:

| file | pins | supplies |
|---|---|---|
| `third_party/flydsl-compiler.txt` | `flydsl==0.3.1` | the pip wheel — the compiler |
| `third_party/flydsl-kernel.txt` | `v0.3.1` | a shallow git clone — `kernels/common` |

`CMakeLists.txt` clones the tag exactly as the `third_party/aiter.txt` block does, and
exports `AOTRITON_FLYDSL_KERNEL_ROOT` (a cache `PATH`, so pointing it at an existing
checkout skips the clone and builds against local kernel work). 9.6 MB shallow.

The released tag cannot supply all six: **`v0.3.1` has no `mma/wmma_ops.py`** — its `mma/`
carries MFMA (CDNA) helpers only, and this is a WMMA (RDNA) kernel. The gfx1201 WMMA work
sits on `xinyazhang/sdpa-gfx1201-feature`, which forked at `v0.3.0` and is contained in no
tag. So the four `utils` helpers and the one `wmma_ops` helper come from
`flyc_polyfill.py` instead; `buffer_ops` and `kernels_common` still come from the clone,
where they exist. See UPSTREAM.md "Import rewrites".

Verified byte-identical against both trees, so the polyfill bodies match the branch
originals. The section below is kept for the packaging endgame: once the wheel ships
`flydsl.kernels.common`, the clone, the env var and the polyfill all disappear together.

The original constraint stands and is why the polyfill exists rather than a copy: the
shared helpers — `buffer_ops`, `kernels_common`, `layout_utils`, `mem_ops`, `utils`,
`mma/wmma_ops` — **must not be copied into this repo**; they are FlyDSL's, not the flash
kernel's, and a copy is a fork. `flyc_polyfill.py` is authored, prefers the packaged
definition when one exists, and empties itself out as helpers land upstream.

They are also **not in the flydsl 0.3.1 wheel**, so this is a real dependency, not a
preference. Verified:

- `find $SP/flydsl -name 'buffer_ops.py'` (and the other five) → nothing
- `importlib.util.find_spec('kernels')` → `None`
- `setup.py:384,390` packages `find_packages(where='python')` only, and `kernels/` lives at
  the FlyDSL repo root, outside `python/`

**Ask the user to have the FlyDSL session ship them**, then import them. Recommended
spelling — `flydsl.kernels.common`, i.e. move or mirror `kernels/common/` under
`python/flydsl/kernels/common/` so `find_packages` picks it up:

```python
from flydsl.kernels.common import buffer_ops, kernels_common, layout_utils, mem_ops, utils
from flydsl.kernels.common.mma import wmma_ops
```

Namespaced rather than a bare top-level `kernels` package: the latter would let FlyDSL's own
sources keep their imports verbatim, but it claims an extremely generic name in every
site-packages that installs flydsl. Confirm the spelling with the user before Task 1 — it
determines four import lines here and every import inside FlyDSL's own `kernels/`.

**Interim, if that change is not ready:** resolve them from the `third_party/flydsl`
checkout instead of the wheel — `sys.path.insert(0, <flydsl repo root>)` makes
`from kernels.common import ...` work verbatim, which is exactly what the (unvendored)
`gfx1201_standalone.py` did. Still not a copy. Put the `sys.path` insert in `flyc_bootstrap`
(Task 2), not in the vendored files.

### 0b. Six helpers exist only on the feature branch

Even once `kernels/common` is importable, **a stock flydsl does not have everything the
kernel calls.** `xinyazhang/sdpa-gfx1201-feature` adds six functions to `kernels/common/`
that are not upstream — `git diff $(git merge-base HEAD upstream/main) HEAD --
kernels/common/` is +121 lines across two files, and the forward kernel uses five of them
heavily:

| function | home | direct uses in the 5 vendored files |
|---|---|---|
| `ssel(pred, a, b)` | `kernels/common/utils.py` | 20 |
| `smax(a, b)` | `kernels/common/utils.py` | 8 |
| `sdiv_rd_pow2(value, divisor)` | `kernels/common/utils.py` | 6 |
| `smin(a, b)` | `kernels/common/utils.py` | 3 |
| `wmma_f32_16x16x16(a, b, acc, acc_type=None)` | `kernels/common/mma/wmma_ops.py` | 1 |
| `vector_elem_type(value)` | `kernels/common/mma/wmma_ops.py` | 0 — but `wmma_f32_16x16x16` calls it |

**`modules/flash/flyc/flyc_polyfill.py` is where these live**, working the way Python's `six`
did: supply the branch-local additions so the vendored kernel runs against a flydsl that
predates them, and delete the module function-by-function as each lands upstream. Copy the
bodies from the branch — they are short and self-contained.

*Polyfill*, not *shim* or *compat*: `shim` is taken in this repo and means something specific
(the generated C++ dispatch layer — `shim.attn_fwd.h/cc`, `KernelShimGenerator`, the aiter
"thin C++ shim"), so reusing it for a Python module would mislead. `polyfill` appears nowhere
else here and says exactly what this does: provide what the environment lacks, and disappear
when it stops lacking it. The `flyc_` prefix stays because the directory goes on `sys.path`;
`flyc_flydsl` stuttered, since flyc and flydsl are the same word here.

Their only dependencies are already upstream, so the port needs nothing else:
`sdiv_rd_pow2` uses `is_pow2` / `pow2_shift` (both in `kernels/common/utils.py` at the
merge-base, alongside the `udiv_pow2` it is the signed counterpart to), and
`wmma_f32_16x16x16` uses `rocdl.wmma_f32_16x16x16_{f16,bf16}` plus
`flydsl.expr.utils.arith._to_raw`.

Prefer the stock package where it has the symbol: import from `flydsl.kernels.common` first
and only fall back to the local definition, so the module empties itself out as upstream
catches up rather than silently shadowing a newer implementation.

### 0c. Reorganise `ir/` by language

`KernelDescription` + `KernelSignature` are how Triton describes a kernel and one compiled
instance of it, and flyc needs the same pair. Make room before adding it. Rationale in
`PLAN.md` 6.10; the mechanics:

```
python/template_instantiation/ir/
  interface.py  axis.py  cfield.py  functional.py      # shared, untouched
  override.py   typed_choice.py  operator.py  ops/     # shared, untouched
  metro.py                                             # shared, UNTOUCHED — see below
  lib/                       NEW — helper functions the language modules CALL
  triton/kdesc.py            <- git mv ir/kdesc.py
  triton/ksignature.py       <- git mv ir/ksignature.py
  affine/kdesc.py            <- git mv ir/affine.py
  flyc/                      created empty here; filled by Task 4
```

Use `git mv` so history follows the files.

**`affine/` gets no `ksignature.py`.** An affine kernel has no functional space
(`gen_functionals` yields nothing), no perf, and its per-image unit is `co_gen()` over
prebuilt `.co` files. Not every language needs every file — the point is that when one does,
it is at the same path.

**`metro.py` does not move and does not change.** Its `for kdesc in self._kernels` at
`:116-117` is a generic local name, not a `KernelDescription` dependency: the loop only calls
`iter_kernel_slot_names()`.

**Share through `ir/lib/`, not a base class.** Abstractions are not your friend; libraries
are. Common code is **functions the per-language modules call** — no `KernelSignatureBase`
for flyc to inherit and override. Seed it with only what must agree across languages today:

- the entry-name grammar as a pure formatter over already-rendered parts —
  `entry_name(unified_signature, arch, perf='', copt='')`
- `blake2b_hash(package_path, entry)`

**psels and copts are generic concepts, not Triton's.** flyc kernels have knobs too, and
knobs are close kin to psels. What is Triton's is the particular vocabulary — `num_warps` /
`num_stages` / `waves_per_eu`, `DEFAULT_COPT`, the `COPT_*` indices, the gfx1250 workaround,
`triton_signature_string` — and that stays in `triton/ksignature.py`. The generic
`perf_section` / `copt_section` renderers stay there too **for now**, only because nothing
else calls them yet; promote them to `ir/lib/` when a second caller exists, not before.

flyc's `ksignature.py` leaves both sections empty. That is a deliberate deferral, not a
claim that flyc has no perf: the flyc tuning model is unsettled and the programmatic build
makes it likely to differ from Triton's (a builder yielding `(tuning key, callable)` tuples
is one candidate). Do not try to settle it here — empty sections keep the archive shape
right without committing to an answer.

`ir/interface.py` is untouched and is not a language module — `Interface` is already generic.
Its five implementers are `KernelDescription` (→ `triton/`), `AffineKernel` (→ `affine/`),
`MetroKernel` and `ConditionalKernel` (shared), and `Operator` (shared) — the last three
implement it without being a language. It is the shared interface of every callable AOTriton
generates: ordinary OOP, not a HAL over backend differences.

**Do not touch it in 0c**, and do not move any of its code into `ir/lib/`. If some belongs
there it is a separate task with its own justification.

**Import sites to fix — five, outside tests:**

| module | importers |
|---|---|
| `ir/kdesc.py` | `ir/operator.py`, `codegen/linker.py` |
| `ir/ksignature.py` | `ir/kdesc.py`, `codegen/autotune.py` |
| `ir/affine.py` | `codegen/linker.py:149` |

`ir/__init__.py` needs **no** change — it imports only `typed_choice`, `cfield`, `interface`,
`axis`, `override` and `functional`, none of which move.

Ignore `grep -rl kdesc`'s 32 hits: almost all are the local variable `kdesc = self._iface` in
`codegen/`, untouched by a move. Tests are extra and mechanical.

**Also fix the moved files' own imports.** Each gains a directory level, so its internal
`from .x` / `from ..y` become `from ..x` / `from ...y`. Mechanical but easy to miss, and not
covered by the five-importer table above, which counts only external callers.

`git mv ir/affine.py ir/affine/kdesc.py` needs `mkdir -p ir/affine` first (git will create an
intermediate directory for a path that does not collide with the source name, but not here,
where the target directory shares the source file's stem).

**Files (0c).**

| | path |
|---|---|
| MOV | `ir/kdesc.py` → `ir/triton/kdesc.py` |
| MOV | `ir/ksignature.py` → `ir/triton/ksignature.py` |
| MOV | `ir/affine.py` → `ir/affine/kdesc.py` |
| NEW | `ir/{lib,triton,affine,flyc}/__init__.py` |
| NEW | `ir/lib/naming.py` — the two shared items above |
| MOD | `ir/operator.py`, `codegen/linker.py`, `codegen/autotune.py` |
| MOD | `python/test/*` import lines |

### Gate 0c

Behaviour-preserving, so prove it: the suite green before and after. 40 files under
`python/test/`, only one importing triton, so it is a real check — **but `pytest` is not
installed in the nogpu venv**; ask for it rather than skipping the gate. Generated output
must be unchanged.

## Task 1 — vendor the kernel sources

Target `modules/flash/flyc/` is a **bare directory of Python files**, modelled on
`modules/flash/kernel/`: no `__init__.py`, not a package, flat sibling imports. Consumers
put the directory on `sys.path` (as `python/compile.py:60` does for the Triton kernel dir).

Copy these 5 files from
`/home/xinyazha/dockerhome/meff/FlyDSL/kernels/attention/parity/` to
`modules/flash/flyc/<basename>.py`:

```
flash_attn_func_gfx1201_aiw.py
fmha_abi_gfx1201.py
fmha_common_gfx1201.py
fmha_tuning_gfx1201.py
philox.py
```

That is the whole of the *copying*. One authored module sits beside them —
`modules/flash/flyc/flyc_polyfill.py`, the `six`-style port of the six branch-local helpers
(Task 0b). It is written, not vendored, so it is exempt from the verbatim rule and from
`UPSTREAM.md`'s rewrite table — but `UPSTREAM.md` must still list what it ports and from
which commits, since every entry is a deletion waiting on an upstream merge.

`kernels/common/*` is **not** copied (Task 0). Also not vendored:
`gfx1201_standalone.py` (its only job was a `sys.path` hack, and Task 0 replaces it), the
bwd kernels, `dropout_mask_gfx1201.py`, `flash_attn_func_gfx1201_interface.py`, `tooling/`,
and every test.

### 1a. Apply exactly these import rewrites

Only the four `gfx1201_standalone` imports change — they are the ones the deleted shim used
to serve. Two files are verbatim: `philox.py` and `fmha_tuning_gfx1201.py`.

| file | replace | with |
|---|---|---|
| `flash_attn_func_gfx1201_aiw.py` | `from gfx1201_standalone import buffer_ops, wmma_ops` | `from flydsl.kernels.common import buffer_ops`<br>`from flydsl.kernels.common.mma import wmma_ops` |
| | `from gfx1201_standalone import utils as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_common_gfx1201.py` | `from gfx1201_standalone import kernels_common` | `from flydsl.kernels.common import kernels_common` |
| | `from gfx1201_standalone import utils as common_utils` | `from flydsl.kernels.common import utils as common_utils` |

(Under the Task 0 interim, the right-hand side is `from kernels.common import ...` instead.
Whichever is chosen, it is four lines in two files.)

Leave every existing flat sibling import alone (`import fmha_abi_gfx1201 as abi`,
`import fmha_common_gfx1201 as fmha`, `from fmha_tuning_gfx1201 import (...)`,
`from philox import Philox`) — the flat layout is what those already assume.

### 1b. Make torch lazy in `fmha_abi_gfx1201.py`

Two module-scope torch imports must become function-local, because the build venv has no
torch and must never get one. Both uses are already inside functions:

- delete `import torch` (line ~33) and `from torch import float32 as torch_f32` (line ~35)
- in `lse_args`, add `from torch import float32 as torch_f32` as the first statement
- in `u64_scalar`, add `import torch` immediately before the `with torch.cuda.stream(...)`
  line (it is only reached when the caller passes a plain int; the AOT driver passes `None`)

Add a short comment at each site saying why, and note the deviation in `UPSTREAM.md`.

### 1c. Write `UPSTREAM.md`

Record: source repo `git@github.com:xinyazhang/FlyDSL.git`, branch
`xinyazhang/sdpa-gfx1201-feature`, the exact commit vendored from (get it with
`git -C /home/xinyazha/dockerhome/meff/FlyDSL rev-parse HEAD`), the 5-file list above, and
the 1a/1b rewrite tables verbatim — so a re-sync is "copy fresh, reapply this". Also record
that `kernels/common/*` is deliberately **not** vendored and comes from the flydsl package
(Task 0), so a future reader does not "fix" the missing files by copying them in.

**Files.**

| | path |
|---|---|
| NEW | `modules/flash/flyc/flash_attn_func_gfx1201_aiw.py` |
| NEW | `modules/flash/flyc/fmha_abi_gfx1201.py` |
| NEW | `modules/flash/flyc/fmha_common_gfx1201.py` |
| NEW | `modules/flash/flyc/fmha_tuning_gfx1201.py` |
| NEW | `modules/flash/flyc/philox.py` |
| NEW | `modules/flash/flyc/flyc_polyfill.py` (authored — the Task 0b port) |
| NEW | `modules/flash/flyc/UPSTREAM.md` |
| NEW | `modules/flash/flyc/README.md` |

`PLAN.md`, `PLAN-PHASE1.md` and `SURVEY.md` are already there. No `__init__.py`.

### Gate 1

```bash
cd modules/flash/flyc && /home/xinyazha/.venvs/nogpu/bin/python -c "
import sys; sys.path.insert(0, '.')
import flash_attn_func_gfx1201_aiw as m
print('import OK', m.KERNEL_NAME)
"
```

Must print `import OK flash_attn_func_gfx1201_aiw_kernel`. This will fail on the torch
import until Task 2 exists — run it after Task 2 instead, or with a throwaway stub.

Also confirm:
- `modules/flash/flyc/__init__.py` does **not** exist
- `ls modules/flash/flyc/*.py | wc -l` is **6** — the 5 vendored kernel files plus
  `flyc_polyfill.py`. If `buffer_ops.py` / `mem_ops.py` / `kernels_common.py` /
  `layout_utils.py` / `utils.py` / `wmma_ops.py` appear as separate files, someone copied
  what Task 0 says to import
- `ssel`, `smin`, `smax`, `sdiv_rd_pow2`, `wmma_f32_16x16x16` and `vector_elem_type` all
  resolve — from the stock package if it has them, else from `flyc_polyfill.py` (Task 0b)

## Task 2 — `python/flyc_bootstrap.py`

`python/` installs into the build venv as the `aotriton` package, so this is
`aotriton.flyc_bootstrap`. Two functions, both called before `flydsl` is imported.

**`resolve_rocm_path() -> str`** — return a directory `D` where `D/llvm/bin/ld.lld` is an
existing file, and set `os.environ['ROCM_PATH']` to it. Try in order:

1. `os.environ.get('ROCM_PATH')` — validate it, do not trust it
2. `rocm-sdk path --root`, when the CLI is on PATH. Returns `<site-packages>/_rocm_sdk_devel`,
   which does carry `llvm/bin/ld.lld` — do not confuse that expanded tree with the
   `rocm_sdk_devel` Python package, whose `_devel.tar` stays unexpanded until `rocm-sdk init`
3. `importlib.util.find_spec('_rocm_sdk_core')` → `Path(spec.origin).parent / 'lib'`
4. `/opt/rocm`

If none validates, raise a `RuntimeError` listing every candidate tried and why each failed.
This message is the whole point of the function: the native failure is
`error: lld invocation failed` with no lld output.

**`ensure_flydsl_importable()`** — `try: import flydsl.compiler`; on
`ModuleNotFoundError` whose `.name == 'torch'`, install a stub into `sys.modules` and retry.
Re-raise anything else. The stub needs exactly:

- `torch.{float16,bfloat16,float32,float64,bool,uint8,int8,int16,int32,int64}` — distinct
  hashable sentinel objects (they are dict keys in `_TORCH_DTYPE_TO_MLIR_BUILDER`)
- `torch.{float8_e5m2,float8_e4m3fn,float8_e5m2fnuz,float8_e4m3fnuz}` — same, optional
- `torch.Tensor` — a real empty `class` (used as a registry key and in an `issubclass` scan)
- `torch.cuda.Stream` — a real empty `class`; register `sys.modules['torch.cuda']` too
- `torch.__flydsl_aot_stub__ = True`

Return whether the stub was installed, so callers can print a one-line notice. The
`try`/`except` guard means the stub self-disables once FlyDSL drops its torch import.

Also provide **`setup(arch)`** that calls both and sets `ARCH`, `FLYDSL_GPU_ARCH`,
`COMPILE_ONLY=1`, `FLYDSL_RUNTIME_ENABLE_CACHE=0`.

**If Task 0 landed on the interim** (helpers resolved from a FlyDSL checkout rather than the
wheel), `setup()` is also where that checkout root goes on `sys.path` — one place, not
scattered through the vendored files.

**`AOTRITON_FLYDSL_ROOT` is required, with no useful fallback.** An earlier draft said "else
`<repo>/third_party/flydsl`", but Task 2.5 settled on a pinned wheel rather than a submodule,
so that directory never exists. Read the env var, and if it is unset or lacks
`kernels/common/`, raise naming the variable and what was looked for. Every invocation of
`flyc_compile` needs it set until Task 0a lands upstream, at which point the whole interim
and this variable disappear.

### Gate 2

```bash
/home/xinyazha/.venvs/nogpu/bin/python -c "
import sys; sys.path.insert(0, 'python')
import flyc_bootstrap as b
print('ROCM_PATH =', b.resolve_rocm_path())
print('stubbed   =', b.ensure_flydsl_importable())
"
```
Then re-run Gate 1; it must now pass.

**Files.**

| | path |
|---|---|
| NEW | `python/flyc_bootstrap.py` |

Lives in `python/` (installed as the `aotriton` package), not in `modules/flash/flyc/`:
it is build tooling, and `flyc/` holds only kernel sources.

## Task 2.5 — install FlyDSL into the build venv (`aotriton_venv_flydsl`)

Task 6 depends on a target named `aotriton_venv_flydsl`, and nothing creates it yet. A pinned
wheel, not a submodule: FlyDSL's `setup.py` needs a prebuilt bundled MLIR
(`build-fly/python_packages/flydsl/_mlir`), so `pip install third_party/flydsl` would not be
self-contained the way `pip install third_party/triton` is.

### 2.5a. Pin the version

`third_party/flydsl.txt`, **requirements.txt syntax**:

```
flydsl==0.3.1
```

Location and naming follow `third_party/aiter.txt`, which is the existing "pin a third-party
dependency in one file" precedent (that one holds a git tag for a clone; this one holds a
requirement for pip).

### 2.5b. Install it beside the triton wheel

In `CMakeLists.txt`, in the triton block at :242-327:

```cmake
set(AOTRITON_FLYDSL_STAMP "${VENV_SITE}/flydsl/__init__.py")
add_custom_command(OUTPUT "${AOTRITON_FLYDSL_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
  "${VENV_BIN_PYTHON}" -m pip install -r
  "${CMAKE_CURRENT_LIST_DIR}/third_party/flydsl.txt"
  VERBATIM)
add_custom_target(aotriton_venv_flydsl ALL DEPENDS "${AOTRITON_FLYDSL_STAMP}")
```

Mirroring `aotriton_venv_triton` (:325-327). Two things to get right:

- **Sentinel is `flydsl/__init__.py`**, not the bundled-MLIR extension. The obvious-looking
  `libFlyPythonCAPI.so.24.0git` carries an LLVM version in its name and would break on the
  next flydsl bump; `__init__.py` is there for every wheel.
- **Guard on `AOTRITON_NOIMAGE_MODE`** like the triton install already is (:246). A
  C++-shim-only build compiles no kernels and needs no flydsl.

No new `AOTRITON_INHERIT_SYSTEM_SITE_FLYDSL`: `AOTRITON_INHERIT_SYSTEM_SITE_TRITON` already
decides whether the venv is created `--system-site-packages` (:209), which is what exposes a
preinstalled flydsl. One switch, not two.

`numpy` is already in `requirements.txt`; nothing to add. Do **not** add torch
(`CMakeLists.txt:142`).

### 2.5c. Specify the alt-venv format; leave implementing it as a TODO

Today a venv maps to exactly one thing. `root.py:_load_altwheel_config` does
`self._altwheels[name] = Path(value)`, with one existing special case
(`value.startswith("python:")` means "use this interpreter, install nothing"). So a per-arch
flydsl is not expressible, and Phase 1 installs the same `third_party/flydsl.txt` pin into
every venv.

**The backward-compatible format**, to be recorded now and implemented when someone needs it.

The guarantee is at the **file** level: *an existing `AltWheelExample.yaml` parses and behaves
identically, with no edits.* The two forms are different YAML — a scalar is not a
one-element sequence — and the loader accepts both:

```yaml
venvs:
  # (a) SCALAR — one triton wheel. The existing form, byte-for-byte unchanged.
  navi3x: triton-3.3.0+git4280ed11-cp310-cp310-linux_x86_64.whl
  # (b) SCALAR with the "python:" prefix — use this interpreter, install nothing.
  #     The existing special case, also unchanged.
  external: python:/opt/venvs/foo/bin/python
  # (c) SEQUENCE — NEW. Several pip requirement lines, installed in order.
  vllm:
    - triton-3.4.0+git6b70e716-cp310-cp310-linux_x86_64.whl
    - flydsl==0.3.2
  gfx1201:
    - triton-3.3.0+git4280ed11-cp310-cp310-linux_x86_64.whl
    - flydsl @ git+https://github.com/ROCm/FlyDSL.git@f83759e953db4b9b0d1e0304f9c3634443a3bf3b
```

Loader rule: branch on the YAML node type, and the two branches mean **different** things —

| node | meaning |
|---|---|
| scalar, `python:` prefix | use this interpreter, install nothing. Unchanged. |
| scalar, otherwise | **a wheel path, and only a wheel path.** Unchanged. |
| sequence | NEW — pip requirement lines, installed in order. Wheel paths, version pins and PEP 508 direct references all allowed. |

`flydsl==0.3.1` is valid only in the sequence form. Keeping the scalar to wheels means the
old form has exactly one meaning rather than quietly gaining expressiveness, and it makes
"which form does this file use" answerable by looking at one line.

**Enforce it, do not just document it.** CMake runs `pip install ${WHEEL}` verbatim
(`CMakeLists.txt:319`), so a scalar `flydsl==0.3.1` would *work today by accident* — pip does
not care that the caller meant a path. Left unchecked, the two forms drift into
interchangeable and the restriction is fiction. Validate in `_load_altwheel_config`: a
non-`python:` scalar must end in `.whl`, and anything else raises pointing at the sequence
form. `Path(value)` stays as the scalar's representation, which is what makes the check
natural to write there.

Normalising internally to a list is fine for downstream code, but the *validation* differs
per branch, so it cannot be a blind `[value]` wrap.

**Caveat on that git example for FlyDSL specifically:** it illustrates the mechanism, not a
working FlyDSL install. FlyDSL's `setup.py` needs a prebuilt bundled MLIR at
`build-fly/python_packages/flydsl/_mlir`, so a plain git install of FlyDSL will fail — for
FlyDSL the practical forms are (b) a version pin from an index, or a built wheel. Say so in
the example file rather than letting the next reader discover it.

**Two readers to update, not one.** The yaml is parsed twice:

- `python/codegen/root.py:_load_altwheel_config` — for the `rules` matcher and `_venvpython`.
  Note `_altwheels` is currently write-only there; the install happens in CMake. So the change
  is `Path(value)` → a list of requirement strings, plus the `str`→`[str]` normalisation.
- `CMakeLists.txt:301-303` — a Python one-liner emitting
  `';'.join(f'{k};{v}' for k, v in d['venvs'].items())`, consumed as alternating name/wheel
  pairs by `list(POP_FRONT)` at :308-309. **A list value breaks this twice**: the alternation
  stops being 2-periodic, and a rendered list would carry `;` into a field — the same
  `list(POP_FRONT)` hazard as Task 5b. Emit one venv per line with a non-`;` separator and
  read with `file(STRINGS)`, or move the flattening into a small helper module instead of an
  inline one-liner.

Phase 1 writes the format down and does not implement it: a TODO at
`_load_altwheel_config` and the commented block above in `docs/AltWheelExample.yaml`, kept
commented so the unsupported form cannot silently break a config. This lands sooner than it
looks — `PLAN.md` 6.7 wants per-arch flyc kernel sources, and different arches may want
different flydsl versions.

**Files.**

| | path |
|---|---|
| NEW | `third_party/flydsl.txt` |
| MOD | `CMakeLists.txt` (install + `aotriton_venv_flydsl`) |
| MOD | `python/codegen/root.py` (TODO at `_load_altwheel_config`) |
| MOD | `docs/AltWheelExample.yaml` (TODO + sketch) |

No submodule and no `.gitmodules` change.

### Gate 2.5

`cmake` configures cleanly; `ninja aotriton_venv_flydsl` succeeds; in the build venv
`python -c "import flydsl; print(flydsl.__version__)"` prints `0.3.1` while
`python -c "import torch"` still fails. Configure with `-DAOTRITON_NOIMAGE_MODE=ON` and
confirm the target is absent.

**Status: PASSED.** All four assertions verified — configure clean, target builds,
`flydsl 0.3.1` present, `import torch` still `ModuleNotFoundError`, and the target is
absent from `ninja -t targets` under `-DAOTRITON_NOIMAGE_MODE=ON`. Rebuilding is a no-op
(`ninja: no work to do`), so the `flydsl/__init__.py` sentinel works. Beyond the gate: the
CMake-built venv's own python cross-compiles the kernel to the byte-identical
13496-byte hsaco (`1821491bae4d1ca3c2f1`), which is the whole Phase 1 chain end to end.

Two environment facts this gate depended on, neither obvious:

1. **`find_package(hip REQUIRED)` needs upstream #207.** Before it, `CMakeLists.txt`
   hardcoded `list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")` with no override, so a
   venv-based ROCm was unreachable no matter what `ROCM_PATH` said. #207 uses
   `$ENV{ROCM_PATH}/lib/cmake` instead. Set `ROCM_PATH=$(rocm-sdk path --root)`, which
   resolves to `_rocm_sdk_devel` — the one directory carrying **both** `lib/cmake/hip`
   (for CMake) and `llvm/bin/ld.lld` (for `flyc_bootstrap`). Note `flyc_bootstrap`'s
   candidate 3, `_rocm_sdk_core/lib`, satisfies the linker but has **no** `lib/cmake`, so
   the two resolutions are not interchangeable even though both "work".
2. **Image mode configures without triton.** Triton is an `add_custom_command`, i.e. a
   build-time rule, so `ninja aotriton_venv_flydsl` can be built on its own without
   triton present. `-DAOTRITON_NOIMAGE_MODE=ON` is NOT a way to gate this task, since
   the flydsl target lives under `if(NOT AOTRITON_NOIMAGE_MODE)`.

### Gating this offline (no PyPI route)

`pip install -r requirements.txt` hangs and the `aotriton` install fails with
`setuptools>=64 (from versions: none)`. `-DAOTRITON_INHERIT_SYSTEM_SITE_TRITON=ON` does not
help: `python -m venv` bases off the *real* interpreter, so `pyvenv.cfg` reads
`home = /usr/bin` and the build venv inherits the bare system python's site-packages, not
those of whatever venv invoked cmake.

What works is pip's own HTTP cache, which holds the wheels from earlier installs.
Reconstruct a wheelhouse from it (each cached `.body` that is a zip with a
`*.dist-info/WHEEL` is a wheel; rebuild the filename from the dist-info stem plus the
`Tag:` lines), then:

```
ROCM_PATH=$(rocm-sdk path --root) PIP_NO_INDEX=1 PIP_FIND_LINKS=<wheelhouse> cmake ...
```

All ten `requirements.txt` entries plus `flydsl==0.3.1` are recoverable this way (~103 MB
trimmed). `triton` is **not** in the cache; supply it as a prebuilt wheel instead:

```
-DAOTRITON_USE_LOCAL_TRITON_WHEEL=<abs path to triton-*.whl>
```

With that, image builds work: `ninja aotriton_venv_triton aotriton_venv_flydsl` installs
both, the venv has triton + flydsl and still **no torch**, and ~51k image rules are
generated. Verified one Triton kernel end to end — a 43,480-byte gfx1201 hsaco with
`compile_status: Complete` — so the `.hsaco` → `.aks2` → `aotriton.images/*.zip` chain
that Task 7 targets is exercisable here.

Two remaining network dependencies at configure time, both of which do work in this
container even though PyPI does not:

- `v3src/CMakeLists.txt:115` clones `https://github.com/ROCm/aiter.git` (~140 MB) under
  `if(NOT AOTRITON_NOIMAGE_MODE)`. github is reachable; PyPI is not.
- git submodule / `git fetch` traffic generally.

Note the wheel must not live in the repo root — it is not covered by `.gitignore`, so a
`git add -A` would sweep several hundred MB into a commit.

## Task 3 — `python/flyc_compile.py`

The FlyDSL analogue of `python/compile.py`, invoked as `python -m aotriton.flyc_compile`.

**It must be kernel-agnostic.** It knows how to drive *any* `@ati.flyc.kernel` description;
it must not import `fmha_tuning_gfx1201`, mention `attn_fwd`, or contain a flash-shaped
argument list. Everything kernel-specific lives in the description's function body. Mirror
`python/compile.py`'s CLI shape:

```
python -m aotriton.flyc_compile modules/flash/aot/flyc_attn_fwd.py \
    --kernel_name flyc_attn_fwd \
    --target gfx1201 \
    --signature "Q='*fp16:16' BLOCK_DMODEL=64 CAUSAL_TYPE=0 BIAS_TYPE=0 ENABLE_DROPOUT=False PADDED_HEAD=False" \
    --hints "seqlen_q=0 seqlen_k=0" \
    --out_path build/flyc/attn_fwd_hd64_f16
```

- positional `path` — the **description** module (`modules/flash/aot/flyc_attn_fwd.py`),
  not the kernel
- `--kernel_name` — the description def to drive, exactly as `compile.py` names the Triton
  kernel symbol
- `--signature` — the functional, as `key=value` pairs. Same name as `compile.py --signature`
  so one term covers both backends.
- `--hints` — the `@ati.flyc.hints` dataclass fields, same encoding
- `--target`, `--out_path`, `--timeout`, `--verbose`, `--verify`

argparse names use underscores (project CLAUDE.md).

### 3a. One parser, two separators

`parse_python` (`v3python/tune/utils.py:43-50`) splits on `;`, then on `=` with
`maxsplit=1`, then calls **`eval(v)`**. Keep the shape, fix the two problems — the eval, and
the hard-coded separator:

```python
import ast

def parse_kv(line, sep=';'):
    d = {}
    for assignment in filter(None, (a.strip() for a in line.split(sep))):
        k, v = assignment.split('=', maxsplit=1)
        d[k.strip()] = ast.literal_eval(v.strip())
    return d
```

`sep` defaults to `';'` so it is a drop-in for every existing `parse_python` caller
(`FlashEntry.parse_text`, `FlashInputMetadata.parse_text`), and `flyc_compile` passes
`sep=' '`. **Space is the wire separator for flyc** — the generator writes it that way
(5b) and the driver reads it that way, so nothing translates in between and there is one
parser, not two spellings of one.

`ast.literal_eval` accepts exactly the forms `as_text()` emits — ints, floats, quoted
strings, tuples, lists, `True`/`False`/`None` — and rejects everything else. The one
behaviour it drops is bare unquoted identifiers, which is the point: build inputs arriving
over a command line must not be able to execute anything. Put `parse_kv` somewhere shared
(`python/utils/`), not in this driver — Phase 2 wants it, and so does `v3python/tune` once
its `eval` is retired.

Constraint that comes with `sep=' '`: **no value may contain a space.** ATI functional values
are dtype strings, ints and bools, so none do. Assert it where the generator writes the line
(5b) rather than discovering it in a truncated payload.

### 3b. Drive the description

1. `flyc_bootstrap.setup(args.target)`.
2. Import the description module from `args.path` (`importlib.util.spec_from_file_location`)
   and get `fn = getattr(module, args.kernel_name)`.
3. `sys.path.insert(0, fn.__ati_node__.module_path.parent)` — the vendored kernel directory,
   so the body's `from flash_attn_func_gfx1201_aiw import ...` resolves. The bare-directory
   contract, same as `compile.py:60`.
4. Reconstruct the two objects the body expects:
   - **functional** — from `--signature`. Phase 1 may use a light stand-in exposing
     `.arch` and `.choices.<NAME>` (an attribute view over the parsed dict), since the driver
     runs in a separate process from the generator and does not have the linked IR. Keep the
     attribute surface identical to `ir.Functional` so Phase 2 can pass the real object with
     no change to any description body.
   - **hints** — the dataclass registered by `@ati.flyc.hints`, constructed from its
     defaults and updated with `--hints`. Reject unknown keys loudly; a typo must not
     silently build the default schedule.
5. `built, sidecar = fn(functional, hints)` — the description body does the tuning and
   returns **two** things: the builder's result, and a JSON-serialisable dict of whatever it
   wants recorded in the sidecar. The driver stays kernel-agnostic — it serialises the dict
   without knowing what is in it. For flyc_attn_fwd that dict is `asdict(knobs)`.

   This is a plumbing gap, not a correctness one. The knobs are applied correctly —
   `resolve_knobs` sets `block_m`, `build_..._primary` unpacks `BLOCK_M_KNOB = knobs.block_m`,
   and the kernel is built with that tile. What does not happen is them travelling back
   *out*: the body's local `knobs` goes out of scope on return, and `built` is the `_launch`
   closure, which `dir()` shows exposing only `compile` and the five `varlen_*` helpers. The
   driver has to write the sidecar and cannot see a value consumed two frames down.

### 3c. Compile and extract

Everything below is kernel-agnostic and belongs in the driver.

Capture the traced `JitFunction`. FlyDSL builders return a `_launch` closure that does not
expose it, so swap the recorder in:

```python
import fmha_abi_gfx1201 as abi          # see note
cap = {}
abi.run_compiled = lambda cache, exe, *a: cap.update(exe=exe, args=a)
built(q, k, v, o, batch, seqlen)        # records instead of compiling
jf = cap['exe']
jf(*cap['args'])                        # COMPILE_ONLY -> returns None, no engine
```

*Note:* this is the one place genericity is not yet achievable — `abi.run_compiled` and the
launch-argument shape are FlyDSL-attention specifics. Isolate them behind a single small
function with a comment saying so; the real fix is an upstream AOT entry point
(`PLAN.md` open question 2), not a cleverer driver.

The `q/k/v/o` are duck-typed descriptors, **not** torch tensors: `.shape` (4-tuple),
`.dim()`, `.stride(i)`, `.data_ptr()`, `.device`. Name the class `FakeTensor` — `abi.ptr_arg`
special-cases that exact name to a null pointer. **The shapes do not affect the artifact**
(measured: four shape/layout combinations give a byte-identical object), so any consistent
BHSD shape works. For dropout builds pass `philox_seed=None` so `u64_scalar` short-circuits
instead of trying to allocate a torch tensor.

Extract the code object:

```python
from flydsl._mlir import ir
from flydsl._mlir.dialects import gpu as gpud
with ir.Context(), ir.Location.unknown():
    m = ir.Module.parse(jf._last_compiled[1]._ir_text)
    for op in m.body.operations:
        if op.operation.name == 'gpu.binary':
            blobs = [gpud.ObjectAttr(op.objects[i]).object
                     for i in range(len(op.objects))]
```

There are **two** objects (`#rocdl.target<chip="gfx1201">` plus the `no_wave64` one the
`rocdl-attach-target` pass adds) and they are byte-identical. Assert that and emit one; raise
loudly if they ever differ. No pipeline patching is needed — flydsl's stock `format=fatbin`
already produces a bare hsaco ELF for the ROCDL target.

### 3d. Outputs

Write `<out_path>.hsaco` and `<out_path>.json`, mirroring `python/compile.py:134-140`. JSON
must contain `compile_status` (`'Complete'`; the only key `python/codegen/autotune.py:58`
reads), the kernel symbol read back from the ELF symtab, `arch`, `warp_size` (32), `shared`
(LDS bytes), the `--signature` and `--hints` strings verbatim, the sidecar dict from step 5,
and `block_m` / `block_size` — Phase 2's `grid_calculator()` needs those and they come from
two different places:

- **`block_m`** is `knobs.block_m` — resolved and used by the builder already; it just needs
  forwarding, so it rides in the step-5 dict. Measured 256 at hd 64.
- **`block_size` is NOT in the knobs** and must not be looked for there. `resolve_knobs`
  leaves `flat_work_group_size = None`, and the builder derives
  `BLOCK_SIZE = FLAT_WORK_GROUP_SIZE or NUM_WAVES * WARP_SIZE` internally
  (`flash_attn_func_gfx1201_aiw.py:404-406`). Recover it from the **pre-lowering** IR, which
  `CompiledArtifact` keeps as `_source_ir`: the `gpu.func` carries
  `known_block_size = array<i32: N, 1, 1>`. Measured 512 for hd 64. It is *not* in
  `_ir_text` — `gpu-module-to-binary` has replaced the module body by then — and not in the
  ELF either, since block size is a host launch decision never baked into the binary.

On any failure, mirror `compile.py`: write an empty `.hsaco` and a `.json` with the failure
status so the build does not stall on a missing file. Honour `--timeout` with the same
subprocess pattern.

`--verify` (default on): run `llvm-readelf -h` from `<ROCM_PATH>/llvm/bin`; assert
`EM_AMDGPU` and that the flags name `args.target`.

### Gate 3

```bash
/home/xinyazha/.venvs/nogpu/bin/python -m aotriton.flyc_compile \
  modules/flash/aot/flyc_attn_fwd.py --kernel_name flyc_attn_fwd --target gfx1201 \
  --signature "Q='*fp16:16' BLOCK_DMODEL=64 CAUSAL_TYPE=0 BIAS_TYPE=0 ENABLE_DROPOUT=False PADDED_HEAD=False" \
  --out_path /tmp/g3
llvm-readelf -h --symbols /tmp/g3.hsaco | grep -E "Machine|Flags|FUNC"
```
Expect `EM_AMDGPU`, `Flags: 0x4e, gfx1201`, `flash_attn_func_aiw_kernel_0`, ~13.5 KB.
Then sweep `BLOCK_DMODEL in {32,64,128}` x `CAUSAL_TYPE in {0,3}` x `Q in {*fp16:16,*bf16:16}`
and confirm all 12 succeed.

`CAUSAL_TYPE=3` (generalized sliding window) needs no window handling in the driver at all:
`flyc_compile.py` never enters host wrapper code, so there is no `abi.resolve_window` call to
satisfy and no sentinel to fabricate. The `window_left`/`window_right` values are runtime
kernel arguments synthesised generically like any other operand (see `synthesise_args`); the
sweep must pass for `CAUSAL_TYPE=3` with **no window-specific code anywhere** in the driver —
if a change to make this case pass looks like it needs one, that is a sign the new entry point
is still going through host code, not a sign the driver needs a window branch.

The agnosticism check itself must also be widened. The original form grepped for the two
literal strings `fmha_tuning_gfx1201` and `attn_fwd`; it passed for weeks while
`import fmha_abi_gfx1201` sat in the driver, because that import didn't match either literal.
Check instead for any `fmha_*`, `flash*`, or `attn*` module import — e.g.
`grep -nE '^\s*(import|from)\s+(fmha_|flash|attn)' python/flyc_compile.py` should return
nothing — and confirm no occurrence of `attn_fwd` outside a docstring or comment.

**Files.**

| | path |
|---|---|
| NEW | `python/flyc_compile.py` |
| NEW | `python/utils/kv.py` (the `parse_kv` from 3a) |
| MOD | `python/utils/__init__.py` (export it) |

`parse_kv` goes in the existing `python/utils/` package beside `dict2json.py` / `lazy_file.py`,
not in the driver: Task 5 wants it and so does `v3python/tune/utils.py` once its `eval` is
retired — that retirement is a **follow-up, not Phase 1**, so do not touch `v3python/` here.

## Task 4 — the `ati.flyc.*` namespace (minimal)

Copy the shape of the affine backend; it is the closest precedent.

1. `python/template_instantiation/decorators/flyc.py` — mirror `decorators/affine.py`:
   - `FlycKernelSpec(StackedSpec)` — the innermost marker, holds the module path resolved
     relative to the caller's `__file__` (copy `decorators/source.py`'s `inspect.stack()[1]`
     idiom). Public name `ati.flyc.kernel(path)`.
   - `ContextHelperSpec` — a value object holding one name, exposed as
     `ati.context_helper(name)` from the top-level `ati` namespace, **stored and otherwise
     unused in Phase 1**. It exists so `aot/flyc_attn_fwd.py` parses whole; Phase 2 gives it
     codegen meaning.
   - `FlycHintsSpec` — `ati.flyc.hints(Dataclass)`, holding the dataclass. Store it on the
     `FlycDecl` and reject a duplicate, the way `_build_tune_spec` does for `PerfSchema`.
     Phase 1 uses it only to build the defaults object the driver passes; no codegen.
     **In `decorators/flyc.py`, not `decorators/tune.py`**: `ati.tune.*` is the shared
     tuning vocabulary and every member of it feeds the LUT and the tuning DB, while this
     feeds one description's builder. `ati.affine.*` is the precedent for a
     backend-specific namespace. See `PLAN.md` 6.9.1. The *domain* question (6.9.2) stays
     open — do not try to settle it here.
2. `python/template_instantiation/specs/flyc.py` — `FlycDecl` + `collect_flyc_decl(specs)`,
   mirroring `specs/affine.py`.
   **Do not route through `describe()`.** `describe()` validates that specs claim every
   parameter of a known signature exactly once, and flyc has no parsed signature. Collect
   passively like `_finalize_affine` does: keep the disable predicate, the cite, the module
   path, the placeholder function, and the tensor/scalar specs as an inert list.
3. `python/template_instantiation/specs/finalize.py:255` — add one branch to the dispatch:
   `elif isinstance(marker, FlycKernelSpec): _finalize_flyc(jit_fn, specs)`.
4. Export `flyc` and `context_helper` from `decorators/__init__.py` and
   `template_instantiation/__init__.py` (`__all__` in both).

**Files.**

| | path |
|---|---|
| NEW | `python/template_instantiation/decorators/flyc.py` (`FlycKernelSpec`, `FlycHintsSpec`) |
| NEW | `python/template_instantiation/ir/context_helper.py` (`ContextHelper`) |
| NEW | `python/template_instantiation/specs/flyc.py` (`FlycDecl`, `collect_flyc_decl`) |
| NEW | `python/template_instantiation/ir/flyc/kdesc.py` (`KernelDescription`, an `Interface`) |
| NEW | `python/template_instantiation/ir/flyc/ksignature.py` (`KernelSignature`, calling `ir/lib/`) |
| — | `python/codegen/linker.py` — **not in Task 4.** A bare `FlycDecl` has no discovery path yet: `compiled.affines` is filled only by `FamilyCompiler.visit_affine`, reached only for a decl listed as an `@ati.operator` backend, and 5a deliberately does not register flyc as one. `ir/flyc/kdesc.py` therefore lands unwired; Task 5 gives it a route. |
| MOD | `python/template_instantiation/specs/finalize.py` (one dispatch branch at :255) |
| MOD | `python/template_instantiation/decorators/__init__.py` (re-export) |
| MOD | `python/template_instantiation/__init__.py` (`flyc`, `context_helper` in `__all__`) |

`context_helper` lands in **`ir/`, not `decorators/`** — it is not a decorator. It is never
stacked on a def; it is a *value* passed to `wires_to=`, stored on the Tensor/ScalarSpec and
read back through `apparel_of` (`triton/kdesc.py:65-71,297-320` after 0c).

The file is named for what is in it. `ir/apparel.py` was the first choice, "apparel" being
this codebase's term for the real→apparel wiring, but a filename naming the *concept the
value participates in* rather than the value itself sends a reader hunting. It is also where
the expression forms `kdesc.py:71` anticipates would go, so if a second wiring-value type
appears, revisit the name then rather than pre-generalising now.

Note `context_helper` is **not** language-specific and does not belong under `ir/triton/` or
`ir/flyc/`: any backend needing host-side translation wants it, and its Triton precedent is
`grid_calculator`'s split between the generated header and hand-written `csrc/`.

It also stays **top-level `ati.context_helper`, not `ati.flyc.*`**: it declares a member on
the generated C++ context class and has a Triton precedent (`PLAN.md` 6.9.1, boundary
paragraph).

### Gate 4

```bash
/home/xinyazha/.venvs/nogpu/bin/python -c "
import sys; sys.path.insert(0, 'python')
sys.path.insert(0, 'modules/flash')
from aot.flyc_attn_fwd import flyc_attn_fwd
n = flyc_attn_fwd.__ati_node__
print(type(n).__name__, n.module_path)
"
```
Must print `FlycDecl` and the resolved path to `flash_attn_func_gfx1201_aiw.py`. Also
confirm the registered hints dataclass round-trips:
`n.hints() == FlycFwdHints(seqlen_q=0, seqlen_k=0, num_heads=0, batch=0)`.

## Task 5 — generator side: enumerate functionals and emit `Fly.compile`

This and Task 6 are what make `ninja` actually build the kernels, which is the only way to
find out whether they build. Read the whole of both before starting either — the pipeline
has three stages and skipping one fails silently rather than loudly.

The existing Triton pipeline, end to end:

```
python/generate.py            per shard, in parallel (ThreadPoolExecutor)
  root.py:162   out_dir = build_dir/Bare.shards/<shard>/   (or build_dir when unsharded)
  root.py:176     Bare.compile   one ';'-joined line per hsaco   (write_hsaco)
  root.py:192     Bare.cluster   one line per functional + a .nsv manifest on disk
                  Bare.shim / Affine.cluster / Bare.flatzip
  root.py:263   shard_names = [...] -> concatenate every shard's file into build_dir/<name>

v3src/CMakeLists.txt
  :180  Bare.compile  -> one add_custom_command per line -> python/compile.py -> .hsaco
        :226            all of them collected into target `aotriton_v2_compile`
  :268  Bare.cluster  -> `python -m aotriton.aks2 --hsaco_manifest <nsv>` -> .aks2
        Bare.flatzip  -> pack .aks2 into the installed .zip images
```

### 5a. Enumerate the functionals

The description declares no functional axes on purpose (`PLAN.md` 6.3): the operator owns
them. Phase 1 must read the operator's list **without registering flyc as a backend** —
`@ati.backend` would pull the params-struct union and launch-arg emission into Phase 1,
which is Phase 2 work.

Give `ati.flyc.kernel` a `functionals_of='op_attn_fwd'` keyword: a read-only reference the
generator resolves against the linked IR to reach `attn_fwd`'s enumerated functionals.
Phase 2 replaces it with `@ati.backend(2, flyc_attn_fwd, 'flyc')`. Filter with the
description's `@ati.disable` predicate (`_flyc_fwd_disabled`), which already excludes
non-gfx1201, fp32 and off-ladder head dims.

### 5b. Store the payload space-separated

There is exactly one constraint, and it is in the **file format**, not in argument passing.
Verified with cmake 3.31.6 + ninja 1.12.1:

- `file(STRINGS)` does **not** corrupt lines. It escapes embedded `;` so each line stays one
  list element, and `foreach(IN LISTS)` yields one line per iteration.
- `list(POP_FRONT RULE ...)` then re-splits the line on `;`, and at that point a field
  boundary and a payload semicolon are indistinguishable. A payload with 5 semicolons turns
  a 7-field line into a 12-field one, and `POP_FRONT` hands back `Q='*fp16:16'` where the
  whole payload was expected.

So the payload stored in `Fly.compile` must not contain `;`. Write it **space-separated**,
which is what `flyc_compile --signature` reads (3a) — CMake passes the field straight through
and nothing translates in between.

Assert at write time that no value contains a space. ATI functional values are dtype strings,
ints and bools, so none do today; the assert is cheaper than debugging a truncated payload.

A `|` field separator was tried and rejected: `string(REPLACE ";" "\;")` followed by
`|`→`;` produces the right field count, but the escape does not survive `list(POP_FRONT)`
and the payload still truncates to its first key.

### 5c. Emit `Fly.compile`

Add a `write_flyc_hsaco` beside `write_hsaco` (`root.py:280`) and a `Fly.compile` `LazyFile`
in the same `out_dir` block as `Bare.compile` (`root.py:176`). Line format:

```
VENVPYTHON;HSACO;DESC;KERNEL_NAME;TGTGPU;SIGNATURE;HINTS
```

where the last two are space-separated `key=value` (5b) and reach the driver unchanged.

No perf columns — ATI does not enumerate perf variants for flyc (`PLAN.md` 6.2). That is not
the same as one hsaco per functional: today's count is one because the shipped schedule
targets long sequences and short ones fall to the Triton backend, and it will grow when
FlyDSL's tuner becomes seqlen-dependent. Nothing in the line format assumes the count, so
emitting several rows per functional later is additive. Reuse
`root.py:344 _get_venv_and_python(functional)` for the first column rather than inventing a
venv selector; it already routes per functional and composes with `SURVEY.md`'s finding that
the linker is shared by path.

**Use flyc's `KernelDescription` / `KernelSignature`, not new naming helpers.** That pair is
what Task 0c and Task 4 put in `ir/flyc/`, and it is what makes the existing helpers work
unchanged:

- `hsaco_ondisk_name(kdesc, ksig)` (`codegen/common.py:26`) and
  `hsaco_dir(build_dir, kdesc)` (`:33`) need only `.NAME` / `.FAMILY` on the description and
  a signature with `hsaco_entry_name`. Give `ir/flyc/kdesc.py` those attributes and neither
  helper changes.
- flyc's `KernelSignature` is constructed with `EMPTY_PERF_STRUCT` (`specs/tune.py:168`,
  which exists precisely for kernels with no `@ati.tune.schema`) and an empty copt list.

An earlier draft invented flyc-specific naming here. It was wrong: `codegen/autotune.py` is
`KernelSignature`-driven throughout, so a parallel scheme would forfeit the reuse 6.2 counts
on.

Note flyc has its **own** `ir/flyc/ksignature.py` (Task 0c) — it does not reuse or subclass
the Triton class, so it carries no `COMPILER_OPTIONS` and none of the Triton workarounds. In
particular `self._copts[COPT_NWARPS_INDEX] *= 2 if f.arch == 'gfx1250'` stays in
`triton/ksignature.py`: it is Triton-specific (gfx1250 borrows gfx942's tuning database and
needs double warps for a compiler issue), retained only to avoid an accidental behaviour
change, and due for deletion once gfx1250 is properly supported. Add a comment saying so
while moving the file in 0c.

### 5d. Respect the two early returns

`root.py:159` returns before any of this when `args.build_for_tuning_second_pass`, and
`root.py:169` returns when `args.noimage_mode`. Emit `Fly.compile` **after** both, in the
same block as `Bare.compile`, so flyc inherits both behaviours. A `--noimage` build that
still tries to compile kernels is the failure mode here.

### 5e. Add `Fly.compile` to the shard merge

`root.py:263` lists the files concatenated from every shard:

```python
shard_names = ['Bare.shim', 'Bare.compile', 'Bare.cluster', 'Affine.cluster', 'Bare.flatzip']
```

Add `'Fly.compile'`. **This is the step that fails silently if missed** — the per-shard file
is written, the merged one never appears, and CMake sees zero flyc rules with no error
anywhere.

**Files.**

| | path |
|---|---|
| MOD | `python/codegen/root.py` (`write_flyc_hsaco`, the `Fly.compile` `LazyFile`, `shard_names` at :263) |
| — | `python/codegen/common.py` — **no change**; `hsaco_ondisk_name` / `hsaco_dir` work as-is once `ir/flyc/kdesc.py` has `.NAME` / `.FAMILY` |
| MOD | `python/codegen/parser.py` (see below) |
| MOD | `modules/flash/aot/__init__.py` (see below) |
| MOD | `python/template_instantiation/decorators/flyc.py` (`functionals_of=` kwarg) |
| MOD | `modules/flash/aot/flyc_attn_fwd.py` (pass `functionals_of='op_attn_fwd'`) |

**A flyc description is not reachable today.** `parser.py:182` walks only
`getattr(self.aot, 'operators', [])`, and 5a deliberately does *not* register flyc as an
operator backend. So `modules/flash/aot/__init__.py` must expose a second root — e.g.
`flyc_kernels = [flyc_attn_fwd]` beside `operators` — and `parser.py` must read it. Keep the
two lists separate; folding flyc into `operators` is the `@ati.backend` change that belongs
in Phase 2.

### Gate 5

Configure a build, then:

```bash
wc -l  <build>/Fly.compile                       # hundreds of lines for gfx1201
awk -F';' '{print NF}' <build>/Fly.compile | sort -u   # must print exactly "7" (no stray ';')
cut -d';' -f5 <build>/Fly.compile | sort -u      # must print exactly "gfx1201"
cut -d';' -f2 <build>/Fly.compile | sort | uniq -d   # must print nothing (no dup outputs)
```

Also configure with `-DAOTRITON_NOIMAGE_MODE=ON` and confirm `Fly.compile` is **empty**
(0 bytes, 0 rows) — not absent. The wording used to say "absent", which is no longer
achievable: `launch_workers()` now `touch()`es every `shard_names` entry, so all four rule
files (`Bare.compile`, `Bare.cluster`, `Bare.flatzip`, `Fly.compile`) always exist and are
empty in noimage mode. That change came from the `AOTRITON_DEBUG_SKIP_TRITON_KERNELS` work,
which needed `file(STRINGS)` to have a file to read; the behaviour under test — no flyc
rules in a noimage build — is unchanged.

Then run a sharded configure and confirm the merged file has the sum of the shards' lines.

**Status: PASSED.** 288 rows for gfx1201, field count exactly 7, column 5 exactly
`gfx1201`, no duplicate outputs; `Fly.compile` 0 bytes under `-DAOTRITON_NOIMAGE_MODE=ON`
while `Bare.shim` is still generated.

## Task 6 — CMake side: build the hsacos as part of `all`

### 6a. The rule loop

In `v3src/CMakeLists.txt`, beside the `Bare.compile` loop at :180-226:

```cmake
if(EXISTS "${AOTRITON_V2_BUILD_DIR}/Fly.compile")
  file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Fly.compile" FLYC_RULES ENCODING UTF-8)
  set(ALL_FLYC_HSACOS "")
  foreach(RULE IN LISTS FLYC_RULES)
    list(POP_FRONT RULE VENVPYTHON)
    list(POP_FRONT RULE HSACO)
    list(POP_FRONT RULE DESC)
    list(POP_FRONT RULE KERNEL_NAME)
    list(POP_FRONT RULE TGTGPU)
    list(POP_FRONT RULE SIGNATURE)
    list(POP_FRONT RULE HINTS)
    add_custom_command(OUTPUT "${HSACO}"
      COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
      "${VENVPYTHON}" -m aotriton.flyc_compile
      "${DESC}" --kernel_name "${KERNEL_NAME}" --target "${TGTGPU}"
      --signature "${SIGNATURE}" --hints "${HINTS}"
      --out_path "${HSACO}" --timeout "${AOTRITON_GPU_BUILD_TIMEOUT}"
      DEPENDS aotriton_venv_flydsl
      VERBATIM)
    list(APPEND ALL_FLYC_HSACOS "${HSACO}")
  endforeach()
  add_custom_target(aotriton_v3_flyc_compile ALL DEPENDS ${ALL_FLYC_HSACOS})
endif()
```

Guarded on the file existing, so a tree without a flyc description still configures.

**`VERBATIM` is load-bearing — do not drop it.** It is what makes CMake emit a properly
quoted argument instead of backslash-escaping. Measured, same cmake/ninja as 5b:

| | argv seen by the driver |
|---|---|
| with `VERBATIM` | `--signature "Q='*fp16:16' BLOCK_DMODEL=64 ..."` — intact |
| without | `--signature "Q=*fp16:16 BLOCK_DMODEL=64 ..."` — **quotes stripped** |

CMake backslash-escapes the spaces either way, so the argument stays in one piece; what it
does not do without `VERBATIM` is quote, so `/bin/sh` eats the `'` around the dtype. The
failure is at least loud — `ast.literal_eval('*fp16:16')` raises — but it is a failure.

Note the `Bare.compile` loop at :199 has **no** `VERBATIM`. It survives only because its
payload (`ksignature.py:106`, `', '.join(...)` of Triton compile signatures) happens to
contain no shell metacharacters. That is luck, not design; worth adding `VERBATIM` there too
as a separate cleanup, but not as part of this task.

Three further differences from the Triton loop:

- `DEPENDS aotriton_venv_flydsl`, **not** `aotriton_venv_triton` — flyc needs flydsl and
  never triton (Part 4).
- No `TRITON_CACHE_DIR` / `TRITON_F32_DEFAULT` / `TRITON_STORE_BINARY_ONLY` env. Nothing
  replaces them: `flyc_bootstrap` sets `ROCM_PATH`, `ARCH` and `COMPILE_ONLY` itself, in one
  place, so the build system does not have to know them.
- `ALL` on the target, so a plain `ninja` builds the kernels. That is the point of this task
  — an unbuilt kernel is an untested kernel.

### 6b. Stop at the hsacos

Task 6 ends with `.hsaco` files on disk. Turning them into loadable images is Task 7 — kept
separate because it is a different pipeline stage (generate-time manifests plus two more
CMake loops), not because it is optional.

**Files.**

| | path |
|---|---|
| MOD | `v3src/CMakeLists.txt` (the `Fly.compile` loop + `aotriton_v3_flyc_compile`) |

Adding `VERBATIM` to the existing `Bare.compile` loop is a **separate cleanup**, not part of
this task.

### Gate 6

```bash
ninja aotriton_v3_flyc_compile
```

Every path in `Fly.compile` column 2 exists and is non-empty; spot-check three with
`llvm-readelf -h` for `EM_AMDGPU` + `gfx1201`; confirm a plain `ninja` (no explicit target)
also builds them. Then touch one description file and confirm only the affected hsacos
rebuild.

**Status: PASSED, with one caveat that is not this task's doing.** 288/288 outputs exist,
3 ELF spot-checks are `EM_AMDGPU` / `Flags: 0x4e, gfx1201`, plain `ninja` builds them.

**`BLOCK_DMODEL=48` is nondeterministic and needs retries.** On a first full pass, 4 of
288 came out as 0-byte stubs with `"compile_status": "Exception"` — all four
`BLOCK_DMODEL=48`. The exception is `flyc_compile._extract_hsaco`'s own assertion:

```
expected every gpu.binary object to be byte-identical, got 2 distinct blob(s) among 2
```

The module is serialized twice — once per attached target, `#rocdl.target<chip="gfx1201">`
and the `no_wave64` variant — and at hd 48 the two results differ in **10,178 of 19,512
bytes**. Characterised further:

- Each serialization independently lands on one of **two** possible outputs. Neither
  target is the stable one — both orderings have been observed (`plain=A, no_wave64=B`
  and the exact mirror). They agree about half the time, which matches the observed
  6 ok / 6 fail over 12 runs for two independent draws.
- The two outputs are **not** semantically distinguishable by anything the artifact
  exposes: both wave32, both `sgpr 104 / vgpr 121`, both LDS 6784, both
  `max_flat_workgroup_size 256`, both exactly 2217 instructions.
- The whole difference is late scheduling: VOPD dual-issue packing (`v_dual_mov_b32`
  31↔32, `v_dual_mul_f32` 31↔30, with compensating single `v_mov_b32`/`v_mul_f32`),
  `s_delay_alu` placement, and one `s_code_end` pad. A compiler determinism/QoI bug,
  not — on this evidence — a codegen correctness bug. Unproven without a gfx1201 GPU.

hd 64 was 8/8 clean. Deleting the empties and rebuilding converged to 0 after three
passes, and every successful run produced the same artifact, so every configuration
*can* build and the accepted artifact is stable.

This is upstream nondeterminism in FlyDSL/LLVM, not a defect in Task 5/6, and the
assertion is doing exactly the job it was added for. Two consequences:

- Gate 6 may need `find <build> -name '*.hsaco' -empty -delete && ninja` a few times.
- A CI build must not treat first-pass success as guaranteed. Worth an upstream report;
  `hd 48` is a legitimate `FLYC_HEAD_DIMS` entry (`resolve_knobs` handles it), so dropping
  it would hide the bug rather than fix it.

**Do not "fix" this by relaxing the assertion**, tempting as it looks. Picking one blob
unconditionally would make hd 48 build every time, but the artifact would then vary run to
run — the assertion is precisely what makes the accepted output reproducible, since it only
accepts when both draws agree. The trade is: keep the canary and retry (reproducible
artifact, flaky build) or relax it (reliable build, irreproducible artifact). Phase 1 keeps
the canary.

Nothing dispatches to flyc yet — it is reached through `functionals_of=`, and
`@ati.backend` registration is Phase 2 — so a missing hd-48 image is not user-visible
today. It becomes visible the moment flyc is a real backend, which is the deadline for
resolving this upstream.

## Task 7 — package into `.aks2` and `aotriton.images/*.zip`

Phase 1's deliverable is a **GPU image artifact**, not a pile of loose `.hsaco` files. This
is the last stage of the same pipeline Tasks 5 and 6 walked:

```
.hsaco  --(aotriton.aks2, one per functional)-->  .aks2
        --(aotriton.flatzip, one per kernel)  -->  aotriton.images/<arch>/<family>/<kernel>.zip
```

### 7a. What the layout should be

The existing tree, from `functional.py:225-242`:

| | value |
|---|---|
| `filepack_ondisk_path` | `<arch-dir>/<family>/<kernel>/<sha256(unified_signature)>` — one per functional |
| `filepack_inzip_name` | `unified_signature` — that functional's entry inside the zip |
| `full_flatzip_path` | `<arch-dir>/<family>/<kernel>.zip` |
| `hsaco_inaks2_name` | `ksig.hsaco_entry_name` — one hsaco's entry inside the `.aks2` |

**Give flyc its own zip**, `<arch-dir>/<family>/flyc_attn_fwd.zip`, keyed by the
description's name. This is what affine already does — `root.py:225-231` folds affine modules
into a separate `<arch>/<family>/affine_kernels.zip` rather than into `attn_fwd.zip`. Do not
merge flyc hsacos into the Triton kernel's zip: those entry names are `ksig`-derived, flyc
has no ksig, and interleaving two naming schemes in one archive buys nothing in Phase 1.

The entry name inside the `.aks2` is `ksig.hsaco_entry_name` — **read it off flyc's
`KernelSignature`, do not format it by hand.** flyc's builds it by calling `ir/lib/`'s
`entry_name(unified_signature, arch)` with perf and copt left at their empty defaults, so it
comes out as

```
;;#F;<unified_signature>;;#P;;;#CO;;;arch=<arch>
```

i.e. the sections present and empty, which is what 6.2 wants: ATI enumerates no perf variants
for flyc *today*, and today's one-hsaco-per-functional is a snapshot rather than a design
property — the shipped schedule targets long sequences, short ones fall to the Triton backend,
and a seqlen-dependent FlyDSL tuner will need to tell variants apart inside one archive.
Populating `<perf>` / `<copt>` is the flyc code generator's job in a later phase; emitting the
empty sections now means that lands without an archive migration.

The same reasoning is why the `.aks2` step is not skipped when N==1. The archive is the
framework's unit of storage, and keeping flyc's layout identical to Triton's is what lets
the later phases reuse Triton's autotune code generator instead of growing a second one.

### 7b. Generator side — reuse `Bare.cluster` and the existing flatzip entry

**`Fly.compile` is the only new rule file.** An earlier draft added `Fly.cluster` and
weighed a `Fly.flatzip` too; neither is needed. `write_cluster` (`root.py:313`) is already
backend-agnostic — it emits `<odp parts>;<paths…>` and knows nothing about how the objects
were produced — and the affine path already proves the point by sharing `flatzip_dict`.
Only the *compile* row is backend-shaped: it encodes a Triton invocation
(`num_warps`/`num_stages`/`waves_per_eu` and `python/compile.py`), and flyc needs a
different tool with a disjoint flag set, `--hints` included.

Skippability is no longer an argument for splitting either. That is now decided in the
generator per row (`AOTRITON_DEBUG_SKIP_TRITON_KERNELS`), which is finer-grained than a
file boundary could ever be — a shared file with Triton rows omitted is exactly a
flyc-only file.

So: in the same `out_dir` block that fills `cluster_dict` (`root.py:185`), add the flyc
entries to that **same** dict, and fold into the **existing** `flatzip_dict` as the affine
loop does at `root.py:225`:

```python
fodp = functional.filepack_ondisk_path        # with the flyc description as .NAME
cluster_dict.setdefault(fodp, {})[flyc_hsaco_abs] = flyc_inaks2_name(functional)
aks2_abs = (aks2_dir / fodp).with_suffix('.aks2').absolute().as_posix()
flatzip_dict.setdefault(fodp.parent, {})[aks2_abs] = functional.filepack_inzip_name
```

`shard_names` gains `'Fly.compile'` and nothing else — `Bare.cluster` and `Bare.flatzip`
are already in the list and now carry the flyc rows too.

### 7c. CMake side — nothing

Once 7b puts the flyc rows in `Bare.cluster` and `Bare.flatzip`, the existing
`ADD_FROM_CLUSTER_RULES` loop and the flatzip loop pick them up verbatim. No new aks2 loop,
no new flatzip loop, no additions to `ALL_AKS2`. The `--ignore_json` question disappears
with the separate loop: flyc rows go through the plain Bare-style invocation, which is what
they want anyway since `flyc_compile` writes a real `<out>.json` (Task 3d) — unlike affine's
prebuilt `.co` files, which is why *that* loop passes the flag.

**One thing Task 6 must get right for this to hold.** `ADD_FROM_CLUSTER_RULES` gives every
aks2 `DEPENDS aotriton_v2_compile`, so the flyc hsaco outputs have to be reachable from that
target or a flyc `.aks2` will be built before its hsaco exists. Task 6 should append the flyc
outputs to the same `AOTRITON_HSACO_RECORD` (`Bare.targets`) the Triton loop writes, so the
existing `add_custom_target(aotriton_v2_compile ALL DEPENDS ${ALL_HSACOS})` at
`v3src/CMakeLists.txt:226` covers both backends. That also means a flyc aks2 inherits the
blanket dependency described in Gate 7 — irrelevant under
`AOTRITON_DEBUG_SKIP_TRITON_KERNELS=ON`, where that target has no Triton inputs at all,
which is precisely the configuration Phase 1 closes in.

Still confirm the `__signature__` copy fires once per arch dir (:335) rather than once per
zip.

### 7d. Install

The zips under `${AOTRITON_KERNEL_STORAGE_V2_DIR}` are already installed by the existing
rules; a new `<kernel>.zip` in an existing arch/family directory needs no new `install()`.
Verify rather than assume — a missing install line is invisible until someone unpacks a
release tarball.

**Files.**

| | path |
|---|---|
| MOD | `python/codegen/root.py` (flyc rows into the existing `cluster_dict` + `flatzip_dict`) |
| — | `python/codegen/common.py` — **no change**; `hsaco_inaks2_name` reads `ksig.hsaco_entry_name` |
| — | `v3src/CMakeLists.txt` — **no change**; the existing cluster and flatzip loops already cover the flyc rows |

`python/aks2.py` and `python/flatzip.py` need **no change** either — flyc uses the plain
Bare-style invocation, and both loops already consume the rows 7b folds in. Task 7 is
generator-only.

### Gate 7

```bash
ninja aotriton_kernel_storage_v3
find <build>/aotriton.images -name 'flyc_attn_fwd.zip'      # one per target arch dir
python -m zipfile -l <build>/aotriton.images/amd-gfx1201/flash/flyc_attn_fwd.zip | head
```

Expect one zip entry per surviving functional, named by `unified_signature`. Unpack one
`.aks2` and confirm it holds a single hsaco whose ELF is `EM_AMDGPU` / `gfx1201`. Then
`ninja install` and confirm the zip appears under the install prefix.

**Status: PASSED.** `ninja` produces `aotriton.images/amd-gfx120x/flash/flyc_attn_fwd.zip`,
288 entries, all `STORED`, 2.6 MB of payload, zero stubs once the hd-48 flakiness above is
retried out. Unpacking one entry (16-byte header: magic + 3×u32, then an LZMA stream)
gives `kernels=1`, a 144-byte directory, and an embedded ELF at offset 144 that is
`AMDGPU - HSA` / `EM_AMDGPU` / `Flags: 0x4e, gfx1201`. No CMake change was needed for
Task 7, as 7c predicted.

**A failed kernel does not corrupt the archive.** `aks2.py:62-68` already handles a
zero-length blob: it writes a zero-size entry and asserts `compile_status != 'Complete'`.
So the 4 hd-48 stubs packed as ~200-byte entries rather than breaking the zip. That is the
pre-existing path Triton uses under `AOTRITON_BUILD_FOR_TUNING`.

**`.aks2` files do NOT rebuild when their hsaco changes.** `DEPENDS aotriton_v2_compile`
on a custom TARGET becomes an **order-only** edge in ninja — `ninja -t query` on any
`.aks2` shows `|| v3src/aotriton_v2_compile`. Order-only means "build that first", not
"I am dirty when it changes", so after rebuilding the 4 flaky hsacos the zip still
contained their stale stubs, and a plain `ninja` reported nothing to do. Fixing the
kernels required `rm -rf <build>/v3src/aks2` and a re-`ninja`.

This is pre-existing and applies to Triton identically (same `DEPENDS` shape), and it is
the same root cause as the cost problem below — the two are opposite faces of one
too-coarse edge: everything must be built first, yet nothing is dirty when it changes.
Both would be fixed by depending on the cluster's own hsacos, the list already written
into the `.nsv`.

*Caveat when clearing stale archives*: delete only `<build>/v3src/aks2`, never
`<build>/v3src/aotriton.images` — `write_cluster` writes the `.nsv` ZIP manifests into the
images tree at GENERATE time, so removing it breaks the build until the next `cmake`
configure regenerates them.

**Budget this gate: building ANY single `.aks2` builds EVERY hsaco in the tree — 47,766
of them for one arch.** Measured, not estimated. `v3src/CMakeLists.txt:262` gives each
aks2 rule `DEPENDS aotriton_v2_compile`, and line 226 is
`add_custom_target(aotriton_v2_compile ALL DEPENDS ${ALL_HSACOS})` over the whole
`Bare.compile` record. The per-cluster hsaco list is real but invisible to ninja: it
travels in the `.nsv` file behind `--hsaco_manifest`, so the build system substitutes a
blanket dependency on everything.

Consequence for anyone gating 7: there is no cheap `.aks2`. Asking for one attn_fwd
cluster (nominally ~17 hsacos, 7240 rules over 432 clusters) compiled 11,233 hsacos in
~18 min of wall time — about 24% of the way — before being cut off; a complete first
build is hours, and `/tmp/bld4` was already 5.6 GB at that point. Do the full image build
once, deliberately, and keep the build directory; do NOT expect to iterate on the
aks2/flatzip stage the way 7a-7d's edit-rebuild loop implies.

Worth fixing separately (out of Phase 1 scope): if the aks2 rules took `DEPENDS
${cluster_hsacos}` — the same list already written into the `.nsv` — instead of the
aggregate target, per-cluster rebuilds would cost their ~17 kernels rather than 47,766.
The FIXME at line 224 about `AOTRITON_HSACO_RECORD` being duplicated from `Bare.compile`
is adjacent to this.

**`-DAOTRITON_DEBUG_SKIP_TRITON_KERNELS=ON` sidesteps the cost while iterating.** It
empties the three `Bare.*` rule reads (compile, cluster, flatzip) so the Triton image
pipeline contributes nothing, leaving the flyc rules Tasks 6/7 add as the only image work.
Measured: 47766/3678/14 hsaco/aks2/zip rules become 0/0/0, and
`ninja aotriton_v2_compile aotriton_v3_aks2 aotriton_kernel_storage_v3` finishes in
0.024 s instead of hours. The Triton *wheel* is still installed — that part is cheap once
the wheel exists, and `aotriton_venv_triton` is untouched.

The result is deliberately an incomplete image set (configure prints a warning saying so),
so it validates that flyc rules are *emitted and run*, not that a shippable
`aotriton.images` tree was produced. Gate 7 proper still needs one full build with the
option off.

**This is also why 7b needs no new rule files.** An earlier draft of this note argued the
opposite — that sharing `Bare.flatzip` would weld the pipelines together, so flyc needed its
own `Fly.flatzip`. That was true only of the first implementation of this option, which
gated the `Bare.*` reads in CMake and therefore could only skip at file granularity. The
option now omits rows *in the generator*, so a shared file with the Triton rows left out is
already a flyc-only file, and the split buys nothing. `Fly.compile` remains the sole new
file, because its row encodes a Triton compiler invocation and flyc's is a different tool
with a disjoint flag set.

## Definition of done

1. `modules/flash/flyc/` holds exactly 5 vendored `.py` files plus the authored
   `flyc_polyfill.py`, no `__init__.py`, and `UPSTREAM.md` records the commit, every
   deviation, why `kernels/common/*` is absent, and which branch-local helpers
   `flyc_polyfill.py` ports (each one a deletion waiting on an upstream merge).
   No copy of `buffer_ops` / `kernels_common` / `layout_utils` / `mem_ops` / `utils` /
   `wmma_ops` exists anywhere under `modules/`.
2. `python -m aotriton.flyc_compile ... --signature "... BLOCK_DMODEL=64 ..."` produces a gfx1201 ELF
   with symbol `flash_attn_func_aiw_kernel_0` and a `compile_status: Complete` sidecar.
3. The matrix sweep in Gate 3 passes for all 12 combinations, and `flyc_compile.py` is
   kernel-agnostic: no `fmha_*` import, no `attn_fwd` outside a docstring, and every
   kernel-specific decision made inside the description body.
4. A full configure emits `Fly.compile`; `ninja` (no explicit target) builds every hsaco in
   it, because `aotriton_v3_flyc_compile` is an `ALL` target. `ninja install` therefore
   exercises the flyc compile path — which is the only way an unbuildable kernel gets
   noticed.
5. `Fly.compile` — the only new rule file — survives a sharded configure (it is in
   `shard_names`) and is absent under `-DAOTRITON_NOIMAGE_MODE=ON`. The flyc cluster and
   flatzip rows ride in `Bare.cluster` / `Bare.flatzip`, which are already sharded.
6. **The deliverable exists**: `aotriton.images/<arch>/flash/flyc_attn_fwd.zip` is built by
   `ninja`, contains one entry per surviving functional, and is installed by `ninja install`.
7. `import torch` still **fails** in the build venv while all of the above succeeds.
8. No GPU was used and no kernel was launched.

## Explicitly out of scope

Params struct, launch-arg vector, `wires_to` consumption, `ati.context_helper` codegen,
`FlycAttnFwdContext`, `grid_calculator()`, the LUT, **any `codegen/autotune.py` reuse**
(Phase 2 wires flyc into it as a one-kernel, default-options, no-tuning-database case —
touching it here is out of scope beyond the 0c import fix),
`modules/flash/csrc/flyc_attn_fwd.cc`,
any C++ shim at all, and numerical correctness (needs a gfx1201 device — Phase 1 stops at "the image exists
and its ELFs name the right arch"). If a task seems to
require one of these, stop and re-read `PLAN.md` §Phasing — it probably does not.

## Known hazards

- **`aotriton` is editable-installed against a FIXED path** — the main checkout's `python/`,
  not whatever tree you are standing in. Running `pytest`, `import aotriton...` or
  `python -m aotriton.flyc_compile` from a git worktree therefore exercises the **main**
  checkout's code, not your edits: a green result can mean nothing, and a red one can be
  someone else's. Force resolution first:

  ```bash
  mkdir -p /tmp/shadow_$$ && ln -sfn <worktree>/python /tmp/shadow_$$/aotriton
  PYTHONPATH=/tmp/shadow_$$ python -m pytest -q
  ```

  Confirm with `python -c "import aotriton; print(aotriton.__file__)"` that the path is under
  your worktree. Do not commit the shadow directory.

- **Do not copy `kernels/common/*` to unblock yourself.** If Task 0 is unresolved, the
  gfx1201 modules will fail to import and copying the six helpers in makes that go away —
  which is exactly the fork this plan exists to avoid. Use the Task 0 interim (`sys.path` to
  the checkout) or stop and ask.

- **The FakeTensor contract is unstable.** The last two FlyDSL commits added an
  unconditional `Q.device` read in `_launch`, which broke the prototype until the descriptor
  grew `device = None`. If Task 3 fails with an `AttributeError` on a duck-typed descriptor,
  that is this: read the current `_launch` body and add the attribute. Record each such
  addition in `UPSTREAM.md`.
- `jf._last_compiled`, `CompiledArtifact._ir_text` and `abi.run_compiled` are all private.
  They work today (verified). If one disappears, do not paper over it — report it, because
  the fix is an upstream AOT entry point, not a workaround (`PLAN.md` open question 2).
