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

### 0a. BLOCKER: the shared kernel helpers must come from the package

The gfx1201 kernel needs six shared FlyDSL helpers — `buffer_ops`, `kernels_common`,
`layout_utils`, `mem_ops`, `utils`, `mma/wmma_ops`. **They must not be copied into this
repo**; they are FlyDSL's, not the flash kernel's, and a copy is a fork.

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
2. `importlib.util.find_spec('_rocm_sdk_core')` → `Path(spec.origin).parent / 'lib'`
3. `/opt/rocm`

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

**If Task 0 landed on the interim** (helpers resolved from the `third_party/flydsl` checkout
rather than the wheel), `setup()` is also where the checkout root goes on `sys.path` — one
place, not scattered through the vendored files. Locate it via `AOTRITON_FLYDSL_ROOT` if set,
else `<repo>/third_party/flydsl`, and raise with the path tried if `kernels/common` is not
under it. Delete this the day the wheel ships them.

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
and confirm all 12 succeed. Also confirm the driver contains no `import fmha_tuning_gfx1201`
and no occurrence of `attn_fwd` outside a docstring.

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
| MOD | `python/codegen/linker.py` (build the flyc kdesc from its `FlycDecl`, as it does for affine at :148) |
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

Also configure with `-DAOTRITON_NOIMAGE_MODE=ON` and confirm `Fly.compile` is absent, and
run a sharded configure and confirm the merged file has the sum of the shards' lines.

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

### 7b. Generator side — `Fly.cluster` and the flatzip entry

In the same `out_dir` block as `Bare.cluster` (`root.py:192`), accumulate a
`flyc_cluster_dict` and reuse `write_cluster` unchanged — it writes both the `.nsv` manifest
on disk and the `;`-joined line. Then fold the result into the **existing** `flatzip_dict`,
exactly as the affine loop does at `root.py:225`, so the zip step needs no new bookkeeping:

```python
fodp = functional.filepack_ondisk_path        # with the flyc description as .NAME
flyc_cluster_dict.setdefault(fodp, {})[flyc_hsaco_abs] = flyc_inaks2_name(functional)
...
aks2_abs = (aks2_dir / fodp).with_suffix('.aks2').absolute().as_posix()
flatzip_dict.setdefault(fodp.parent, {})[aks2_abs] = functional.filepack_inzip_name
```

Add `'Fly.cluster'` to `shard_names` (`root.py:263`) alongside `'Fly.compile'` — same silent
failure mode as 5e if missed. `Bare.flatzip` needs no new entry in that list; it is already
there and now carries the flyc rows too.

### 7c. CMake side — two more loops

`Fly.cluster` gets an aks2 loop mirroring `ADD_FROM_CLUSTER_RULES` (`v3src/CMakeLists.txt`
:236-266). Two differences:

- `DEPENDS aotriton_v3_flyc_compile`, not `aotriton_v2_compile`.
- **No `--ignore_json`.** Affine passes it because prebuilt `.co` files have no sidecar;
  `flyc_compile` writes a real `<out>.json` (Task 3d), so the plain Bare-style invocation is
  correct and the metadata rides along.

Append the resulting `.aks2` paths to `ALL_AKS2` so they join the existing
`aotriton_v3_aks2` target. The flatzip loop at :311 then needs **no change at all** — the
flyc rows are already in `Bare.flatzip` from 7b, and `aotriton_kernel_storage_v3` picks them
up. Confirm the `__signature__` copy still fires once per arch dir (:335) rather than once
per zip.

### 7d. Install

The zips under `${AOTRITON_KERNEL_STORAGE_V2_DIR}` are already installed by the existing
rules; a new `<kernel>.zip` in an existing arch/family directory needs no new `install()`.
Verify rather than assume — a missing install line is invisible until someone unpacks a
release tarball.

**Files.**

| | path |
|---|---|
| MOD | `python/codegen/root.py` (`Fly.cluster` `LazyFile`, fold into `flatzip_dict`, `shard_names`) |
| — | `python/codegen/common.py` — **no change**; `hsaco_inaks2_name` reads `ksig.hsaco_entry_name` |
| MOD | `v3src/CMakeLists.txt` (aks2 loop for `Fly.cluster`; append to `ALL_AKS2`) |

`python/aks2.py` and `python/flatzip.py` need **no change** — flyc uses the plain Bare-style
invocation, and the flatzip loop already consumes the rows 7b folds into `Bare.flatzip`.

### Gate 7

```bash
ninja aotriton_kernel_storage_v3
find <build>/aotriton.images -name 'flyc_attn_fwd.zip'      # one per target arch dir
python -m zipfile -l <build>/aotriton.images/amd-gfx1201/flash/flyc_attn_fwd.zip | head
```

Expect one zip entry per surviving functional, named by `unified_signature`. Unpack one
`.aks2` and confirm it holds a single hsaco whose ELF is `EM_AMDGPU` / `gfx1201`. Then
`ninja install` and confirm the zip appears under the install prefix.

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
5. `Fly.compile` and `Fly.cluster` survive a sharded configure (both in `shard_names`) and
   are absent under `-DAOTRITON_NOIMAGE_MODE=ON`.
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
