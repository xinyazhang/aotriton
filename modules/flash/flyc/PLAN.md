# `modules/flash/flyc`: vendor the FlyDSL gfx1201 forward kernel and cross-compile it to hsaco

> **Status: draft, not approved for execution.** Open questions below are unresolved; resolve
> them before any of Parts 1-5 are implemented.

## Open questions / known flaws

Nothing below is settled. Edit freely.

1. ~~`kernels/` as a vendored top-level package name.~~ **Resolved.** `flyc/` is a bare
   collection of Python files, exactly like `modules/flash/kernel/`. See Part 1.
2. **Three private hooks into FlyDSL internals.** The driver depends on
   `fmha_abi_gfx1201.run_compiled` (monkeypatched to capture the traced launcher),
   `JitFunction._last_compiled`, and `CompiledArtifact._ir_text`. None is public API, so a
   FlyDSL refactor breaks the build silently. Alternative: add a supported AOT entry point to
   FlyDSL and depend on that — which reorders the work, putting Part 5 first.
3. ~~The `_0` suffix on the exported kernel symbol.~~ **Resolved by measurement.** Two builds
   in one process both export `flash_attn_func_aiw_kernel_0` — the counter is per
   `CompilationContext`, not global, so the name does not drift with build order. The driver
   still reads it from the ELF rather than predicting it, but it is not a landmine.
4. ~~Flat-namespace generality.~~ **Resolved: the shared helpers move into the flydsl wheel**
   and are imported, not vendored, so `flyc/` holds only the five gfx1201 files and no
   generic name (`utils.py`, `mem_ops.py`) is claimed in a directory that goes on `sys.path`.
   Remaining: the upstream packaging change is a **blocker** — flydsl 0.3.1 ships none of the
   six — and the `flydsl.kernels.common` vs top-level `kernels` spelling needs confirming.
5. ~~Building FlyDSL from source in CMake is not equivalent to Triton.~~ **Resolved: pin a
   wheel.** `third_party/flydsl.txt` (`flydsl==0.3.1`, requirements syntax, following
   `third_party/aiter.txt`) installed beside the triton wheel. No submodule, no from-source
   path. Remaining: per-venv flydsl is not expressible in `docs/AltWheelExample.yaml` /
   `root.py:_load_altwheel_config`, left as a TODO in both.
6. **How is the linker staged?** `SURVEY.md` shows ROCm contributes exactly one file
   (`ld.lld`, self-contained in a 143 MB subtree) versus 1.4 GB for the whole
   `rocm-sdk-core` wheel. Extract the subset during the build, or install the wheel whole and
   just document it? Packaging decision, not made.
7. **Device bitcode.** This kernel needs no `ocml`/`ockl`, so `<ROCM_PATH>/amdgcn/bitcode` was
   provably unnecessary — but that is unverified for kernels that do reference them. Ship the
   3.3 MB directory anyway?
8. ~~`ati.expr`~~ **Resolved: `ati.context_helper` (Part 6.5)** — declares a member function
   on the generated context class, hand-implemented in `modules/flash/csrc/`, exactly as
   `grid_calculator` already is. Remaining sub-question: `BLOCK_M`/`BLOCK_SIZE` are knobs,
   not perf fields, so `grid_calculator()` has no way to reach them yet (Part 6.3).
9. **Per-arch ABI (Part 6.7).** A unified `flash_attn_func_interface` implies
   `@ati.flyc.kernel` takes an arch→source map. If the gfx942/gfx950 launchers have a
   different kernarg list, the `@ati.tensor`/`@ati.scalar` block must become
   arch-conditional too. Decide before the second arch lands.
10. **`num_heads` pinning (Part 6.3).** Pinned to 1 because `STRIDE_TOKEN` is only read
    under `STRIDES_CONSTEXPR`. That is a fact about today's kernel, not a contract — it
    needs an assertion, or an upstream note that the field is AOT-irrelevant.
11. **Untriaged.** Additional flaws raised but not yet written down.

> **Executing Phase 1?** Use `PLAN-PHASE1.md` — the ordered task list with verification
> gates. This file is the design rationale behind it.

## Phasing

**Phase 1 — produce hsacos.** Source in, one gfx1201 code object per functional out, on
disk beside a JSON sidecar. Nothing is dispatched, nothing is linked into the runtime.
**Phase 2 — wire to the operator.** Params struct, launch-arg vector, context helpers,
`grid_calculator`, LUT, C++ shim.

The split is cleaner than it first looks: **the entire ABI block of `aot/flyc_attn_fwd.py`
is Phase 2.** Every `wires_to=`, every `rank=`/`strides=`, all four `ati.context_helper`s,
and the launch grid exist to build the kernarg vector and the params struct — none of it
affects which hsacos are produced or what is in them. Phase 1 needs only three things from
the description: which functionals, how to turn one into a builder call, and where the
module is.

| Part | Phase | note |
|---|---|---|
| 1 vendor sources | 1 | |
| 2 bootstrap (ROCM_PATH + torch stub) | 1 | |
| 3 `python/flyc_compile.py` | 1 | |
| 4 FlyDSL as a pinned build dep | 1 | |
| 5 upstream torch removal | 1 | blocking: `CMakeLists.txt:142` forbids torch in the venv |
| 6.1 `ati.flyc.*` namespace + finalize dispatch | 1 | minimal form |
| 6.2 no perf axes | 1 | |
| 6.3 `flyc_request` body | 1 | the `wires_to`/grid half is Phase 2 |
| 6.4 rank-shortfall rule | 2 | |
| 6.5 `ati.context_helper` | 2 | |
| 6.6 `Fly.compile` rule generation | 1 | |
| 6.7 per-arch source map | 2 | |

### Phase 1 readiness — measured, not assumed

Everything below was run in this container against the current kernel
(FlyDSL `971dce48`), cross-compiling for gfx1201 with no GPU:

- **The build is reproducible.** `grep -n "environ\|getenv"` over `fmha_tuning_gfx1201.py`,
  `flash_attn_func_gfx1201_aiw.py`, `fmha_common_gfx1201.py` and `philox.py` returns nothing:
  `resolve_knobs` is a pure function of `(meta, knobs)`. (The interface docstring's warning
  about "tuning env vars" describes a version that no longer exists.)
- **The hsaco does not depend on the FakeTensor shapes.** Four shape/layout combinations —
  `(1,8,512,64)`, `(4,32,4096,64)`, `(1,1,16,64)` and a BSHD-strided `(2,16,1024,64)` — all
  produce a byte-identical 13496-byte object. The descriptor is purely a vehicle for reaching
  the `JitFunction`; it contributes nothing to the artifact.
- **The kernel symbol is stable.** Two builds in one process (hd 64 then hd 128) both export
  `flash_attn_func_aiw_kernel_0`. The `_0` suffix comes from a per-`CompilationContext`
  counter, not a global one, so it does not drift with build order (open question 3).
- **Dropout builds work torch-free**, passing `philox_seed=None` so `u64_scalar` short-circuits
  instead of allocating (19768 bytes vs 13496 for the non-dropout build).
- **Build cost is a non-issue.** 1.1 s at hd 64, 1.7 s at hd 128, 2.7 s at hd 256; flydsl
  imports in 0.1 s. The full gfx1201 matrix (12 tiles x PADDED_HEAD x CAUSAL_TYPE x BIAS_TYPE
  x ENABLE_DROPOUT x 2 dtypes = 384 functionals) is roughly 13 minutes single-threaded. No
  parallelism scheme needed, and `AOTRITON_GPU_BUILD_TIMEOUT`'s 8-minute default is ample.

### Phase 1 gaps

1. **Functional enumeration needs the operator.** 6.3 says flyc declares no axes and inherits
   them, so Phase 1 cannot be fully operator-free: `@ati.backend(2, flyc_attn_fwd, 'flyc')`
   must be added to `aot/flash_entry.py`. That is one line and no C++ — the alternative
   (a standalone axis list, deleted in Phase 2) is throwaway work that can drift from the
   operator it must eventually match.
2. **`Fly.compile` + its CMake loop are designed, not written** (6.6).
3. **The FakeTensor contract is not stable, and this is not hypothetical.** The last two
   kernel commits added an unconditional `Q.device` read in `_launch`, which broke the probe
   until the descriptor grew a `device = None`. Two commits, one break. This is open question
   2 arriving early, and it argues for asking FlyDSL for a supported AOT entry point rather
   than deferring it.
4. **Phase 1 cannot check correctness.** It ends at "an ELF exists, `EM_AMDGPU`, flags name
   the requested arch, expected symbol present". Numerical validation needs a gfx1201 device.

## Context

AOTriton needs to consume FlyDSL kernels as prebuilt HSA code objects, the way it already
consumes Triton kernels (`python/compile.py:134-140` writes `<name>.hsaco` + `<name>.json`).
The FlyDSL gfx1201 SDPA forward kernel is the first candidate: its host-side comments already
say "AOTriton dispatches the compiled hsaco directly rather than through the Python wrapper"
(`fmha_abi_gfx1201.py:243`, `fmha_common_gfx1201.py:137`), so the kernarg layout is already
treated as a frozen ABI on the FlyDSL side.

This box has no gfx1201 GPU (and GPU access is blocked), so the deliverable is specifically a
**cross-compile** path: source in, gfx1201 code object out, no device, no runtime.

**Torch-free is a hard constraint, not a preference.** `CMakeLists.txt:142` states outright:
*"VENV_DIR must never have torch."* The build venv deliberately excludes it so `pyaotriton`
stays ABI-compatible with the user's separate test environment. Anything that makes a FlyDSL
kernel build require torch is therefore unbuildable here by construction — which is why Part 5
is load-bearing rather than a cleanup. `numpy` is already in `requirements.txt`.

### What was verified while planning (not speculation)

Driving the real kernel through flydsl 0.3.1 in `/home/xinyazha/.venvs/nogpu` produced:

```
gpu.binary objects: 2 (byte-identical)
13240 bytes, magic \x7fELF, OS/ABI: AMDGPU - HSA, ABI Version: 4
Type: DYN, Machine: EM_AMDGPU, Flags: 0x4e, gfx1201
symbols: flash_attn_func_aiw_kernel_0 (FUNC, 7692 bytes)
         flash_attn_func_aiw_kernel_0.kd (OBJECT, 64 bytes)
```

Four findings shape the design:

1. **`ROCM_PATH` must point at a directory with `llvm/bin/ld.lld` at exactly that relative
   path.** MLIR's documented PATH fallback for `ld.lld` does **not** work in flydsl's bundled
   MLIR (24.0git); putting `ld.lld` on `PATH` still fails. For the venv the correct value is
   `<site-packages>/_rocm_sdk_core/lib` (which has `lib/llvm/bin/ld.lld`), *not* the more
   natural-looking `.../lib/llvm`. Getting this wrong yields the useless diagnostic
   `error: lld invocation failed` with no lld output. This is the single non-obvious step and
   the main thing the tool exists to encode.
2. **`format=fatbin` and `format=binary` are byte-identical for the ROCDL target** — both emit
   a bare hsaco ELF. flydsl's stock pipeline needs no patching; the blob it already produces
   *is* the hsaco.
3. **`COMPILE_ONLY=1` + `ARCH`/`FLYDSL_GPU_ARCH` is a real cross-compile mode**;
   `ensure_compile_runtime_pairing_from_env` explicitly avoids constructing a device runtime.
   But `flyc.compile()` cannot be used: it ends in `artifact._get_func_exe()`
   (`jit_function.py:1697`), which builds an ExecutionEngine and needs HIP. The `JitFunction`
   must be invoked directly; it returns `None` early under `COMPILE_ONLY`
   (`jit_function.py:1591`) and leaves the artifact in `_last_compiled`.
4. **The kernel's host layer is already torch-free by duck typing.** `_prep`/`strides_of`/
   `ptr_arg` need only `.shape`, `.dim()`, `.stride(i)`, `.data_ptr()`, and `ptr_arg`
   (`fmha_abi_gfx1201.py:220`) special-cases a class *named* `FakeTensor` to a null pointer.

## Part 1 — `modules/flash/flyc` is a bare collection of kernel sources

Modelled directly on `modules/flash/kernel/`: **no `__init__.py`, not a package**, flat sibling
imports between files (`import fmha_abi_gfx1201 as abi`, exactly as `fwd_kernel.py` does
`from dropout import PHILOX_RN_PER_OFFSET`). The compiler driver puts the directory on
`sys.path` and imports from it, mirroring `python/compile.py:60`
(`sys.path.insert(0, str(arg_path.parent))`).

Nothing but kernel sources lives here. The driver goes in `python/` (Part 3).

```
modules/flash/flyc/
├── PLAN.md / PLAN-PHASE1.md / SURVEY.md
├── README.md                        provenance + the ROCM_PATH gotcha
├── UPSTREAM.md                      FlyDSL sha + file list + the import-rewrite table
│
│   # the gfx1201 kernel and its host layer — all of it
├── flash_attn_func_gfx1201_aiw.py
├── fmha_abi_gfx1201.py
├── fmha_common_gfx1201.py
├── fmha_tuning_gfx1201.py
└── philox.py
```

Source: `/home/xinyazha/dockerhome/meff/FlyDSL/kernels/attention/parity/` at branch
`xinyazhang/sdpa-gfx1201-feature`.

**The shared helpers are imported, not vendored.** `buffer_ops`, `kernels_common`,
`layout_utils`, `mem_ops`, `utils` and `mma/wmma_ops` are FlyDSL's, not the flash kernel's,
so a copy here would be a fork of six files that have nothing to do with attention. They
come from the flydsl package:

```python
from flydsl.kernels.common import buffer_ops, kernels_common, layout_utils, mem_ops, utils
from flydsl.kernels.common.mma import wmma_ops
```

This needs an upstream change, and it is a blocker rather than a preference — verified
against the installed wheel: none of the six is under `site-packages/flydsl`,
`find_spec('kernels')` is `None`, and `setup.py:384,390` packages
`find_packages(where='python')` only, while `kernels/` sits at the FlyDSL repo root outside
`python/`. Recommended fix is to move or mirror `kernels/common/` under
`python/flydsl/kernels/common/` so `find_packages` picks it up; namespacing it under
`flydsl.` rather than shipping a bare top-level `kernels` package, which would claim a very
generic name in every site-packages that installs flydsl.

Until that lands, resolve them from the `third_party/flydsl` checkout via `sys.path` (which
is what the deleted `gfx1201_standalone.py` did) — still not a copy. See `PLAN-PHASE1.md`
Task 0.

**`gfx1201_standalone.py` is deleted, not vendored.** It existed only to put the FlyDSL repo
root on `sys.path` so `kernels.common.*` would resolve for files run in place. A flat directory
makes it dead weight, and its own docstring predicts exactly this: *"Once these files move under
the installed package the import becomes a plain `from kernels.common import mem_ops` and this
module goes away."*

Not vendored: `dropout_mask_gfx1201.py`, the three bwd kernels, `tooling/`, all tests,
`flash_attn_func_gfx1201_interface.py` (torch-only; its host-side contract is summarised in
README rather than shipping an unimportable file).

### The import rewrite

Because the flat layout matches how these files already imported each other inside FlyDSL's
`parity/` directory, **the vendored files are almost untouched** — six edits across three
files. Full table, reproduced in `UPSTREAM.md` so a re-sync is "copy fresh, reapply this":

| file | from | to |
|---|---|---|
| `flash_attn_func_gfx1201_aiw.py` | `from gfx1201_standalone import buffer_ops, wmma_ops` | `from flydsl.kernels.common import buffer_ops` / `...mma import wmma_ops` |
| | `from gfx1201_standalone import utils as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_common_gfx1201.py` | `from gfx1201_standalone import kernels_common` | `from flydsl.kernels.common import kernels_common` |
| | `from gfx1201_standalone import utils as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_abi_gfx1201.py` | `import torch` | **lazy**, into `u64_scalar` |
| | `from torch import float32 as torch_f32` | **lazy**, into `lse_args` |

Unchanged and verbatim: `philox.py`, `fmha_tuning_gfx1201.py`. Also verbatim within the
edited files: every flat sibling import already present (`import fmha_abi_gfx1201 as abi`,
`import fmha_common_gfx1201 as fmha`, `from fmha_tuning_gfx1201 import (...)`,
`from philox import Philox`).

The two torch imports must become function-local because the build venv must never have
torch. Both uses already sit inside functions: `torch_f32` in `lse_args` (a dtype check no
build reaches) and `torch.tensor`/`torch.cuda.stream` in `u64_scalar` (reached only when a
caller passes a plain int seed; the AOT driver passes `None`).

## Part 2 — ROCm toolkit discovery and the torch-free bootstrap

Lives with the driver in `python/`, not in `flyc/`. Both jobs must happen before `flydsl` is
imported.

**ROCm toolkit discovery.** Resolve a directory `D` with `D/llvm/bin/ld.lld`, in order:
existing `$ROCM_PATH` (validated, not trusted), `importlib.util.find_spec("_rocm_sdk_core")` →
`<parent>/lib`, then `/opt/rocm`. Export `$ROCM_PATH=D`. If none validates, raise listing the
candidates tried — anything beats `lld invocation failed`. Do **not** use
`rocm-sdk path --root`: it returns the unexpanded `_rocm_sdk_devel` tar tree. See `SURVEY.md`
for why `ld.lld` is the *only* ROCm component involved.

**Torch-free import.** `try: import flydsl.compiler`; on `ModuleNotFoundError(name="torch")`
install a minimal stub into `sys.modules` and retry. The guard means the stub self-disables the
moment FlyDSL ships Part 5. Complete required surface (everything `jit_argument.py` touches at
import time):

- `torch.{float16,bfloat16,float32,float64,bool,uint8,int8,int16,int32,int64}` — distinct
  hashable sentinels (`_TORCH_DTYPE_TO_MLIR_BUILDER`, `jit_argument.py:513`)
- `torch.{float8_e5m2,float8_e4m3fn,float8_e5m2fnuz,float8_e4m3fnuz}` — optional, `getattr`-probed
- `torch.Tensor` — a real class; registry key and `issubclass` scan (`jit_argument.py:84`)
- `torch.cuda.Stream` — a real class (`jit_argument.py:669`)

The stub sets `torch.__flydsl_aot_stub__ = True` and the driver prints a one-line notice when
active, so a stubbed build is never silently mistaken for a real one.

## Part 3 — `python/flyc_compile.py`, the FlyDSL analogue of `python/compile.py`

`python/` installs into the venv as the `aotriton` package (`CMakeLists.txt:234-240`), so this
is invoked the same way the Triton compiler is:

```
python -m aotriton.flyc_compile modules/flash/flyc/flash_attn_func_gfx1201_aiw.py \
       --target gfx1201 --head_dim 64 --dtype f16 --num_heads 8 --causal 0 \
       --out_path build/flyc/attn_fwd_aiw_hd64_f16
```

Positional `path` and `--out_path` keep `compile.py`'s CLI shape; argparse names use
underscores per CLAUDE.md. Pipeline:

1. Toolkit discovery + torch-free bootstrap (Part 2). Set `ARCH`, `FLYDSL_GPU_ARCH`,
   `COMPILE_ONLY=1`, `FLYDSL_RUNTIME_ENABLE_CACHE=0`. (The env var names are literally `ARCH`
   and `COMPILE_ONLY`, unprefixed — `env_var=` is given explicitly in
   `flydsl/utils/env.py:233,238`.)
2. `sys.path.insert(0, str(Path(args.path).parent))`, then import the builder from the named
   source — same mechanism as `compile.py:60`.
3. `build_flash_attn_func_aiw_module(**meta_and_knobs)` → the `_launch` wrapper.
4. Capture the traced launcher: temporarily replace `fmha_abi_gfx1201.run_compiled` with a
   recorder, call `_launch(q, k, v, o, batch, seqlen)` with duck-typed `FakeTensor` BHSD
   descriptors. This routes through the **real** host ABI-marshalling code (`_prep`,
   `abi.varlen_args`, `abi.dropout_args`, `_resolve_scale`, …), so the 41-argument kernarg order
   the hsaco is built for is the one the tested launcher produces, not a second transcription.
   (See open question 2.)
5. Invoke the captured `JitFunction` directly (not `flyc.compile`); read
   `jf._last_compiled[1]._ir_text`.
6. Re-parse that text, walk for `gpu.binary`, extract each object via
   `flydsl._mlir.dialects.gpu.ObjectAttr(op.objects[i]).object`. Two objects appear
   (`#rocdl.target<chip="gfx1201">` plus the `no_wave64` one added by `rocdl-attach-target`);
   they are byte-identical, so assert that and emit one. Fail loudly if they ever differ.
7. Write `<out_path>.hsaco` + `<out_path>.json`, mirroring `python/compile.py:134-140`. JSON
   carries `compile_status` (the only key `python/codegen/autotune.py:58` actually reads), the
   kernel symbol read back from the ELF (see open question 3), `arch`, `warp_size`, `shared`
   (LDS bytes), block size, and the resolved `FmhaInputMetadata`/`FmhaKnobs` so a build is
   reproducible from its sidecar.
8. `--verify` (default on): `llvm-readelf -h` from the discovered toolkit; assert `EM_AMDGPU`
   and that the flags name the requested arch. The only "test" that runs — no kernel is launched.

Like `compile.py`, a failed build still writes an empty `.hsaco` and a `.json` recording the
failure status, so the driving build does not stall on a missing file.

## Part 4 — FlyDSL as a pinned build dependency

A **pinned wheel**, not a submodule. FlyDSL's `setup.py` requires a prebuilt bundled MLIR at
`build-fly/python_packages/flydsl/_mlir` (LLVM 24.0git), so `pip install third_party/flydsl`
is not self-contained the way `pip install third_party/triton` is — a submodule would buy a
source tree nothing can build in-tree.

- `third_party/flydsl.txt`, requirements.txt syntax, `flydsl==0.3.1`. Location and naming
  follow `third_party/aiter.txt`, the existing one-file dependency pin.
- Installed beside the triton wheel in `CMakeLists.txt` :242-327, with an
  `aotriton_venv_flydsl` target mirroring `aotriton_venv_triton` (:325-327). Sentinel is
  `${VENV_SITE}/flydsl/__init__.py` — not the bundled-MLIR extension, whose filename carries
  an LLVM version. Guarded on `AOTRITON_NOIMAGE_MODE` like the triton install.
- **No new inherit-system-site option.** `AOTRITON_INHERIT_SYSTEM_SITE_TRITON` already decides
  whether the venv is created `--system-site-packages` (:209), which is what exposes a
  preinstalled flydsl. One switch, not two.
- `AOTRITON_FLYDSL_ROCM_PATH` (default: auto-detect per Part 2) so a build can point at a
  shared or pre-staged linker tree instead of installing a 1.4 GB `rocm-sdk-core` wheel.
- `numpy` is already in `requirements.txt`; nothing to add.

**Per-venv flydsl needs a format change, specified but not implemented in Phase 1.**
`docs/AltWheelExample.yaml` maps a venv name to a single triton wheel and
`root.py:_load_altwheel_config` reads exactly that (`self._altwheels[name] = Path(value)`),
so every venv currently gets the same `third_party/flydsl.txt` pin. The agreed extension
keeps **existing yaml files working unedited**: the loader branches on the node type, so a
scalar value means what it means today (one wheel, or the `python:` interpreter form) and a
**sequence** is new — several pip requirement lines, installed in order. The scalar stays
**wheels-only**; `flydsl==0.3.2` is valid solely in the sequence form, and the loader must
enforce that rather than document it, since CMake's verbatim `pip install ${WHEEL}` would
otherwise accept a scalar requirement by accident and let the two forms drift. A scalar and
a one-element sequence are different YAML; the guarantee is that the scalar form is still
accepted, not that the two spellings coincide. The sequence form admits `flydsl==0.3.2` and
PEP 508 direct references like
`flydsl @ git+https://github.com/ROCm/FlyDSL.git@<sha>` per venv. Full spec, the requirement
forms, and the two parsers that must agree are in `PLAN-PHASE1.md` 2.5c. It matters sooner
than it looks: 6.7 wants a per-arch flyc kernel source, and different arches may want
different flydsl versions.

The dependency survey (`SURVEY.md`) constrains this part more tightly than the Triton analogy
suggested:

- The build venv needs **only** `flydsl` + `numpy`. No torch, no HIP, no `rocm_sdk` package —
  flydsl contains zero references to `rocm_sdk`, and its compiler library has no HIP/HSA/ROCm
  link dependency at all.
- ROCm contributes **exactly one file**: `<ROCM_PATH>/llvm/bin/ld.lld`. flydsl bundles the LLVM
  AMDGPU backend and reaches complete gfx1201 ISA with `ROCM_PATH` unset; it just has no linker.
  A self-contained 143 MB subtree (`libLLVM.so` + `lld` + wrappers + 2 sysdeps libs) is enough,
  against 1.4 GB for the whole wheel.
- Because the linker is reached by **path, not import**, alt-venvs do not each need a copy.
  Every alt-venv carries flydsl + numpy (~300 MB) and shares one `$ROCM_PATH`.
- Hermeticity is measured, not assumed: full venv, minimal tree, and cross-venv all produce a
  byte-identical hsaco (sha256 `d537bb2ee33b0c121c865a9e…`).

## Part 5 — Eliminating torch from FlyDSL (upstream)

Written up in `UPSTREAM.md`; implemented in the FlyDSL repo, not here. Because
`CMakeLists.txt:142` forbids torch in the build venv, this is on the critical path — the
bootstrap stub is a bridge, not a solution. `flydsl/autotune.py:18` already does a
function-local `import torch`; this applies that pattern to the one place that does not.

**The problem.** `python/flydsl/compiler/jit_argument.py:12` imports torch at module scope and
`flydsl/compiler/__init__.py:5` imports that module, so `import flydsl.expr` transitively
requires torch. Every use is launch-side argument marshalling: the dtype→MLIR map (`:513`),
`TorchTensorJitArg` (`:544`), `from_torch_tensor` (`:647`), `register(torch.cuda.Stream)`
(`:669`). None participates in tracing, lowering, or code-object emission — `from_c_void_p` and
the DLPack-generic `from_dlpack` already cover framework-neutral callers.

**The change.**
- Move the torch block into `flydsl/compiler/frameworks/torch_adapter.py`.
- Give `JitArgumentRegistry.get()` a lazy hook: on a miss, if `py_type.__module__` roots at a
  known framework whose adapter is not loaded, import it and retry once. The dict-hit fast path
  is unchanged after first use.
- Keep `flyc.from_torch_tensor` working via a PEP 562 `__getattr__` in
  `flydsl/compiler/__init__.py` that imports the adapter on access and raises a clear
  "torch is not installed" otherwise.
- Declare `numpy` as a dependency and `torch` as an extra (`flydsl[torch]`). flydsl 0.3.1's
  METADATA has **no `Requires-Dist` at all**, which is why the venv silently lacked numpy;
  numpy *is* a genuine compiler dependency (`expr/numeric.py`, fp16/bf16 bit-punning).

**Three secondary FlyDSL issues found, worth filing alongside:**
- `flyc.compile()` ignores `COMPILE_ONLY` at its tail and calls `_get_func_exe()`
  (`jit_function.py:1697`), so the documented GPU-less compile mode is unusable through the
  public API. It should early-return as `JitFunction.__call__` does.
- `kernel_function.create_gpu_module(targets=backend.gpu_module_targets())` and the
  `rocdl-attach-target` pass both attach a `#rocdl.target`, so every module is serialized to a
  code object twice, byte-identically. Pure waste of link time.
- A missing/incorrect `ROCM_PATH` surfaces only as `lld invocation failed` with no lld output.
  A pre-flight check for `<toolkit>/llvm/bin/ld.lld` with a real message would have saved this
  entire investigation.

## Part 6 — ATI language features for the FlyDSL backend

Demo description: `modules/flash/aot/flyc_attn_fwd.py` (written, not wired in). Read it
alongside this section — it is the concrete form of everything below.

### 6.1 A third backend kind

ATI has two backend shapes today and flyc is neither:

| | compiled in-build | owns perf space | 1:1 arg names | precedent |
|---|---|---|---|---|
| triton | yes | ATI (`@ati.tune.*`) | yes | `aot/attn_fwd.py` |
| aiter | no (prebuilt `.co`) | none | no (C++ cookie) | `aot/aiter_fwd.py` |
| **flyc** | **yes** | **the kernel** | **no** | — |

So it borrows the compile pipeline from triton and the "not 1:1" problem from aiter, and
needs a new `ati.flyc.*` namespace finalized like the others. The dispatch is a one-line
addition to `specs/finalize.py:255` (`elif isinstance(marker, FlycKernelSpec)`), mirroring
`AffineKernelSpec`. It attaches to the operator with the existing
`@ati.backend(2, flyc_attn_fwd, 'flyc')` in `aot/flash_entry.py`.

### 6.2 Programmatic tuning moves the perf space, it does not remove it

`fmha_tuning_gfx1201.resolve_knobs()` is the only producer of a schedule, so the flyc backend
declares no `@ati.tune.schema` and no `@ati.tune.configs`: ATI does not enumerate perf
variants for it, and there are no perf axes in the ATI sense.

**That is not the same as one hsaco per functional, and the plan must not assume it is.**
Today the count happens to be one, for a reason that is a snapshot of the kernel rather than
a property of the design: the shipped schedule is tuned for long sequences, and short
sequences are currently served by the Triton backend instead. When FlyDSL grows a
seqlen-dependent schedule (6.9.2), flyc will emit several hsacos per functional and need a
real selection key — the same thing the Triton path already does.

So build the structure N-capable from the start and let N be 1 today:

- pack every hsaco into a per-functional `.aks2`, even the single one (Task 7). The archive
  is the framework's unit; skipping it because N==1 would have to be undone.
- keep the `.aks2` entry name able to distinguish variants — do not collapse it to a constant
  on the grounds that there is only one (7a).
- do not hardcode `total_hsacos == 1` anywhere; read it from the archive.

The payoff is concrete: keeping the layout identical to Triton's means **Triton's autotune
code generator is reusable for flyc in the later phases** rather than needing a parallel one.
The only thing flyc genuinely lacks is ATI-side perf *enumeration* — the storage, selection
and tuning machinery downstream of it all still applies.

Use **`resolve_knobs`, not `plan`**. `plan()` is the JIT entry: it takes a *real* head_dim
and rounds it up the ladder, deriving `padded_head` as a side effect. AOT already knows the
tile (it *is* `BLOCK_DMODEL`) and `PADDED_HEAD` is its own ATI axis, so calling `plan()`
would silently re-derive an axis the operator already fixed. The builder's keyword front
end documents the same distinction.

### 6.3 Wiring operator params to the builder

Two separate bridges, and conflating them is the trap:

**Functional → builder request** is `flyc_request(f)`, a plain Python function registered
with `@ati.flyc.request(...)`. The generator never calls it; `aotriton.flyc_compile` does,
at build time, in the venv that has flydsl. Almost all of it is 1:1 — FlyDSL's `causal_type`
*is* AOTriton's `CAUSAL_TYPE` and the kernel emits only `{0, 3}`, the same pair
`@ati.scalar('CAUSAL_TYPE', options=[0, 3])` declares. One trap: `FmhaInputMetadata.num_heads`
reaches the emitted kernel only via `STRIDE_TOKEN`, read exclusively under
`STRIDES_CONSTEXPR` (a dense-only diagnostic arm AOT never selects). Pin it to 1 so it
does **not** become a functional axis — but that is a fact about the current kernel, so
assert it rather than assume it.

**Kernel arg → operator operand** is `wires_to=`, which already exists and is fully
implemented (`ir/kdesc.py:65`, `apparel_of`/`real_of`, launch-arg emission at
`kdesc.py:400`). The full mapping is in the demo; it is a rename for 20 of the 40.

Declare against **`flash_attn_func_aiw_kernel`, not `launch_flash_attn_aiw`**. The launcher
is host code the C++ shim replaces; the `@flyc.kernel` def is what fixes the kernarg layout,
and the two differ — the launcher carries a `stream` the kernel has not got, and it passes
its `batch_size` into the kernel's `num_seqlens` slot, which is the tell that that slot is a
count rather than AOTriton's mode-and-count overload (6.5).

**Declare no functional axes.** The operator owns them (via the default triton backend);
flyc inherits and narrows with `@ati.disable`, exactly as `aiter_fwd.py` does. Re-declaring
them as `@ati.scalar` is not merely redundant, it is wrong: `BLOCK_DMODEL`/`CAUSAL_TYPE`/…
are not arguments of the flyc kernel at all — they are build-time Python values consumed by
the builder — so a spec claiming them would claim arguments that do not exist. Arch belongs
in the same predicate rather than a separate `@ati.flyc.arch([...])`; `f.arch` is available
to predicates, and one exclusion list beats two places to look.

**Put the builder call in the def body, not in a `builder=` argument.** FlyDSL builders are
not required to share an API — the gfx942/gfx950 modules need not take `(meta, knobs)` — so
naming a symbol and handing it a fixed argument shape only works until the second kernel.
The body is the general form. The cost is that the description module now contains code the
generator must never execute; that is already true of every `@ati.disable` predicate.

**Do not declare `entry=`.** The kernarg order is declared by the `@ati.tensor`/`@ati.scalar`
block; naming the kernel def as well gives one fact two sources. The kernel *symbol* is still
needed at dispatch, but the driver reads it back out of the ELF, which cannot go stale. A
future `verify_abi=` that AST-checks declarations against the kernel def would be worth
having — as a check, not as the source.

**The launch grid goes in `grid_calculator()`** — the same context member 6.5 builds on.
`FlycAttnFwdContext::grid_calculator()` in `modules/flash/csrc/flyc_attn_fwd.cc` returns
`(num_head_q, cdiv(max_seqlen_q, BLOCK_M), batch)` with block `(BLOCK_SIZE, 1, 1)`.

One unresolved piece: `BLOCK_M` and `BLOCK_SIZE` are FlyDSL **knobs**, and 6.2 removed the
perf axes, so they are not perf fields on the context the way `this->BLOCK_M` is for the
triton backend. They have to reach the context some other way — folding the JSON sidecar
(Part 3 step 7) into the compiled-in metadata is the obvious candidate, but it is not
designed.

### 6.4 The last dimension — `rank` shortfall, not `contiguous=`

The concept ATI needs already exists (`Axis.is_launch_data`, `ir/axis.py:86`: *"unit
(contiguous) strides are constexpr 1 (not passed)"*), but **`contiguous=-1` is the wrong
spelling for flyc and would silently corrupt the ABI.** `resolve_contiguous` indexes the
*matched stride list*; the triton kernel matches four names (`stride_qz/qh/qm/qk`) so `-1`
picks `stride_qk`, but the flyc kernel matches only three, so `-1` would mark
`stride_q_seq` as the unit stride.

The difference is that in triton the unit stride *is a parameter*; in flyc there is no
argument to point at. So declare the rank and let the shortfall carry the meaning:

```python
@ati.tensor('Q', 'T_io', rank=4, strides='stride_q_*', wires_to='Q')   # 3 strides, rank 4
```

Rule: the trailing `rank - len(matched_strides)` dimensions are implicitly unit-stride,
constexpr 1, not passed. No new keyword, `rank=` becomes self-justifying, and it fixes the
rank inference too (`resolve_rank` would otherwise infer rank 3 from the stride count).
Validation to add: the shortfall must be `>= 0`, and `contiguous=` and a rank shortfall on
the same tensor is an error (two ways to say one thing).

The operator's params struct still carries `stride_qk`/`stride_kk`/`stride_vn`/`stride_on`
and simply never forwards them. Add a **debug-build assertion** in the shim that the unit
stride really is 1: FlyDSL's `strides_of` raises on the host today, and dispatching a bare
hsaco drops that check.

### 6.5 The arguments that are not a rename

The real cost of the intentionally-not-1:1 interface, and the only place the demo needs
something ATI lacks:

| flyc argument | why it is not a rename | status |
|---|---|---|
| `varlen_bits` | packed layout descriptor with no AOTriton operand; also absorbs the mode half of `Num_seqlens` | permanent |
| `idropout_p` | fixed-point i32, not `dropout_p`'s fp32 | permanent |
| `dropout_scale` | `1/(1-p)`, precomputed host-side, no operand at all | permanent |
| `batch_size` | must be `q.size(0)`, which is not AOTriton's `Batch` under packed varlen | permanent |
| `num_seqlens` | AOTriton's signed three-way encoding vs FlyDSL's unsigned count | permanent |

The PRNG arguments were on this list and are not any more: FlyDSL `971dce48` ("the philox
seed becomes a pointer, and the forward reports what it drew") and `53334317` ("the philox
offset splits into a pointer and an immediate") make `philox_seed_ptr`, `philox_offset1`,
`philox_offset2`, `philox_seed_output` and `philox_offset_output` 1:1 with the triton kernel.
Plain renames now, and the demo declares them with the same grouped `@ati.tensor([...],
'T_u64', rank=0)` the triton description uses.

**`num_seqlens` is the instructive one**, because the names match exactly and the semantics
do not. AOTriton overloads `Num_seqlens` as mode *and* count, three-way — `> 0` varlen
compact/stacked with the value as the sequence count, `== 0` dense, `< 0` varlen with BHSD
layout padded to `Max_seqlen_q` (`kernel/fwd_kernel.py:247,267` branch on precisely this).
FlyDSL's slot is a plain count: every mode bit lives in `varlen_bits`, and the launcher just
passes `batch_size` in.

That is the superset relationship — FlyDSL supports more varlen layouts *because* it factored
mode out of count. So no rename is right; the helper is where the three-way encoding is
decoded once, and `flyc_varlen_bits` consumes the same sign to build its mode bits.

**UPDATE — the `abs()` design is dead; the pair now needs two helpers.**

That design existed because the launcher passed its `batch_size` into the kernel's
`num_seqlens` slot, so one helper had to undo the conflation. FlyDSL `f79182b7` / `1d231767`
split them into separate kernel arguments, which removed that reason and left a different one.

FlyDSL's contract (`fmha_abi_gfx1201.varlen_args` docstring): `batch_size` is `q.size(0)`
*always, whatever the layout*; `num_seqlens` is how many sequences are packed into a 1HTD
tensor, and 0 when nothing is packed. Dense is `(B, 0)`; packed with N sequences is `(1, N)`.
The kernel branches on the pair —
`nseq_idx = (num_seqlens != 0).select(num_seqlens, batch_size)`.

AOTriton spells the same information as a `Batch` operand plus a **signed** three-way
`Num_seqlens` (`>0` packed count, `0` dense, `<0` BHSD-padded varlen). Hence:

| flyc operand | source | why not a rename |
|---|---|---|
| `batch_size` | `params.Q->size(0)` | under packed varlen Q is 1HTD, so `q.size(0)` is 1 while `Batch` is not |
| `num_seqlens` | `max(Num_seqlens, 0)` | `wires_to='Num_seqlens'` would hand a negative value to a `select` that reads it as a count |

The `<0` case is *padded*, not packed, so FlyDSL wants 0 with the layout carried in
`varlen_bits` — **verify that against `flyc_varlen_bits` before implementing**, since the two
helpers must agree on how the padded case is encoded.

Both failures are **silent**, which is the argument for asserting as well as wiring. FlyDSL's
own docstring on the mistake: *"it launches N programs over a tensor whose batch axis is 1,
and every one of them addresses a plausible row."* No crash, no wrong-arch ELF, just wrong
numbers.

Still worth raising upstream, unchanged by the split: `decode_addressing`
(`fmha_common_gfx1201.py:1088`) takes a `num_seqlens` parameter it never references.

`ir/kdesc.py:71` already anticipates this: *"The apparel value is a plain operand name for
now; the representation is kept opaque so it can later carry a tuple of operator params or
an expression."* The representation to activate is a **context-class member function**:

```python
wires_to=ati.context_helper('flyc_num_seqlens')
  -> declares  int32_t flyc_num_seqlens() const;  on FlycAttnFwdContext
  -> author implements it in modules/flash/csrc/flyc_attn_fwd.cc
```

**This is not a new concept — it is the existing `grid_calculator` split.**
`codegen/template/shim.h:73` declares `dim3 grid_calculator() const;` inside the generated
context struct, and `modules/flash/csrc/attn_fwd.cc:14` implements it by hand as
`AttnFwdContext::grid_calculator()`. Same class, same namespace
(`AOTRITON_NS::v3::flash` — v3 before flash, `include/aotriton/flash.h:28`, and the root is
the `AOTRITON_NS` macro since release builds version it). `context_helper` only lets a
description declare *more* of these instead of the set being hardwired in the template.

**No arguments**, because the context already carries everything: `params` (the whole
operator params struct) and the selected perf fields are members
(`codegen/template/shim.h:35-50`). `grid_calculator` shows the range — it reads
`params->Num_seqlens`, `params->Q->size(1)`, `params->Batch`, `this->BLOCK_M` and
`this->PERSISTENT_TYPE`, and takes nothing.

The return type is not declared twice: it comes from the `@ati.scalar` type on the same
line, so `'i32'` fixes the signature as `int32_t`.

Rejected alternative — `wires_to=ati.expr('<C++ expression>')`, a C++ string in the Python
description:

- a typo surfaces only when the generated `.cc` compiles, with the error pointing at
  generated code; a missing context helper is a **link error naming the symbol**
- a string cannot be unit-tested, stepped through in a debugger, or carry a comment
  explaining the three-way `Num_seqlens` encoding at the point it is decoded
- anything past a single expression (`abs()`, the varlen bit packing) does not fit
- it would mean two mechanisms for host-side code — `grid_calculator` in `csrc/`,
  expressions in the description — and therefore two places to look

The cost is that trivial cases become functions too (`dropout_scale` is one divide).
Uniformity wins: one mechanism, one file, one place a reviewer looks.

Together with 6.4 this is the whole new-language-feature budget: `ati.context_helper`, and
the rank-shortfall rule. Everything else reuses what is there.

### 6.6 Rule generation

The triton path writes `Bare.compile` — one `;`-separated line per hsaco
(`ALTNAME;COMPILER_PYTHON;HSACO;SRC;KNAME;NWARPS;NSTAGES;WAVESPEREU;TGTGPU;SIG`) — which
`v3src/CMakeLists.txt:180-220` turns into one `add_custom_command` per line invoking
`python/compile.py`. The flyc analogue is a second file, `Fly.compile`, with its own tuple
(`VENVPYTHON;HSACO;SRC;BUILDER;ENTRY;TGTGPU;REQUEST_JSON`) and a parallel CMake loop
invoking `aotriton.flyc_compile`. No perf columns — 6.2 removed them.

**The per-functional venv selector already exists.** `root.py:344 _get_venv_and_python(f)`
matches functional attributes against the alt-wheel rules and returns a python executable.
That is exactly the hook for routing flyc functionals to a flydsl venv, and it composes
with `SURVEY.md`'s finding that the linker is shared by path rather than duplicated per
venv. Reuse it rather than inventing a parallel mechanism.

### 6.7 Arch in template instantiation

Mostly already there. `Functional` carries `arch` and `arch_number`
(`ir/functional.py:93`), and predicates already receive it — `attn_fwd.py` does
`@ati.derives('NUM_XCDS', to=8, when=lambda f: f.arch in ('gfx942', 'gfx950'))`. So
`flyc_request(f)` can read `f.arch` today with no change.

What is missing for a future unified `flash_attn_func_interface` is only that
`@ati.flyc.kernel(...)` take a **per-arch source**, so one description covers
gfx1201/gfx942/gfx950/gfx1250:

```python
@ati.flyc.kernel({'gfx1201': '../flyc/flash_attn_func_gfx1201_aiw.py',
                  'gfx950':  '../flyc/flash_attn_func_gfx950_aiw.py'},
                 builder='build_flash_attn_func_aiw_module_primary',
                 entry='launch_flash_attn_aiw')
```

with `@ati.flyc.arch([...])` derived from the mapping's keys. Note this implies the ABI may
differ per arch, so the argument declarations would have to become arch-conditional too —
worth deciding before the second arch lands, not after.

### 6.8 `ati.flyc.kernel` takes a module, and parses nothing

Unlike `@ati.source`, it has nothing to AST-parse: the ABI is declared (6.3) and the builder
is called from the body. Its one argument is the module path, which the driver puts on
`sys.path` before executing the body — the same bare-directory contract as
`modules/flash/kernel/` (Part 1).

Worth recording because it was nearly designed the other way: both `build_..._primary` and
`launch_flash_attn_aiw` would have been awkward to introspect anyway, since the launcher is
**nested inside** the builder and `decorators/source.py:_ast_kernel_param_names` scans
top-level defs only. If `verify_abi=` is added later it will need a nested-aware walk.

### 6.9 Survey — what object does the build function receive?

`flyc_attn_fwd(f)` works today and does not generalise: FlyDSL may come to tune on seqlen,
which is not and must not become a functional axis. Four candidates were considered.

**`Functional`** (`ir/functional.py:93`) — carries `arch`, `arch_number`, `godel_number` and
the `choice` map, with `.choices` (by choice-variable name) and `.compact_choices`
(multi-choice axes only, what `_common.check_value` reads). It is *the compile-time
identity*: the enumeration unit and the godel key.
Right for everything that is an axis. It cannot carry seqlen, and should not — seqlen is a
**binning** dimension (`@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le)`), and promoting
it to an axis would multiply the functional space and the godel numbering *for every
backend* in order to serve one.

**`FlashEntry`** (`v3python/tune/flash/module.py:26`, codegen-side copy at
`modules/flash/aot/flash_entry.py:22`) — `dtype/hdim/seqlen_q/seqlen_k/causal/dropout_p/
bias_type`. The tuning DB key, and it does carry seqlen. Rejected on three counts: it is
flash-family specific, and a generic `flyc_compile.py` must not know it; its vocabulary is
the *tuner's* (`hdim`, `causal` as `bool | tuple[int,int]`), not the kernel's; and it is
explicitly unmodularized — the codegen-side copy exists only to sever the `v3python.tune`
edge and its docstring says *"TODO: Merge with modules/flash/tune in ATI Phase 2"*.

**`FlashInputMetadata`** (`module.py:64`) — `FlashEntry` plus `N_HEADS`, `BATCH`, `sm_scale`,
`storage_flip`, `prng_seed`. Rejected more firmly: those fields describe a *benchmark input*
— what to allocate and how to seed it. `prng_seed` and `storage_flip` cannot affect an hsaco.

**`FmhaInputMetadata`** (FlyDSL) — the *output* of the mapping, not its input. Worth noting
it has **no seqlen field**, so FlyDSL's tuner is seqlen-independent today and "one hsaco per
functional" currently holds. The concern is real but not yet active.

**Conclusion: two objects, `(f, hints)`,** because these are two kinds of fact —

| | what it says | enumerable? |
|---|---|---|
| `f` | what the kernel must **support** | yes: axes, godel-keyed, arch-aware |
| `hints` | what the kernel should be **optimized for** | no |

with the second object a description-declared dataclass. Its shape deliberately mirrors
`@ati.tune.schema(PerfCls)` — a dataclass whose fields all have defaults — with the direction
reversed: `schema` declares perf parameters ATI chooses and bakes into a Triton signature;
this declares tuning inputs the description's *own* builder consumes. The description owns
the vocabulary, so neither the tuner's nor the family's spelling leaks in.

### 6.9.1 Naming — surveyed against FlyDSL, then namespaced

`ati.tune.hints` was a placeholder and a bad one: `ati.hints.*` already exists
(`decorators/hints.py`, `union_precedence`) and means something unrelated. Surveying
FlyDSL's own vocabulary first, since the object is fed to a FlyDSL tuner.

**FlyDSL's generic autotuner** (`python/flydsl/autotune.py`), Triton lineage:

| term | meaning |
|---|---|
| `Config` | "a single tuning configuration" — the chosen knobs |
| `key: List[str]` | argument names whose runtime values select a cached config; the docstring glosses it as *"the portable call axes"* |
| `default` | a heuristic returning a `Config` without benchmarking |
| `prune_configs_by`, `artifact_name` | search control, cache identity |

**FlyDSL's gfx1201 FMHA tuner** (`kernels/attention/parity/fmha_tuning_gfx1201.py:407-416`),
purpose-built, and it states the split outright:

> *"The two halves of a build request. Split on who decides ... a caller states a **problem**,
> the tuning policy answers with a **schedule**."*

| term | meaning |
|---|---|
| `FmhaInputMetadata` | *"What to compute. Set by the caller; never by policy."* — the **problem** |
| `FmhaKnobs` | *"How to compute it."* — the **schedule** |
| `FmhaPlan(meta, knobs)` | both halves; `plan(request: FmhaInputMetadata, overrides)` |

Lining the three vocabularies up:

| concept | flydsl autotune | gfx1201 tuner | ATI today |
|---|---|---|---|
| chosen knobs | `Config` | `FmhaKnobs` / schedule | `@ati.tune.schema` perf struct |
| what to compute | the call args | `FmhaInputMetadata` / problem | `Functional` (axes only) — **gap** |
| runtime values that select | `key` | (none yet; seqlen would join `FmhaInputMetadata`) | `@ati.tune.binning` |

**Decision: `@ati.flyc.hints(Cls)`, passed as `hints`.** Namespace, do not rename.

The survey's answer ("FlyDSL calls it the *problem*") is real but it answers the wrong
question. The collision was never that `hints` is a poor word — it is that
`ati.tune.hints` sits one dot away from an unrelated `ati.hints`, and that `ati.tune.*`
is the **shared** tuning vocabulary. Every member of it — `schema`, `configs`, `binning`,
`fallback` — feeds the LUT and the tuning DB, and is available to every backend. This
object feeds neither: it goes to one description's own builder and nowhere else. Putting
it in `ati.tune.*` would advertise it to backends that have no use for it, Triton included
(which covers the same need with `@ati.tune.binning` plus the LUT).

`ati.affine.*` is the settled precedent: aiter's backend-specific surface lives under its
own namespace (`affine.arch`, `affine.limitations`, `affine.structures`,
`affine.directories`, `affine.aiter_asm`, `affine.supplies`) rather than being spread
through the shared decorators. `ati.flyc.*` should be the same — one place to look for
"what is different about the FlyDSL backend" — and it already holds `flyc.kernel`.
The prefix dissolves the collision without needing a new word at all.

Boundary, so the submodule does not become a junk drawer: `ati.flyc.*` is for things that
exist **because the backend is FlyDSL**. `ati.context_helper` (6.5) stays top-level — it
declares a member on the generated C++ context class, which is ATI infrastructure with a
Triton precedent (`AttnFwdContext::grid_calculator`), and any backend needing host-side
translation would want it.

Two notes carried over from the survey, still worth having in the docstring:

- FlyDSL would call these fields part of the **problem**, opposite its **schedule** — worth
  saying, since a reader coming from `fmha_tuning_gfx1201.py` will look for that word.
- It is strictly the **runtime half**: dtype / head_dim / causal are also part of FlyDSL's
  problem, but in ATI they are functional axes and arrive on `f`. `f` and `hints` together
  are what `FmhaInputMetadata` keeps in one dataclass.

### 6.9.2 The remaining open item

**A domain is needed, not just a schema — and that is the hard part.** A schema says the
   fields exist; enumerating builds needs their *values*. Three options: declare a domain in
   the description (`seqlen_q=[512, 4096]`), at which point hints are an axis in all but
   name and the build count multiplies; take bin edges from the tuning DB, which matches
   `@ati.tune.binning` semantics but makes the build depend on DB contents and has no Phase 1
   path; or **defaults only**, one build per functional. Phase 1 does the third.
   When a domain appears, flyc emits several hsacos per functional and needs a real
   selection key — exactly the machinery AOTriton already runs for Triton. 6.2 says to build
   for that now rather than assume today's count of one, precisely so that day is a
   configuration change and not a re-architecture.

### 6.10 `ir/` is reorganised by language, sharing via a library not a base class

`KernelDescription` + `KernelSignature` are how Triton describes a kernel and one compiled
instance of it. flyc needs the same pair — the earlier plan hand-rolled flyc's hsaco naming
and entry names instead, which quietly broke 6.2's promise that Triton's autotune code
generator stays reusable, since that generator is `KernelSignature`-driven throughout
(`codegen/autotune.py`: `_sigs`, `codegen_compact_kernels`, `codegen_kernel_psels/copts`,
`all_signatures`).

**Layout.** Each language owns a directory with the same filenames:

```
ir/
  interface.py  axis.py  cfield.py  functional.py       # shared, unchanged
  override.py   typed_choice.py  operator.py  ops/      # shared, unchanged
  metro.py                                              # shared, unchanged
  lib/                       NEW — helper functions the language modules CALL
  triton/kdesc.py            <- ir/kdesc.py
  triton/ksignature.py       <- ir/ksignature.py
  affine/kdesc.py            <- ir/affine.py
  flyc/kdesc.py              NEW
  flyc/ksignature.py         NEW
```

`affine/` gets no `ksignature.py`: an affine kernel has no functional space
(`gen_functionals` yields nothing), no perf, and its per-image unit is `co_gen()` over
prebuilt `.co` files. Not every language needs every file; the point is that when a language
does need one, it is at the same path.

`metro.py` stays shared and **unchanged**. Its `for kdesc in self._kernels` loop
(`:116-117`) is a generic local name, not a `KernelDescription` dependency — it only calls
`iter_kernel_slot_names()`.

**Sharing is a library, not a hierarchy.** Following the kernel maxim — *abstractions are not
your friend; libraries are* — common code goes in `ir/lib/` as **functions the per-language
modules call**, not as a base class they inherit. So there is no `KernelSignatureBase` with
flyc overriding the Triton-specific bits.

Seed `ir/lib/` with only what must *agree* across languages today:

- the entry-name grammar, as a pure formatter over already-rendered parts:
  `entry_name(unified_signature, arch, perf='', copt='') -> ";;#F;…;;#P;…;;#CO;…;;arch=…"`
- `blake2b_hash(package_path, entry)`

**psels and copts are generic concepts, not Triton's** — flyc kernels have knobs too, and
knobs are close kin to psels. What is Triton's is the *particular* vocabulary:
`COMPILER_OPTIONS` = `num_warps` / `num_stages` / `waves_per_eu`, `DEFAULT_COPT`, the
`COPT_*` indices, the gfx1250 warp-doubling workaround, and `triton_signature_string`. Those
stay in `triton/ksignature.py`.

The generic `perf_section` / `copt_section` renderers also stay there **for now**, not
because they belong to Triton but because nothing else calls them yet. Promote them to
`ir/lib/` when a second caller appears; promoting them before that would be guessing at a
shape flyc has not chosen.

**flyc leaves both sections empty, deliberately.** Not "flyc has no perf" as a permanent
truth — the flyc tuning model is simply unsettled, and the programmatic build makes it likely
to look nothing like Triton's. One candidate is the builder yielding `(tuning key, callable)`
tuples rather than ATI enumerating a perf struct, but that is unsettled and this is far too
early in the adoption to fix it. Empty sections keep the archive shape right (6.2, 7a)
without committing to an answer.

`ir/interface.py` is untouched, and is not a language module. `Interface` is already the
generic concept — five implementers, only two of which are backends:

| implementer | where it ends up |
|---|---|
| `KernelDescription` (`ir/kdesc.py:51`) | `triton/` |
| `AffineKernel` (`ir/affine.py:39`) | `affine/` |
| `MetroKernel`, `ConditionalKernel` (`ir/metro.py:83,36`) | shared |
| `Operator` (`ir/operator.py:21`) | shared |

Operator and metro implement it without being a language at all, which is the tell.
`Interface` is the shared interface of **every callable AOTriton generates** — ordinary OOP
doing what OOP is for, not a HAL papering over backend differences — and `codegen/`
dispatches through it polymorphically.

Nothing about the per-language split touches it. If some of its code should later move into
`ir/lib/`, that is its own task with its own justification, not a rider on this one.

**Cost is small.** Non-test import sites: `ir/kdesc.py` has 2 (`ir/operator.py`,
`codegen/linker.py`), `ir/ksignature.py` has 2 (`ir/kdesc.py`, `codegen/autotune.py`),
`ir/affine.py` has 1 (`codegen/linker.py:149`). Five, plus tests. The much larger `grep`
counts are the local variable `kdesc = self._iface`, which is untouched by a move.

**What flyc's signature actually is.** Not the Triton class reused — `ir/flyc/ksignature.py`
is its own small class whose whole job is the entry name: it calls `ir/lib/`'s `entry_name`
with perf and copt left empty. No `COMPILER_OPTIONS`, no perf struct, no `num_warps` /
`num_stages`. The empty `<perf>` / `<copt>` sections 7a wants are what a signature with no
tuning naturally produces, not a special case to arrange. On the description side
`ir/flyc/kdesc.py` takes `EMPTY_PERF_STRUCT` (`specs/tune.py:168`) like any kernel with no
`@ati.tune.schema` — that one is shared, not Triton-specific.

**How `codegen/autotune.py` gets reused — next phase, not this one.** With one kernel per
functional and empty psels/copts, flyc looks to that generator exactly like a Triton kernel
built with no tuning database and default options. That is the whole trick: the reuse needs
no new generator, just a signature that answers the same questions with empty answers. It is
also squarely **Phase 2** work. Phase 1 stops at emitting enough build rules to produce
hsacos from flyc kernels and pack them (Task 7); it generates no autotune code and no LUT.

What stays deferred beyond that is only *what shape* the answers eventually take. FlyDSL's
knobs (`block_m`, `row_subtiles`, `k_prefetch_dist`) are not `num_warps` / `num_stages`, and
the programmatic build may not enumerate a perf struct at all — a builder yielding
`(tuning key, callable)` tuples is one candidate. Unsettled, and correctly so at this
stage.

This is the concrete payoff of library-over-abstraction. `KernelSignature.__init__` currently
does `self._copts[COPT_NWARPS_INDEX] *= 2` when `f.arch == 'gfx1250'` — a Triton-specific
workaround (gfx1250 falls back to gfx942's tuning database and needs double the warps to
dodge a compiler issue), kept only so nothing changes accidentally, and to be deleted once
gfx1250 is properly supported. Had flyc inherited or reused the Triton class it would have
inherited that too, and an empty copt list would index-error on it. With a per-language file
the hack stays in `triton/ksignature.py`, where it is true, and flyc never sees it. Mark it
there as Triton-only and temporary so the eventual deletion is obvious.

## Verification

No GPU is touched and no kernel is launched at any point.

1. `python -m aotriton.flyc_compile modules/flash/flyc/flash_attn_func_gfx1201_aiw.py
   --target gfx1201 --head_dim 64 --dtype f16 --causal 0 --out_path /tmp/k`
   → expect a ~13 KB ELF, `Flags: 0x4e, gfx1201`, symbol `flash_attn_func_aiw_kernel_0`.
2. Repeat over a small matrix — `head_dim` 32/64/128, `causal` 0/1, `dtype` f16/bf16 — so the
   knob-resolution path is exercised, not just one tile.
3. `llvm-readelf -h --symbols` on each artifact (the `--verify` step, also runnable by hand).
4. Negative check: unset `ROCM_PATH`, confirm the driver fails with the actionable message
   rather than `lld invocation failed`.
5. Confirm the build venv stays torch-free: `python -c "import torch"` must fail in `VENV_DIR`
   while `python -m aotriton.flyc_compile` still succeeds.
6. Confirm `flyc/` has no `__init__.py` and that every file in it is importable with only
   `modules/flash/flyc` on `sys.path` — the same contract `modules/flash/kernel/` holds.
7. Re-sync check: copy the vendored files fresh from `third_party/flydsl`, reapply the
   `UPSTREAM.md` import table, and confirm the result is identical to what is checked in.

Correctness of the emitted code objects is explicitly **out of scope** — that needs a gfx1201
device and is a separate follow-up.
