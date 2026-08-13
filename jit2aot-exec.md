# Executable plan: make `flyc_compile` kernel-agnostic

Ordered task list. Design rationale is in `jit2aot.md`; this file is the work.

**Goal.** `python/flyc_compile.py` stops knowing anything about FlyDSL attention. It enters at
the `@flyc.jit` `JitFunction` instead of the host wrapper, synthesises arguments from the
launcher's own signature, and takes `choices` rather than a fabricated `Functional`. The
artifact must not change: the hsaco is byte-identical before and after.

---

## Rules of engagement

- **Never launch a GPU kernel.** No gfx1201 device; GPU access is deliberately blocked.
  `COMPILE_ONLY=1` is what makes every step safe.
- **Never `pip install`.** Firewalled. If something is missing, STOP and report.
- **Never modify `/home/xinyazha/dockerhome/meff/FlyDSL`** — separate repo, read-only.
- **Commit per step**, small and reviewable, each leaving the suite green. End every message
  with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. Do not push.
- **Working from a git worktree?** `aotriton` is editable-installed against the MAIN checkout
  (`/home/xinyazha/dockerhome/meff/flyati/python`), so `pytest` and `python -m aotriton.*` run
  naively will exercise the *main* tree, not your edits. Shadow it first:
  ```bash
  mkdir -p /tmp/shadow_j2a && ln -sfn <WORKTREE>/python /tmp/shadow_j2a/aotriton
  PYTHONPATH=/tmp/shadow_j2a python -c "import aotriton; print(aotriton.__file__)"   # must be your worktree
  ```

## Caching: what is off, and the one that is not

The byte-identity gate below is only meaningful if each measurement is a real compile.
Measured:

| cache | state | evidence |
|---|---|---|
| flydsl disk cache | **off** | `flyc_bootstrap` sets `FLYDSL_RUNTIME_ENABLE_CACHE=0`; both gates (`jit_function.py:1250`, `:1409`) read `enable_cache or run_only`; `~/.flydsl` is never created |
| cross-build reuse | **none** | two independent builds of one config: 1.07 s and 1.06 s, same sha |
| `JitFunction._mem_cache` | **live, per instance** | same `jf` called twice: 1.02 s then **0.00 s** |

`_mem_cache` is populated unconditionally by design ("keep compiled artifacts alive within the
process even when disk cache is off"). It is harmless in production — `flyc_compile` is one
process, one build, one fresh `JitFunction` — but it is a trap when verifying:

> **A verification script that builds twice in one process and reuses the same `JitFunction`
> gets a 0.00 s cache hit on the second, and proves nothing.** Either shell out to the CLI
> (separate processes, which is what the sweep does), or call `build_..._primary` again to get
> a fresh `JitFunction`. Treat a sub-0.1 s "compile" as a cache hit, not a fast machine.

## Verified facts — treat as given, do not re-derive

| fact | value |
|---|---|
| python | `/home/xinyazha/.venvs/nogpu/bin/python` |
| required env | `AOTRITON_FLYDSL_ROOT=/home/xinyazha/dockerhome/meff/FlyDSL` (no fallback exists) |
| pytest baseline | **192 passed, 7 skipped** |
| reference artifact | hd 64 / f16 / non-causal → **13496 bytes**, sha256 prefix `bc6d0fca66a0c2d9f476` |
| ELF | `EM_AMDGPU`, `Flags: 0x4e, gfx1201`, symbol `flash_attn_func_aiw_kernel_0`, `shared` 8960 |
| block_m / block_size | 256 / 512 at hd 64; **512 → 256** at hd 128 |
| launcher | `launch_flash_attn_aiw`, 44 params: 14 Pointer, 11 Int32, 16 Int64, 2 Float32, 1 Stream |

Measured and relied on below:

- The 44 argument **values** do not affect the artifact. Annotation-derived zeros give a
  byte-identical hsaco to the full host-marshalling path.
- The `fx.Pointer` **element type** is inert — `Uint8`/`Float16`/`Int32` all identical.
- `flyc.compile()` is unusable: `_compile_impl` ends in `artifact._get_func_exe()`, which
  builds an ExecutionEngine and needs HIP. Call the `JitFunction` directly.

---

## Step 1 — signature + object helpers

Add to `python/flyc_compile.py`. Nothing is wired in yet; this step only adds and tests.

**`_launcher_signature(jf)`** — use flydsl's own resolver, not `inspect.signature`:
```python
from flydsl.compiler.jit_argument import resolve_signature   # local import, after bootstrap
```
`resolve_signature` is `inspect.signature(func, eval_str=True)` (`jit_argument.py:35`) and is
what `JitFunction._ensure_sig` binds against (`jit_function.py:1224`). **This matters**: four
`@flyc.jit` files in the FlyDSL tree use `from __future__ import annotations`, so their
annotations are *strings*; bare `inspect.signature` would yield `'fx.Pointer'` and every
downstream check would misfire. Signature comes from `jf.func` (the AST-rewritten function),
matching flydsl.

**`_find_unique(obj, cls)`** — one search, two callers. Walk closure cells, and also tuple /
list / dict values (`pa_decode_swa` returns `{'launch': jit, 'kernel': kernel}`; `moe_sorting`
returns a 9-tuple). Collect by identity; return the single match, or raise naming how many
were found.

**`jit_function_of(built)`**:
1. `isinstance(built, JitFunction)` → return it. **This is the common case** — 47 of 69
   builders in the FlyDSL tree return their `@flyc.jit` directly.
2. else `_find_unique(built, JitFunction)` — 18 builders, gfx1201 among them. For gfx1201 the
   JitFunction is in `built.compile`'s closure.
3. zero or several → raise. Several means a **multi-kernel builder** (3 in the tree); that is
   out of scope and needs a plural contract, not a guess. Say so in the message.

**`kernel_function_of(jf)`** — `_find_unique(jf.func, KernelFunction)`. Reachable from the
launcher's closure, **not** from `built.compile`'s.

**Files**

| | path |
|---|---|
| MOD | `python/flyc_compile.py` |

**Verify** — against the real builder, not a mock:
```python
jf = jit_function_of(built);  len(_launcher_signature(jf).parameters) == 44
kernel_function_of(jf)._known_block_size == [512, 1, 1]     # hd 64
```
and `[256, 1, 1]` at hd 128. Suite still 192/7.

---

## Step 2 — synthesise arguments from the signature

**`_operand_for(param, desc=None)`** returning the value for one parameter:

| annotation | value |
|---|---|
| `Pointer` | `flyc.from_c_void_p(desc.dtype if desc else fx.Uint8, 0)` |
| `Int32`, `Int64` | `0` |
| `Float32` | `0.0` |
| `Stream` | `fx.Stream(None)` |
| `Tensor` | raise — the existing `_assert_supported_operands` message |
| anything else | raise, naming parameter and annotation |

**`synthesise_args(jf)`** → the positional list. Takes no functional, no choices, no shapes.

Keep `FakeTensor` and its docstring — it is the descriptor type the `Tensor` row will need,
and `jit2aot.md` records why (`MemRefJitArg` needs `element_bits`/`shape`/`strides`/`dtype`,
none derivable from an annotation). Phase 1 passes `desc=None` everywhere.

Fold `_assert_supported_operands` into `_operand_for`'s `else` branch rather than keeping two
places that enumerate annotations — but keep the wording, especially the `fx.Tensor` note.

**Files**

| | path |
|---|---|
| MOD | `python/flyc_compile.py` |

**Verify**: `jf(*synthesise_args(jf))` then extract → sha256 prefix `bc6d0fca66a0c2d9f476`,
13496 bytes. The guard still fires on a synthetic `fx.Tensor` launcher.

---

## Step 3 — swap the entry point, delete the host-marshalling path

Replace the `_trace_fmha_launch(built, functional)` call with
`jf = jit_function_of(built); args = synthesise_args(jf); jf(*args)`.

**Delete** (all of it, not just the call site):

| symbol | line (approx) | why it goes |
|---|---|---|
| `_trace_fmha_launch` | 296 | replaced |
| `_FAKE_SHAPE` | 254 | the launch shape is no longer passed |
| the `abi.run_compiled` monkeypatch | in 296 | never restored; a global mutation of an imported module |
| `import fmha_abi_gfx1201` | in 296 | the last `fmha_*` import in the driver |
| the `CAUSAL_TYPE=3` window special-case | in 296 | only existed because `_launch` called `resolve_window` |
| `philox_seed=None` | in 296 | only existed to stop `u64_scalar` allocating a torch tensor |

Every one is a symptom of entering at `_launch`. None has a replacement.

**Files**

| | path |
|---|---|
| MOD | `python/flyc_compile.py` |

**Verify**: full Gate 3 — the literal command plus the 12-combination sweep
(`BLOCK_DMODEL` ∈ {32,64,128} × `CAUSAL_TYPE` ∈ {0,3} × `Q` ∈ {`*fp16:16`,`*bf16:16`}).
`CAUSAL_TYPE=3` must now pass **without** any window handling — if it does not, the entry
point is still going through host code somewhere. `grep -n "fmha_" python/flyc_compile.py`
must return nothing outside docstrings.

---

## Step 4 — `choices`, not a fabricated `Functional`

The driver has only `--signature` text. A `Functional` carries `arch_number`, `godel_number`,
a `meta_object` back-reference and the axis table, all from the linked IR the driver does not
have — so the stand-in fabricates two attributes and drifts silently the moment a description
reads a third.

- **Delete** `_Choices` (88) and `_FunctionalStandIn` (100).
- Pass `parse_kv(args.signature, sep=' ')` — a plain `dict` — straight to the body.
- New contract: `def flyc_attn_fwd(choices, hints)`, returning `(built, sidecar_dict)`.
- Update the demo body: `f.choices.BLOCK_DMODEL` → `choices['BLOCK_DMODEL']`, and likewise
  `CAUSAL_TYPE`, `Q`, `BIAS_TYPE`, `ENABLE_DROPOUT`, `PADDED_HEAD`.

**Do not touch `_flyc_fwd_disabled`.** Disable predicates run **generator-side**, where a real
`Functional` exists, and it is the only reader of `f.arch`. The asymmetry is intentional:
disable takes a `Functional`, the build body takes `choices`. Document it in the description's
docstring so nobody "fixes" it.

**Files**

| | path |
|---|---|
| MOD | `python/flyc_compile.py` |
| MOD | `modules/flash/aot/flyc_attn_fwd.py` |

**Verify**: Gate 3 + the sweep again. `grep -n "Functional" python/flyc_compile.py` returns
nothing but comments.

---

## Step 5 — `block_size` from the declared value

Replace `_extract_block_size(source_ir)` (349) with
`kernel_function_of(jf)._known_block_size[0]`.

Do **not** use the ELF's `.max_flat_workgroup_size`: it is a *bound*, and `.reqd_workgroup_size`
is emitted **empty**, so the exact launch geometry is not in the artifact. It happens to read
512 here only because flydsl derives `flat_work_group_size = "512,512"` (min == max) from
`known_block_size`. flydsl validates the real launch against `_known_block_size` in
`KernelLauncher._check_block_vs_known`, so the declared value is authoritative.

This retires the last use of `CompiledArtifact._source_ir`; the driver then touches only
`_ir_text` and the ELF.

**Files**

| | path |
|---|---|
| MOD | `python/flyc_compile.py` |

**Verify**: sidecar `block_size` is 512 at hd 64 and **256 at hd 128** — the second value is
the one that proves it is read, not hardcoded.

---

## Step 6 — tighten the agnosticism gate

The Gate 3 check greps for `fmha_tuning_gfx1201` and `attn_fwd`; it passed for weeks while
`import fmha_abi_gfx1201` sat in the driver. Widen it to any `fmha_*`, `flash*` or `attn*`
module import in `python/flyc_compile.py`, and record the new check in
`modules/flash/flyc/PLAN-PHASE1.md` Gate 3.

**Files**

| | path |
|---|---|
| MOD | `modules/flash/flyc/PLAN-PHASE1.md` |

---

## Definition of done

1. `python/flyc_compile.py` imports no `fmha_*` / `flash*` / `attn*` module, and contains no
   `FakeTensor`-as-launch-argument, no `_FAKE_SHAPE`, no monkeypatch, no window or philox
   special-casing, no `Functional` stand-in.
2. `FakeTensor` and its docstring remain, unused by the pointer path, as the descriptor the
   `fx.Tensor` row will need.
3. The artifact is unchanged: **13496 bytes, sha256 `bc6d0fca66a0c2d9f476`** at hd 64 / f16 /
   non-causal. This is the single most important check — a kernel-agnostic driver that emits
   a different binary has failed.
4. 12-combination sweep passes, `CAUSAL_TYPE=3` included, with no window handling anywhere.
5. Sidecar carries `block_m` (from the description's dict) and `block_size` (from
   `_known_block_size`), 256/512 at hd 64 and 128/256 at hd 128.
6. Suite: 192 passed, 7 skipped.
7. No GPU used, no kernel launched.

## Out of scope

`fx.Tensor` operands (asserted, see `FakeTensor`'s docstring); multi-kernel builders (raise);
Phase 2's kernarg vector, `ati.context_helper` codegen, C++ shim; the upstream
`_launch.jit_function` (step 1's closure walk is the interim and a one-line swap later).

## Hazards

- **The closure walk is the fragile part.** Steps 1's `jit_function_of` and
  `kernel_function_of` both reach into internals. If either stops finding its object, do not
  broaden the search until it matches something — report it. The fix is upstream.
- **`jf.func` vs `jf._original_func`.** Use `jf.func`: it is what flydsl binds against. The
  AST rewrite preserves the signature.
- **Byte-identity is the real gate.** Every step above can "work" — produce a valid gfx1201
  ELF — while quietly changing the binary. After *each* step, re-run the compile and
  `sha256sum` the emitted `.hsaco` against `bc6d0fca66a0c2d9f476`. Not just at the end: if it
  drifts you want to know which step did it.
