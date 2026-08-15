* `python/flyc_bootstrap.py`: let `ROCM_PATH` resolves to `rocm-sdk path --root` if `ROCM_PATH` is not set, and rocm-sdk is available
* `from aotriton.template_instantiation.ir.triton.kdesc import KernelDescription` is too verbose, can we import `from aotriton.template_instantiation.ir.triton import KernelDescription` instead?
  + An additional question is can we `import aotriton.template_instantiation as ati`, `KernelDescription = ati.ir.triton.KernelDescription` and then we can use `KernelDescription`?
    - The point is multiple `from aotriton.template_instantiation.XXXXXXXXXX import XXXXX` is too verbose
* `lib_naming.entry_name`
  + let it accept functional directly
* `684190d` is not needed, which only updates the comment and I don't see the needs to
* `from ...decorators import TensorSpec` contains to many dots, is it easier to read with `from aotriton.XXX import`?
* `python/template_instantiation/ir/triton/kdesc.py` has too many lazy imports (unlike torch, I don't think these lazy imports are necessary)
* Comment at `4e73bcf` is too long, half it
* We need to discuss `_trace_fmha_launch`

---

## Progress (updated by Claude)

### Phase 1 status

| task | state |
|---|---|
| 0a shared helpers | **RESOLVED.** Two pins: `flydsl-compiler.txt` (wheel) and `flydsl-kernel.txt` (git tag `v0.3.1`, shallow-cloned by CMake). WMMA helpers, absent from the tag, come from `flyc_polyfill.py`. |
| 0b `flyc_polyfill.py` | done, merged |
| 0c `ir/` reorganised by language | done, merged. Gate green: 192 passed / 7 skipped both before (`8d24bd8`) and after. |
| 1 vendor the kernel | done, merged. The 2 verbatim files are byte-identical to upstream; the other 3 differ only by the documented import rewrites. |
| 2 `flyc_bootstrap.py` | done, merged |
| 2.5 flydsl in the build venv | done, merged. **Gate now FULLY verified** — see "CMake unblocked" below. The old `find_package(hip REQUIRED)` failure was upstream #207, fixed by rebasing. |
| 3 `flyc_compile.py` | done, merged. Then **rewritten** by `jit2aot-exec.md` steps 1-6 (`4731404`..`1ea44e5`) — see below. |
| 4 `ati.flyc.*` + `ir/flyc/` | done, merged. **Gate 3 verified independently**: 13496-byte ELF, `EM_AMDGPU`, `Flags: 0x4e, gfx1201`, symbol `flash_attn_func_aiw_kernel_0`, sidecar `block_m=256` / `block_size=512`; 12/12 sweep. |
| 5 generator emits `Fly.compile` | done, merged (`751b186c`). Gate 5 verified: 288 rows, 7 fields, `gfx1201`, no dup outputs. |
| 6 CMake rule loop | done, merged (`c18ade5e`). Gate 6 verified: 288/288 hsacos, ELF spot-checks pass. **hd 48 is flaky upstream** — see below. |
| 7 aks2 / flatzip packaging | done, merged (`3b3279e1`). **Deliverable exists**: `aotriton.images/amd-gfx120x/flash/flyc_attn_fwd.zip`, 288 entries. |

### This review

**Items 1-7: done and merged.** Suite 192 passed / 7 skipped; driver still emits the
13496-byte gfx1201 ELF. Two follow-ups I made on top:

- item 6's hoisted imports were 3-dot relative, the exact pattern item 5 removed. Made
  absolute; no 3-dot relative imports remain under `template_instantiation/`.
- items 1 corrected *my* error, not the agent's: I had written that `rocm-sdk path --root`
  returns an unexpanded tree and should not be used. Wrong — I confused `_rocm_sdk_devel`
  (expanded, has `llvm/bin/ld.lld`) with the `rocm_sdk_devel` Python package (whose 2.6 GB
  `_devel.tar` is unexpanded until `rocm-sdk init`). One underscore apart. `rocm-sdk path
  --root` is now candidate 2 and is what resolves in this container. SURVEY.md and
  PLAN-PHASE1.md corrected in two places each.

**Item 8 is answered, and your instinct was right — see below. Needs one decision from you.**

### Item 8: FakeTensor is unnecessary. Measured.

The 44 launcher arguments can be built from the launcher's own annotations alone:
`Pointer -> flyc.from_c_void_p(fx.Uint8, 0)`, `Int32/Int64 -> 0`, `Float32 -> 0.0`,
`Stream -> fx.Stream(None)`. The resulting hsaco is **byte-identical** to the FakeTensor
route (`bc6d0fca66a0c2d9f476` both ways).

`PointerJitArg` (`flydsl/compiler/jit_argument.py:592`) only ever needs an element type and
an integer — `__get_ir_types__` derives the IR type from `element_type`. The tensor
requirement belongs entirely to the JIT-facing host wrapper, whose whole job is turning real
torch tensors into exactly these values. We were entering one layer too high.

Going in at the `JitFunction` deletes all of: `FakeTensor`, `_FAKE_SHAPE`, the
`abi.run_compiled` monkeypatch (never restored today), `import fmha_abi_gfx1201` (the last
`fmha_*` import in the "kernel-agnostic" driver), the `CAUSAL_TYPE=3` window fix,
`philox_seed=None`, and reading `functional.choices` in the driver. Every one of those was a
symptom of the wrong entry point. It is also *more* generic than moving the code into
`flyc_attn_fwd.py`: the arg list comes from `inspect.signature`, so it works for any
`@flyc.jit` launcher.

**The one open problem: getting the `JitFunction`.** It is a closure local inside
`build_flash_attn_func_aiw_module_primary`; `dir(built)` exposes only `compile` and the
`varlen_*` helpers. It *is* reachable via `built.compile.__closure__` (verified), but that is
reaching into internals.

**Decision needed** — one line upstream in FlyDSL (`_launch.jit_function = launch_flash_attn_aiw`)
is the clean fix, and is the minimal form of PLAN.md open question 2. Either:
  (a) ask the FlyDSL session for it now and build against it, or
  (b) use the closure walk as a documented interim, one-line switch when it lands.

### Environment notes worth keeping

- `pytest` + an **editable** `aotriton` are installed. The editable install points at the
  main checkout, so anything run from a git worktree silently tests the *main* tree unless
  `PYTHONPATH` is shadowed. This nearly produced a false green twice.
- `AOTRITON_FLYDSL_KERNEL_ROOT=<FlyDSL source root>` must be set for every
  `flyc_compile` run. CMake sets it to the `third_party/flydsl-kernel.txt` clone; set it
  by hand for a direct run. There is no fallback (the old `third_party/flydsl` fallback
  died when 2.5 chose a pinned wheel over a submodule).
- ~~No HIP toolchain here~~ — wrong diagnosis, see "CMake unblocked" above. ROCm was
  installed and fine; the block was `/opt/rocm` hardcoded in the pre-#207 `CMakeLists.txt`.
- Every cmake invocation here needs three things set:
  `ROCM_PATH=$(rocm-sdk path --root)`, `PIP_NO_INDEX=1`, and
  `PIP_FIND_LINKS=~/.cache/aotriton-wheelhouse`.

### jit2aot steps 1-6: done and merged

Item 8 was resolved as **(b)** — closure walk as a documented interim — and the whole of
`jit2aot-exec.md` is now implemented, six commits `4731404`..`1ea44e5`, merged
fast-forward. The driver enters at the `JitFunction` and never touches host code.

Deleted, all of it symptom rather than cause: `_trace_fmha_launch`, the
`abi.run_compiled` monkeypatch, `import fmha_abi_gfx1201`, `_FAKE_SHAPE`, the
`CAUSAL_TYPE=3` window fix, `_Choices`/`_FunctionalStandIn`, and
`_extract_block_size` (which parsed pre-lowering MLIR; `block_size` now comes from the
declared `_known_block_size`, retiring the last use of `_source_ir`).

`FakeTensor` survives on purpose — unused by the pointer path, kept as the descriptor the
`fx.Tensor` row of `_operand_for` will need, with the open questions written into its
docstring.

**Verified independently, not taken on report** — worktree shadowed via `PYTHONPATH` so
the editable install could not serve the main tree:

- reference artifact `1821491bae4d1ca3c2f1`, 13496 bytes, 1.39 s (a real compile, not a
  `_mem_cache` hit)
- 12/12 sweep `Complete`, **12 distinct sha256** across 12 fresh subprocesses — which is
  also the proof there is no cross-process cache reuse
- `CAUSAL_TYPE=3` passes at all three head dims with **no window code anywhere** in the
  driver
- 45 launcher params (14 Pointer / 12 Int32 / 16 Int64 / 2 Float32 / 1 Stream);
  launcher-minus-`stream` equals the kernel's 44 names **in order** — your ABI alignment
  request landed and is now load-bearing
- `_known_block_size` `[512,1,1]` at hd 64, sidecar `block_size` 512, `block_m` 256
- suite 192 passed / 7 skipped

Two stale numbers in `jit2aot-exec.md` were found by the agent and corrected in `5cc0e51`
(a `44` that should have been `45`, and an ambiguous `block_m`/`block_size` shorthand).

### CMake unblocked: rebased onto upstream/main (#207)

`find_package(hip REQUIRED)` was not a missing toolchain and not a missing `ROCM_PATH`
alone — the old `CMakeLists.txt` hardcoded `/opt/rocm` into `CMAKE_PREFIX_PATH` with no
override, so your venv ROCm was unreachable whatever the environment said. #207 replaces
it with `$ENV{ROCM_PATH}/lib/cmake`. Rebased: 52 commits onto `5d3ffed0`, **no
conflicts**; only `CMakeLists.txt` and `docs/AltWheelExample.yaml` were touched by both
sides, in disjoint regions. Verified the rebase introduced exactly #207 and nothing else
by diffing `prerebase-flydsl-backup..HEAD` against `209d484..upstream/main` — identical.
Backup tag `prerebase-flydsl-backup` still exists; **not pushed** (rebase makes it a
force-push, your call).

With `ROCM_PATH=$(rocm-sdk path --root)`: hip 7.14.60850 found, configure clean in both
image and noimage mode, **Gate 2.5 fully passed**, and the CMake-built venv's own python
emits the byte-identical hsaco. Details and the offline recipe are in PLAN-PHASE1.md's
Gate 2.5 section rather than here, since they belong with the task.

The one gotcha worth repeating: `rocm-sdk path --root` → `_rocm_sdk_devel` is the only
directory with **both** `lib/cmake/hip` and `llvm/bin/ld.lld`. `flyc_bootstrap`'s
candidate 3 (`_rocm_sdk_core/lib`) has the linker but no cmake config, so the two
resolutions are not interchangeable. They live in different scopes today (configure-time
env vs driver process), but that is luck, not design.

**Your pip cache did the job** — 1.2 GB of it, holding the wheels from your earlier
installs. Reconstructed a wheelhouse (trimmed to 103 MB) at
`~/.cache/aotriton-wheelhouse`, driven with `PIP_NO_INDEX=1 PIP_FIND_LINKS=...`. All ten
`requirements.txt` entries plus `flydsl==0.3.1` recovered.

**Triton is no longer a blocker either**, with the wheel you supplied at
`~/triton-3.7.0+gitdb82b800.aotriton0.14-...whl` passed as
`-DAOTRITON_USE_LOCAL_TRITON_WHEEL`. The build venv now has triton 3.7.0 + flydsl 0.3.1
and still no torch, ~51k image rules generate, and one Triton kernel builds end to end
(43,480-byte gfx1201 hsaco, `compile_status: Complete`). So the full
`.hsaco` → `.aks2` → `aotriton.images/*.zip` chain — exactly Task 7's target shape — is
exercisable here.

Configure also clones `https://github.com/ROCm/aiter.git` (~140 MB,
`v3src/CMakeLists.txt:115`). That works: github is reachable from this container even
though PyPI is not.

Keep the wheel out of the repo root — `.gitignore` does not cover `*.whl`, so `git add -A`
would sweep 412 MB into a commit.

Also fixed while gating: `flyc_bootstrap`'s dead `<repo>/third_party/flydsl` fallback
(`1a0b182c`). PLAN-PHASE1.md:376 already said the variable is required with no fallback;
the code had drifted, and the stale default produced a *misleading* error under a
non-editable install — it advertised `<site-packages>/third_party/flydsl`.

### Tasks 5-7: done and merged. Phase 1's deliverable exists.

`751b186c` / `c18ade5e` / `3b3279e1`, merged fast-forward. Verified independently in a
worktree-shadowed build, not taken on report:

- **Gate 5** — 288 rows, field count exactly 7, column 5 exactly `gfx1201`, no duplicate
  outputs. Under `-DAOTRITON_NOIMAGE_MODE=ON`, `Fly.compile` is 0 bytes (see below).
- **Gate 6** — 288/288 hsacos, three ELF spot-checks `EM_AMDGPU` / `Flags: 0x4e, gfx1201`,
  plain `ninja` builds them.
- **Gate 7** — `aotriton.images/amd-gfx120x/flash/flyc_attn_fwd.zip`, 288 entries, all
  `STORED`. Unpacked one aks2: 1 kernel, embedded ELF `AMDGPU - HSA` / gfx1201. **No CMake
  change was needed**, exactly as your `Fly.compile`-only simplification predicted.
- suite 192 passed / 7 skipped; plain `ninja` links `libaotriton_v2.so.0.14.0`.

Three things worth your attention, none of them defects in Tasks 5-7:

1. **`BLOCK_DMODEL=48` is nondeterministic upstream.** First full pass produced 4 zero-byte
   stubs, all hd 48, from `_extract_hsaco`'s assertion: the module is serialized twice
   (plain and `no_wave64` targets) and the two results differ in 10,178 of 19,512 bytes.
   Intermittent — 8 repeats of one config gave 7 ok / 1 fail; hd 64 was 8/8 clean. Three
   delete-and-rebuild passes converged to 0. So every config *can* build, but a CI run
   cannot assume first-pass success. `hd 48` is a legitimate `FLYC_HEAD_DIMS` entry, so
   dropping it would hide the bug rather than fix it. Worth an upstream FlyDSL report.
2. **`.aks2` never rebuilds when its hsaco changes.** `DEPENDS aotriton_v2_compile` is an
   **order-only** ninja edge (`||` in `ninja -t query`), so after fixing the flaky hsacos
   the zip still held stale stubs and `ninja` said nothing to do. Pre-existing, identical
   for Triton, and the same root cause as the "one aks2 builds all 47,766 hsacos" cost
   noted in Gate 7 — one edge that is too coarse in both directions.
3. **`write_image_signature.py` imports triton unconditionally**, so even a
   `-DAOTRITON_DEBUG_SKIP_TRITON_KERNELS=ON` build needs the triton wheel installed in the
   venv. Pre-existing; my debug option deliberately leaves the wheel install alone.

Also corrected: Gate 5 said `Fly.compile` must be **absent** under noimage. It cannot be —
my `touch()` fix made every `shard_names` entry always exist. All four rule files are
0 bytes there; the behaviour under test is unchanged. Plan updated.

### Where to continue

Tasks 5 → 6 → 7, and **all three are now fully gateable in this container** — none of them
were this morning. Image-mode configure works, triton and flydsl both install into the
build venv, and the `.hsaco` → `.aks2` → `aotriton.images/*.zip` chain runs.

The environment recipe, needed for every cmake invocation here:

```
export ROCM_PATH=$(rocm-sdk path --root)
export PIP_NO_INDEX=1 PIP_FIND_LINKS=~/.cache/aotriton-wheelhouse
cmake -S . -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAOTRITON_NO_PYTHON=ON -DAOTRITON_TARGET_ARCH=gfx1201 \
  -DPython3_EXECUTABLE=/home/xinyazha/.venvs/nogpu/bin/python \
  -DAOTRITON_USE_LOCAL_TRITON_WHEEL=~/triton-3.7.0+gitdb82b800.aotriton0.14-cp313-cp313-linux_x86_64.whl
```

A configured image-mode build dir is at `/tmp/bld4`, now **5.6 GB** with aiter and 11,233
hsacos. It is in `/tmp`, so it will not survive a reboot — worth relocating before Tasks
5-7 lean on it, because rebuilding it is not cheap:

**Every `.aks2` depends on every hsaco.** `v3src/CMakeLists.txt:262` gives each aks2 rule
`DEPENDS aotriton_v2_compile`, and line 226 makes that target depend on all `${ALL_HSACOS}`
— 47,766 for one arch. The real per-cluster list is invisible to ninja: it rides in the
`.nsv` behind `--hsaco_manifest`, so the build compensates with a blanket dependency. I
asked for one attn_fwd cluster (nominally ~17 kernels) and got 11,233 hsacos in ~18 min
before cutting it off at 24%. Task 7's gate therefore needs one deliberate full build, not
an edit-rebuild loop. Details and a suggested fix are in PLAN-PHASE1.md's Gate 7.

Still outstanding, unchanged:

- **0a is done.** `third_party/flydsl-kernel.txt` pins `v0.3.1` and CMake shallow-clones
  it (9.6 MB); `AOTRITON_FLYDSL_KERNEL_ROOT` is a cache PATH, so pointing it at a checkout
  skips the clone. The tag has no `mma/wmma_ops.py` — it is MFMA-only, this is a WMMA
  kernel — so those helpers come from `flyc_polyfill.py`, which already held exactly the
  six fallbacks and was dead code until now. Byte-identical against both the released tag
  and the feature branch. Everything still collapses the day the wheel ships
  `flydsl.kernels.common`.
- the `num_seqlens`/`batch_size` context helpers want checking against `flyc_varlen_bits`
  for the `<0` padded case.
- `jit_function_of`'s closure walk is the documented interim. One line upstream
  (`_launch.jit_function = launch_flash_attn_aiw`) retires it; the driver raises rather
  than broadening the search if it ever stops finding exactly one.
