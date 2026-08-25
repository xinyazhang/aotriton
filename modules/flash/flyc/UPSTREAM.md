# Provenance and re-sync instructions

## Source

```
repo:   git@github.com:xinyazhang/FlyDSL.git
branch: xinyazhang/sdpa-gfx1201-feature
commit: caee9257  (was 93d8d497f8e9bbc66106617feb851cf0fb12acd3)
path:   kernels/attention/parity/
```

Get the commit fresh with:

```bash
git -C <flydsl checkout> rev-parse HEAD
git -C <flydsl checkout> branch --show-current
```

## Vendored files (verbatim at the commit above, before the rewrites in "Import
rewrites" below are reapplied)

```
flash_attn_func_gfx1201_aiw.py      forward
fmha_tuning_gfx1201.py              forward tuning policy
fmha_bwd_dkdv_gfx1201_kernel.py     backward dK/dV
fmha_tuning_bwd_dkdv_gfx1201.py     backward dK/dV tuning policy
fmha_bwd_dq_gfx1201_kernel.py       backward dQ/dB
fmha_tuning_bwd_dq_gfx1201.py       backward dQ/dB tuning policy
fmha_abi_gfx1201.py                 shared host ABI
fmha_common_gfx1201.py              shared device helpers
philox.py                           shared PRNG
```

Copied straight from `kernels/attention/parity/<basename>` to
`modules/flash/flyc/<basename>` with no renaming.

The four tuning modules import nothing but `dataclasses`, which is what lets the
ATI descriptions call `resolve_knobs` at GENERATE time without pulling in flydsl.
Keep it that way on re-sync: a flydsl import appearing in one of them breaks the
code generator, not just the build.

**Not vendored, and must never be**: `kernels/common/*` (`buffer_ops.py`,
`mem_ops.py`, `kernels_common.py`, `layout_utils.py`, `utils.py`,
`mma/wmma_ops.py`). These are FlyDSL's shared helpers, not the flash kernel's;
copying them here would fork six files that have nothing to do with attention.
They are resolved via `sys.path` (see "Task 0 interim" below), and the day the
flydsl wheel ships them under `flydsl.kernels.common`, only `flyc_bootstrap.py`
needs to change — not any file in this directory.

Also not vendored: `gfx1201_standalone.py` (existed only to put the FlyDSL repo
root on `sys.path`; replaced by the interim described below),
`dropout_mask_gfx1201.py`, the FUSED backward
(`fmha_bwd_fuse_gfx1201_kernel.py` + `fmha_tuning_bwd_fuse_gfx1201.py`), every
`*_interface.py` (torch-only — they are the JIT entry points, and AOTriton's C++
shim is what replaces them), `tooling/`, and every `test_*.py`.

## Task 0 interim — how `kernels.common` resolves today

`kernels/common` is not in the installed flydsl 0.3.1 wheel (verified:
`find_spec('kernels')` is `None` in the flydsl wheel). Until an
upstream packaging change ships it as `flydsl.kernels.common`, the build
shallow-clones a FlyDSL source tree at the `third_party/flydsl-kernel.txt` tag
and `python/flyc_bootstrap.py` puts its root — `$AOTRITON_FLYDSL_KERNEL_ROOT`,
required, no default — on `sys.path`, so the vendored imports below resolve
verbatim as `from kernels.common import ...`, exactly what the now-deleted
`gfx1201_standalone.py` did.

Two FlyDSL pins, deliberately independent:

| file | pins | used for |
|---|---|---|
| `third_party/flydsl-compiler.txt` | `flydsl==0.3.1` | the pip wheel: the compiler itself |
| `third_party/flydsl-kernel.txt` | `v0.3.1` | the git tag: a source tree supplying `kernels/common` |

**Do not copy `kernels/common/*` into this repo to unblock an import error.**
That is the fork this vendoring strategy exists to avoid. If the interim
`sys.path` mechanism is broken, fix `flyc_bootstrap.py`, not this directory.

## Import rewrites (1a)

Four lines, two files, replacing the four `gfx1201_standalone` imports that
the deleted shim used to serve — the ones a bare `parity/` checkout is not
importable without.

Under the Task 0 interim (current state), the right-hand side is
`from kernels.common import ...` (resolved via the `sys.path` insert in
`flyc_bootstrap.py`). Once the upstream packaging change lands, the
right-hand side becomes `from flydsl.kernels.common import ...` and only this
table's right column changes.

| file | replace | with (interim) | with (post-packaging) |
|---|---|---|---|
| `flash_attn_func_gfx1201_aiw.py` | `from gfx1201_standalone import buffer_ops` | `from kernels.common import buffer_ops` | `from flydsl.kernels.common import buffer_ops` |
| `flash_attn_func_gfx1201_aiw.py` | `from gfx1201_standalone import utils as common_utils` | `import flyc_polyfill as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_common_gfx1201.py` | `from gfx1201_standalone import buffer_ops, kernels_common, wmma_ops` | `from kernels.common import buffer_ops, kernels_common`<br>`import flyc_polyfill as wmma_ops` | `from flydsl.kernels.common import buffer_ops, kernels_common`<br>`from flydsl.kernels.common.mma import wmma_ops` |
| `fmha_common_gfx1201.py` | `from gfx1201_standalone import utils as common_utils` | `import flyc_polyfill as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_bwd_dkdv_gfx1201_kernel.py` | `from gfx1201_standalone import buffer_ops` | `from kernels.common import buffer_ops` | `from flydsl.kernels.common import buffer_ops` |
| `fmha_bwd_dkdv_gfx1201_kernel.py` | `from gfx1201_standalone import utils as common_utils` | `import flyc_polyfill as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_bwd_dq_gfx1201_kernel.py` | `from gfx1201_standalone import buffer_ops` | `from kernels.common import buffer_ops` | `from flydsl.kernels.common import buffer_ops` |
| `fmha_bwd_dq_gfx1201_kernel.py` | `from gfx1201_standalone import kernels_common as common_kernels` | `from kernels.common import kernels_common as common_kernels` | `from flydsl.kernels.common import kernels_common as common_kernels` |
| `fmha_bwd_dq_gfx1201_kernel.py` | `from gfx1201_standalone import utils as common_utils` | `import flyc_polyfill as common_utils` | `from flydsl.kernels.common import utils as common_utils` |

The two backward kernels take `smax` (dK/dV) and `smax`/`smin` (dQ) from
`common_utils` — both already in `flyc_polyfill.py` for the forward, so the
backward needed no new polyfill entry. `buffer_ops.get_element_ptr` and
`kernels_common.dtype_to_elem_type` are both present in `v0.3.1` and so are NOT
rewritten, same rule as the forward's.

`wmma_ops` moved from `flash_attn_func_gfx1201_aiw.py` to `fmha_common_gfx1201.py` in the
`93d8d497` re-sync (upstream's "the shared prologue moves to fmha_common"); the set of
symbols taken is unchanged, only which file takes them.

### Why two of those go to `flyc_polyfill` rather than `kernels.common`

The source tree the build clones is the **released** `v0.3.1` tag, and that
tag has no `kernels/common/mma/wmma_ops.py` at all — its `mma/` holds only
MFMA (CDNA) helpers, while this is a WMMA (RDNA) kernel. The gfx1201 WMMA work
lives on `xinyazhang/sdpa-gfx1201-feature`, which forked at `v0.3.0` and is not
contained in any tag.

Rather than pin a moving branch, the four `utils` helpers and the one
`wmma_ops` helper come from `flyc_polyfill.py`, which already carried exactly
these six fallbacks. Every symbol these two files take from those two modules
is branch-local, so each name aliases the polyfill *wholesale* and no call site
changes — which keeps the re-sync diff to one import line per file.

`buffer_ops` and `kernels_common` are deliberately NOT rewritten:
`get_element_ptr`, `_if_then` and `dtype_to_elem_type` are all present in
`v0.3.1`, so they still come from the clone. The rule below — do not copy
`kernels/common/*` into this directory — is unchanged and still applies.

Verified: building against the released `v0.3.1` tree and against the feature
branch produce the **byte-identical** hsaco (`1821491bae4d1ca3c2f1`, 13496
bytes), and the 12-combination sweep passes against `v0.3.1` with 12 distinct
artifacts. So the polyfill bodies are equivalent to the branch originals, and
the `buffer_ops`/`kernels_common` drift between `v0.3.1` and the branch does
not reach this kernel.

Unchanged and verbatim: `philox.py`, `fmha_tuning_gfx1201.py`. Also verbatim
within the two edited files: every flat sibling import already present
(`import fmha_abi_gfx1201 as abi`, `import fmha_common_gfx1201 as fmha`,
`from fmha_tuning_gfx1201 import (...)`, `from philox import Philox`,
`from philox import dropout_threshold`) — the flat, bare-directory layout is
what those already assume, same contract as `modules/flash/kernel/`.

## Torch-lazy rewrites (1b)

`fmha_abi_gfx1201.py` has two module-scope torch imports. The build venv must
never have torch (`CMakeLists.txt:142`), so both become function-local. Every
use is already inside a function.

**Three** insertion points as of `93d8d497`, up from two: upstream split the
rank-2 f32 row check into two callers, so `torch_f32` is now needed in both.

| where | before | after |
|---|---|---|
| module scope | `import torch` | deleted |
| module scope | `from torch import float32 as torch_f32` | deleted |
| the rank-2 f32 row check, first statement (`if t is None:`) | — | `from torch import float32 as torch_f32  # lazy: the build venv has no torch` |
| the logsumexp pointer helper, first statement (`if lse is None:`) | — | `from torch import float32 as torch_f32  # lazy: the build venv has no torch` |
| the philox u64 scalar helper, first statement | — | `import torch  # lazy: only reached for a plain int seed; AOT passes None` |

Locate them by searching for `torch_f32` and `torch.cuda.stream` rather than by
line number — the file has been restructured once already.

Neither rewrite changes behaviour: `lse_args`'s `torch_f32` check and
`u64_scalar`'s `torch.cuda.stream`/`torch.tensor` calls only execute when a
caller actually reaches that code path, and the AOT driver (`flyc_compile.py`)
never does — it passes `philox_seed=None`, which short-circuits `u64_scalar`
before the `import torch` line runs.

## `flyc_polyfill.py` (Task 0b — authored, not vendored)

`flyc_polyfill.py` is hand-written, not copied from FlyDSL, so it is exempt
from the rewrite tables above. It ports six helpers that exist only on
`xinyazhang/sdpa-gfx1201-feature` and are not yet upstream — verified with
`git diff $(git merge-base HEAD upstream/main) HEAD -- kernels/common/`
against merge-base `5675194f18f0655ce1f979c517f673533963fa93`, giving a
121-line diff across two files:

| function | ports from (branch file) | notes |
|---|---|---|
| `ssel(pred, a, b)` | `kernels/common/utils.py` | |
| `smin(a, b)` | `kernels/common/utils.py` | calls `ssel` |
| `smax(a, b)` | `kernels/common/utils.py` | calls `ssel` |
| `sdiv_rd_pow2(value, divisor)` | `kernels/common/utils.py` | uses `is_pow2`/`pow2_shift`, both already upstream at the merge-base |
| `wmma_f32_16x16x16(a, b, acc, acc_type=None)` | `kernels/common/mma/wmma_ops.py` (new file on the branch) | uses `rocdl.wmma_f32_16x16x16_{f16,bf16}` and `flydsl.expr.utils.arith._to_raw`, both upstream |
| `vector_elem_type(value)` | `kernels/common/mma/wmma_ops.py` (new file on the branch) | not called directly by the vendored kernel, but `wmma_f32_16x16x16` calls it |

Each entry is a deletion waiting on an upstream merge: `flyc_polyfill.py`
prefers `flydsl.kernels.common`'s own definition when it has the symbol, and
only falls back to its local copy — so as each function lands upstream, the
polyfill silently stops shadowing it, and the module can be trimmed
function-by-function without touching any other file.

## FakeTensor contract additions

None yet. If `python/flyc_compile.py`'s `FakeTensor` needs a new attribute
because `_launch` in `flash_attn_func_gfx1201_aiw.py` reads one unconditionally
(known hazard: the last two FlyDSL commits added an unconditional `Q.device`
read), record the addition here:

- `Q.device` (and by extension every duck-typed tensor's `.device`) — required
  as of the vendored commit; `_launch` passes it to `abi.dropout_args(...,
  Q.device, stream)` unconditionally (`flash_attn_func_gfx1201_aiw.py:2284`),
  even when dropout is disabled.

## Re-sync cost, measured at `93d8d497`

The `971dce48` -> `93d8d497` re-sync was done deliberately as a cost experiment: copy all
five files verbatim first, then re-wire, and see what it takes.

| | |
|---|---|
| upstream commits spanned | 287 (the branch had also been **rebased**, so `<old>..HEAD` is not a usable diff) |
| upstream lines changed in vendored files | 443 across 3 of 5 files |
| **our re-wiring** | **28 lines**: 2 import blocks + 3 lazy-torch insertions |
| polyfill changes needed | none — same 5 symbols, only redistributed between files |
| ATI description changes needed | none |
| emitted hsaco | **byte-identical across all 12 sweep configurations** |

So a large upstream drift cost almost nothing here, and that is a property of the
vendoring strategy rather than luck: the only coupling points are the `gfx1201_standalone`
imports and the module-scope torch imports, both of which are mechanical and both of which
this file enumerates. The kernarg ABI was independently re-verified — 45 launcher / 44
kernel params, `launcher-minus-stream == kernel` names in order, and the description's 29
declared operands still in matching relative order (the other 15 kernel params are
`stride_*`, supplied by `strides='stride_q_*'` wildcards).

Two things that would have made it expensive, neither of which happened: a new
branch-local helper (would need a `flyc_polyfill.py` entry) or a kernarg reorder (would
need the description's operand list re-ordered, and it is order-sensitive). Check both
explicitly on every re-sync — the AST order check is the cheap way.

## Re-sync cost, measured at `caee9257`

The second of the two predicted expensive cases happened: **a kernarg reorder**. FlyDSL
`1b58fb93` gave the forward and all three backward kernels one convention, and `67a3ace0`
dropped `batch_size` from the kernarg.

| | |
|---|---|
| upstream commits spanned | 11 (`f967e90b`..`caee9257`) |
| vendored files changed upstream | 2 of 5 — `flash_attn_func_gfx1201_aiw.py` (68 lines), `fmha_abi_gfx1201.py` (61) |
| our re-wiring | the same 4 lines as always, **plus** the forward kernarg ABI below |
| polyfill changes needed | none — `kernels/common/` is byte-identical across the span |
| ATI description changes needed | **yes** — see the table |

Forward kernarg changes, 44 parameters down to 43:

| | at `93d8d497` | at `caee9257` |
|---|---|---|
| bias tensor | `Bias`, after `L` | `B`, between `V` and `O` |
| logsumexp | `L` | `LSE`, after `O` |
| batch | `batch_size` is a kernarg | dropped; the launcher keeps it for the grid |
| scale | `sm_scale_arg`, last | `sm_scale`, ahead of the strides |
| bias strides | `stride_b0/1/2` | `stride_b_batch/head/seq_q` |

`batch_size` leaving the kernarg is the one with a C++ consequence: the generated context no
longer declares `flyc_batch_size()`, so `modules/flash/csrc/flyc_attn_fwd.cc` lost that member
— but `grid_calculator()` still needs a batch count, since the grid's z extent is
`num_seqlens != 0 ? num_seqlens : batch_size`. It now comes off `FlycVarlenRow` directly
(`modules/flash/csrc/flyc_common.h`).

## Re-sync procedure

1. Copy every file in the "Vendored files" list from
   `<flydsl checkout>/kernels/attention/parity/` into `modules/flash/flyc/`.
2. Reapply the import rewrites (1a) and the torch-laziness edits (1b) above.
   `diff` each file against its upstream original afterwards: the diff must be
   EXACTLY those tables and nothing else.
3. **Check the kernarg order**, which is the expensive failure mode. The
   descriptions in `modules/flash/aot/flyc_*.py` declare it and are
   order-sensitive; a reorder upstream needs them re-ordered here. The cheap
   check is to link the family and dump the launch-argument vector:

   ```python
   from aotriton.codegen.linker import Linker
   k, o, a, f = Linker('modules').link_all_families()
   fl = [x for x in f if x.NAME == 'flyc_bwd_dkdv'][0]
   for la in fl.iter_launch_arguments():
       print(la.kind, la.aname, la.expr)
   ```

   An undeclared kernel parameter asserts; a *misdeclared* one does not, so read
   the names against the `@flyc.kernel` def.
4. **Check `block_n` for dK/dV.** `modules/flash/aot/flyc_bwd_dkdv.py` mirrors
   the builder's `BLOCK_N = ROWS_PER_WAVE * NUM_TEAMS` derivation, because it is
   not a `resolve_knobs` output and the grid needs it. If that expression moves
   upstream, the mirror is wrong and the symptom is a wrong grid, not a build
   failure.
5. Re-run Gate 1 and Gate 3 from `PLAN-PHASE1.md`.
6. Update the commit hash at the top of this file.

**A re-sync needs a CLEAN build to test.** Editing a file in this directory does
not invalidate any `.hsaco`: `v3src/CMakeLists.txt`'s `add_custom_command` for
each kernel image lists only `DEPENDS aotriton_venv_flydsl` (and
`aotriton_venv_triton` for Triton), never the source. An incremental build after
a re-sync therefore regenerates the C++ shim against the new ABI and keeps the
old code objects, which is a silent wrong-kernarg launch rather than an error.

## Open FlyDSL issues, verified against upstream

Checked against `upstream/main` at `11c4174d`, **41 commits ahead of the vendored
`9de9628a`**. All three are still present there; none is fixed by re-syncing.

### 1. Every kernel is compiled twice, and the wrong copy is the one that runs

Two independent code paths attach a `#rocdl.target` to the same `gpu.module`:

| where | target attached |
|---|---|
| `jit_function.py:1495` `create_gpu_module("kernels", targets=backend.gpu_module_targets())` | `#rocdl.target<chip = "gfx1201">` — bare, all defaults |
| `backends/rocm.py:93` `rocdl-attach-target{O=2 abi=600 chip=... wave64=false ...}` | `#rocdl.target<chip = "gfx1201", flags = {no_wave64}>` |

`gpu-module-to-binary` then emits **one code object per attached target**, so
every `gpu.binary` carries two objects for one kernel. Dumped from a real
gfx1201 head-dim-48 compile:

```
object 0   #rocdl.target<chip = "gfx1201">
object 1   #rocdl.target<chip = "gfx1201", flags = {no_wave64}>
```

**The second one never runs.** The `gpu.binary` op carries no offloading
handler, so MLIR defaults to `#gpu.select_object`, and its specification
(`mlir/Dialect/GPU/IR/CompilationAttrs.td:266`) is explicit:

> The first object in a `gpu.binary` operation is selected if no target is
> specified.

So the object that is launched is the BARE one, and the object built with the
backend's actual compile options — `O=2`, `abi=600`, `correct-sqrt`, `daz`,
and `fast`/`unsafe-math` from `compile_hints` — is discarded.

This is worse than the wasted link time it looks like. It is currently harmless
only because the ROCDL defaults for gfx1201 already imply wave32 and the fp-math
hints default off, so the two happen to agree semantically. A description
setting `fast_fp_math` or `unsafe_fp_math` would have those options applied to
object 1 alone and silently dropped from the shipped kernel.

**How we found it.** `flyc_compile.py`'s `_extract_hsaco` asserts every
`gpu.binary` object is byte-identical, which held for 284 of 288 gfx1201
functionals and failed intermittently on head-dim 48. Investigating produced a
second finding worth recording on its own: the two objects have identical wave
size, register footprint (sgpr 104 / vgpr 121), LDS (6784) and instruction count
(2217), differing only in VOPD packing, `s_delay_alu` hints and one
`s_code_end` — and **neither target is stable across runs**, agreeing about half
the time. So the assertion was inadvertently a determinism test for the AMDGPU
backend, and head-dim 48 is a tile where that determinism does not hold.

**Fix:** attach the target once. Either drop `targets=` at
`create_gpu_module` and let the pass own it (preferred — the pass is the one
carrying the real options), or drop the pass and pass the full target at
creation. Either removes the duplicate object, halves link time for every
kernel, and makes the compile options reach the kernel that runs.

### 2. `flyc.compile()` ignores `COMPILE_ONLY`

Its tail calls `_get_func_exe()` (`jit_function.py:1697`), which builds an
ExecutionEngine and needs HIP, so the documented GPU-less compile mode is
unusable through the public API. `JitFunction.__call__` early-returns correctly;
`compile()` should too. Working around this is why `flyc_compile.py` invokes the
`JitFunction` directly rather than calling `compile()`.

### 3. A wrong `ROCM_PATH` surfaces only as `lld invocation failed`

No lld output, no mention of the path. A pre-flight check for
`<toolkit>/llvm/bin/ld.lld` with a real message would have saved the entire
investigation recorded in `python/flyc_bootstrap.py`'s `resolve_rocm_path`.
