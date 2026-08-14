# Provenance and re-sync instructions

## Source

```
repo:   git@github.com:xinyazhang/FlyDSL.git
branch: xinyazhang/sdpa-gfx1201-feature
commit: 971dce489ee5d4eed6938675549d7d7a3143ce4f
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
flash_attn_func_gfx1201_aiw.py
fmha_abi_gfx1201.py
fmha_common_gfx1201.py
fmha_tuning_gfx1201.py
philox.py
```

Copied straight from `kernels/attention/parity/<basename>` to
`modules/flash/flyc/<basename>` with no renaming.

**Not vendored, and must never be**: `kernels/common/*` (`buffer_ops.py`,
`mem_ops.py`, `kernels_common.py`, `layout_utils.py`, `utils.py`,
`mma/wmma_ops.py`). These are FlyDSL's shared helpers, not the flash kernel's;
copying them here would fork six files that have nothing to do with attention.
They are resolved via `sys.path` (see "Task 0 interim" below), and the day the
flydsl wheel ships them under `flydsl.kernels.common`, only `flyc_bootstrap.py`
needs to change — not any file in this directory.

Also not vendored: `gfx1201_standalone.py` (existed only to put the FlyDSL repo
root on `sys.path`; replaced by the interim described below),
`dropout_mask_gfx1201.py`, the three bwd kernels
(`fmha_bwd_{dkdv,dq,fuse}_gfx1201_{kernel,interface}.py`),
`flash_attn_func_gfx1201_interface.py` (torch-only), `tooling/`, and every
`test_*.py`.

## Task 0 interim — how `kernels.common` resolves today

`kernels/common` is not in the installed flydsl 0.3.1 wheel (verified:
`find_spec('kernels')` is `None` in `/home/xinyazha/.venvs/nogpu`). Until an
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
| `flash_attn_func_gfx1201_aiw.py` | `from gfx1201_standalone import buffer_ops, wmma_ops` | `from kernels.common import buffer_ops`<br>`import flyc_polyfill as wmma_ops` | `from flydsl.kernels.common import buffer_ops`<br>`from flydsl.kernels.common.mma import wmma_ops` |
| `flash_attn_func_gfx1201_aiw.py` | `from gfx1201_standalone import utils as common_utils` | `import flyc_polyfill as common_utils` | `from flydsl.kernels.common import utils as common_utils` |
| `fmha_common_gfx1201.py` | `from gfx1201_standalone import kernels_common` | `from kernels.common import kernels_common` | `from flydsl.kernels.common import kernels_common` |
| `fmha_common_gfx1201.py` | `from gfx1201_standalone import utils as common_utils` | `import flyc_polyfill as common_utils` | `from flydsl.kernels.common import utils as common_utils` |

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

`fmha_abi_gfx1201.py` had two module-scope torch imports. The build venv must
never have torch (`CMakeLists.txt:142`), so both became function-local. Both
uses were already inside functions.

| where | before | after |
|---|---|---|
| module scope (~line 33) | `import torch` | deleted |
| module scope (~line 35) | `from torch import float32 as torch_f32` | deleted |
| `lse_args()` (first statement) | — | `from torch import float32 as torch_f32  # lazy: build venv has no torch (CMakeLists.txt:142)` |
| `u64_scalar()` (immediately before `with torch.cuda.stream(...)`) | — | `import torch  # lazy: only reached when caller passes a plain int seed; the AOT driver passes None` |

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

## Re-sync procedure

1. `cp <flydsl checkout>/kernels/attention/parity/{flash_attn_func_gfx1201_aiw,fmha_abi_gfx1201,fmha_common_gfx1201,fmha_tuning_gfx1201,philox}.py modules/flash/flyc/`
2. Reapply the four import rewrites (1a) and the two torch-laziness edits (1b)
   above.
3. Re-run Gate 1 and Gate 3 from `PLAN-PHASE1.md`.
4. Update the commit hash at the top of this file.
