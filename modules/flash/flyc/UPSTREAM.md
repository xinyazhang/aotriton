# Provenance and re-sync instructions

Two architectures are vendored from the same upstream directory, at **two
different commits on two different branches**. They are tracked separately
throughout this file, because they re-sync independently: a gfx950 re-sync must
not silently move gfx1201's files, and vice versa. The three files they *share*
(`fmha_abi_gfx1201.py`, `fmha_common_gfx1201.py`, `philox.py`) are gfx1201's
by provenance, and pulling them forward for gfx950 moves both arches at once.

## Source

### gfx1201

```
repo:   git@github.com:xinyazhang/FlyDSL.git
branch: xinyazhang/sdpa-gfx1201-feature
commit: caee9257  (was 93d8d497f8e9bbc66106617feb851cf0fb12acd3)
path:   kernels/attention/parity/
```

### gfx950

```
repo:   git@github.com:xinyazhang/FlyDSL.git
branch: xinyazhang/sdpa-gfx950-feature-bwd
commit: 7cd69444  (was 70b2dbc5)
path:   kernels/attention/parity/
```

Get the commit fresh with:

```bash
git -C <flydsl checkout> rev-parse HEAD
git -C <flydsl checkout> branch --show-current
```

## Vendored files (verbatim at the commit above, before the rewrites in "Import
rewrites" below are reapplied)

### gfx1201 — nine files

```
flash_attn_func_gfx1201_aiw.py      forward
fmha_tuning_gfx1201.py              forward tuning policy
fmha_bwd_dkdv_gfx1201_kernel.py     backward dK/dV
fmha_tuning_bwd_dkdv_gfx1201.py     backward dK/dV tuning policy
fmha_bwd_dq_gfx1201_kernel.py       backward dQ/dB
fmha_tuning_bwd_dq_gfx1201.py       backward dQ/dB tuning policy
fmha_abi_gfx1201.py                 shared host ABI       ) also imported
fmha_common_gfx1201.py              shared device helpers ) verbatim by
philox.py                           shared PRNG           ) every gfx950 file
```

### gfx950 — twelve files

```
flash_attn_func_gfx950.py           forward                     (ONE edit -- see 1c)
fmha_tuning_gfx950.py               forward tuning policy
fmha_bwd_dkdv_gfx950.py             backward dK/dV
fmha_bwd_dkdv_m16_gfx950.py         backward dK/dV, MFMA16 body
fmha_tuning_bwd_dkdv_gfx950.py      backward dK/dV tuning policy
fmha_bwd_dq_gfx950.py               backward dQ/dB
fmha_bwd_dq_m16_gfx950.py           backward dQ/dB, MFMA16 body
fmha_tuning_bwd_dq_gfx950.py        backward dQ/dB tuning policy
fmha_traits_gfx950.py               ParityDualwaveTraits + make_traits
fmha_dualwave_gfx950.py             dual-wave device helpers
fmha_wide_gfx950.py                 wide-tile device helpers
fmha_mfma16_gfx950.py               MFMA16 addressing constants
```

**Eleven of the twelve are byte-identical to upstream and must stay that way**
(`diff` each against `git show <commit>:kernels/attention/parity/<basename>` and
expect empty). The twelfth, `flash_attn_func_gfx950.py`, carries the single
deletion recorded in "Vendored edits (1c)" below. There are **zero** import
rewrites for gfx950 — see "Import rewrites (1a)" for why, and what it cost.

**Not vendored for gfx950:** `gfx950_standalone.py` (we author our own — see
below), `kernels/attention/flash_attn_utils.py` (polyfilled — see below), every
`*_interface.py`, every `test_*.py`, `tooling/`, and every `.md`/`.pdf`.

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

gfx950 takes exactly four names from that tree —
`buffer_ops.buffer_load`, `.buffer_store`, `.create_buffer_resource` and
`.get_element_ptr`. All four are present at `v0.3.0`, at `upstream/main` and on
the gfx950 branch, so gfx950 needs **no** new `flyc_polyfill` entry for
`kernels/common` and the rule above is unchanged. (The `v0.3.1` tag itself could
not be resolved in the checkout used to verify this and there was no network to
fetch it; `v0.3.0` is the nearest tag that could be read.)

### The gfx950 kernel-root pin is stricter than gfx1201's

gfx950 needs one more module out of that same source tree:
`kernels/attention/flash_attn_utils.py`, which supplies `DualwaveSwpTraits`
(see the polyfill section). Two consequences, both retired by the same event:

* **The pinned tag is not sufficient for a *correct* gfx950 build.** The gfx950
  branch carries a ~122-line delta in `flash_attn_utils.py` — the DS transpose
  reads moved off inline asm onto ROCDL ops, because `SIInsertWaitcnts` cannot
  see through asm and the result was non-deterministic NaN above head_dim 128.
  Until that lands upstream and `third_party/flydsl-kernel.txt` is bumped, a
  gfx950 build must set `-DAOTRITON_FLYDSL_KERNEL_ROOT=<live FlyDSL checkout>`,
  which the CMake cache variable is documented for.
* **That moves gfx1201 too.** Both arches read one kernel root, and every gfx950
  file imports `fmha_common_gfx1201`. Re-run the gfx1201 Level-0 pass right
  after pointing the root at a live checkout, *before* touching anything gfx950
  — otherwise a gfx1201 regression looks like a gfx950 one.

**Retiring condition for both:** the `flash_attn_utils.py` delta merges upstream
and `third_party/flydsl-kernel.txt` is bumped to a tag containing it. Then
`AOTRITON_FLYDSL_KERNEL_ROOT` goes back to the shallow clone for both arches.

## Vendored edits (1c) — gfx950 only, exactly one

gfx1201 has none of these; its coupling is all in table 1a. gfx950 inverts that:
zero import rewrites, one deletion.

| file | what is deleted | why | retire when |
|---|---|---|---|
| `flash_attn_func_gfx950.py` | the `# Split-K combine.` comment, `COMBINE_BLOCK` / `COMBINE_LANES_PER_ROW` / `COMBINE_ROWS_PER_BLOCK`, and the whole `@flyc.kernel def flash_attn_splitk_combine_kernel` (block 1, 45 lines at `7cd69444`: 917–961; was 38 lines at 888–925 at `70b2dbc5`) **and** the `if const_expr(traits.SPLITK):` block in the launcher that computes `combine_rows` and launches it (block 2, 7 lines: 1137–1143; was 1101–1107) | the file otherwise holds **two** `@flyc.kernel`, and two AOTriton sites locate the kernel by uniqueness (`specs/flyc.py:_flyc_kernel_stub`, `flyc_compile.py:kernel_function_of`). The combine kernel is dead for us: the descriptions pin `num_kv_splits=1`, so `traits.SPLITK` is always false and it is never traced | AOTriton builds a split-K forward, **or** upstream moves the combine kernel to its own module |

Both blocks go, not just the first: leaving the call site would be a `NameError`
at trace time if SPLITK were ever enabled, which is a worse failure than the
honest one.

**Re-derive the line numbers from the AST on every re-sync; do not trust the
ones above.** The deletion is verified by:

```bash
python3 -c "
import ast; t=ast.parse(open('modules/flash/flyc/flash_attn_func_gfx950.py').read())
ks=[n.name for n in ast.walk(t) if isinstance(n,ast.FunctionDef)
    and any(getattr(d.func if isinstance(d,ast.Call) else d,'attr',None)=='kernel'
            for d in n.decorator_list)]
assert ks==['flash_attn_func_gfx950_kernel'], ks"
grep -n 'COMBINE_\|splitk_combine' modules/flash/flyc/flash_attn_func_gfx950.py   # expect nothing
diff <(git -C <flydsl checkout> show <commit>:kernels/attention/parity/flash_attn_func_gfx950.py) \
     modules/flash/flyc/flash_attn_func_gfx950.py                                 # expect ONLY the two blocks
```

This costs the zero-vendored-edit property the `gfx950_standalone.py` design
otherwise achieves, and that is a deliberate trade: in exchange the file has one
`@flyc.kernel` and neither uniqueness assertion can fire at all.

## Import rewrites (1a) — gfx1201 only; gfx950 has **none**

**gfx950 contributes zero rows to this table, by construction.** Rather than
rewrite the eight `from gfx950_standalone import ...` lines the way gfx1201's
four `gfx1201_standalone` lines are rewritten below, we supply our own
`gfx950_standalone.py` — see "Authored files" — so every vendored gfx950 file
keeps its imports verbatim and a re-sync diff for those files is empty. That is
a strictly better outcome than gfx1201 got; if gfx1201 is ever re-vendored,
copy the pattern rather than this table.

The rest of this section is gfx1201's.

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

## Torch-lazy rewrites (1b) — gfx1201 only; gfx950 needs none

No vendored gfx950 file imports torch. The only mention is the *string*
`"torch.bfloat16"` in a host-side dtype check inside `fmha_bwd_dkdv_gfx950._args`,
which the AOT driver never calls. Check this on re-sync — a new module-scope
`import torch` would break the build venv, which must never have torch
(`CMakeLists.txt:142`) — but expect it to stay empty.

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

### The seventh entry: `DualwaveSwpTraits` (gfx950), and why its row is different

| class | ports from | notes |
|---|---|---|
| `DualwaveSwpTraits` | `kernels/attention/flash_attn_utils.py` @ `7cd69444`, lines 1476–1582 | **verbatim, 107 lines**, `@dataclass(frozen=True)`, no bases, 78 annotated scalar fields, one `cache_tag` property with zero flydsl references. sha256 `6750ff4e…` — unchanged from `70b2dbc5`, same lines and same hash |

**Retiring condition — not the usual one.** The other six wait on an upstream
merge. This class is *already* upstream, and byte-identical on `upstream/main`
and the gfx950 branch alike. It is copied because of **where** it lives:
`kernels/attention/flash_attn_utils.py`, whose module scope does
`import flydsl.compiler as flyc`. The generator reaches this class —
`fmha_traits_gfx950.ParityDualwaveTraits` subclasses it, and
`fmha_tuning_*_gfx950.resolve()`'s last step `_checked_against_traits`
*constructs* one to validate a configuration before discarding it — and the
generator must never import flydsl, or `.ci/build-shim.sh` stops being cheap.

So the row retires **when `fmha_traits_gfx950` no longer needs a flydsl-bearing
module for its base class** — i.e. when `DualwaveSwpTraits` moves to a module
that imports no flydsl, or when the generator no longer constructs traits.

**Why the copy is safe.** Do not weaken these; they are the justification, not
decoration:

* All 78 fields are passed **by name** at every construction — only `RETURN_LSE`
  and `XCD_SWIZZLE` carry defaults and `make_traits` passes both explicitly — so
  an upstream add/remove/rename is a loud `TypeError`, never a wrong number. The
  one silent channel a polyfill normally has (a changed default nobody passes)
  is empty here.
* The copy never reaches the compiler. At build time `flyc_bootstrap.setup()`
  has run, `gfx950_standalone` binds `dualwave` to the real module, and the copy
  is used *only* by the generator, whose sole output is the knob dict.
* **The claim is checked, not asserted.** At build time both definitions exist,
  so `gfx950_standalone` calls
  `flyc_polyfill.assert_dualwave_swp_traits_equivalent()`, which compares
  `dataclasses.fields()` — names, types and order. For a 78-field dataclass that
  check is *total*, not a sample. Defaults are deliberately excluded (see above:
  they cannot reach a kernel, so checking them would turn a harmless upstream
  edit into a failed build).

**The check is called from `gfx950_standalone`, not from `flyc_polyfill`'s own
module scope, and that placement is load-bearing.** The gfx1201 kernels alias
`flyc_polyfill` *wholesale* as `common_utils` and `wmma_ops`, so a module-scope
`try: from kernels.attention.flash_attn_utils import DualwaveSwpTraits` there
would import a gfx950 module into every gfx1201 build and run the comparison
against whatever the kernel-root pin happens to hold. That is not hypothetical:
the class is **105 lines at `v0.3.0`** against 107 at the merge-base,
`upstream/main` and the gfx950 branch — so that spelling could fail a gfx1201
build over a class gfx1201 never touches.

**`flyc_polyfill.py` must import with no flydsl installed.** The generator
reaches it through `gfx950_standalone`'s fallback, so every `flydsl` import in
it is function-local and the two `ir.Type` annotations are quoted — an
annotation on a `def` is evaluated when the `def` executes. This is the same
laziness section 1b applies to torch, for the same reason. A new module-scope
`import flydsl…` there breaks the code generator, not just the build.

## Authored files (not vendored, exempt from the rewrite tables)

| file | status | what it is |
|---|---|---|
| `flyc_polyfill.py` | authored | the seven fallbacks above |
| `gfx950_standalone.py` | authored | our implementation of upstream `gfx950_standalone.py`'s *interface* |

Upstream's `gfx950_standalone.py` is a compatibility layer between `parity/`
(not part of the flydsl package) and its parent (which is), and it does that
with `sys.path` surgery relative to `Path(__file__).parents[3]`. Under
`modules/flash/flyc/` that resolves to the **AOTriton repository root**, so
vendoring it verbatim would put the wrong directory on `sys.path`. There is
nothing to fix in it: it is correct where it lives and wrong where we would put
it. Its *interface*, though, is two names — measured across all eight importers
among the vendored files:

```
fmha_traits_gfx950.py        from gfx950_standalone import dualwave
flash_attn_func_gfx950.py    from gfx950_standalone import dualwave
fmha_wide_gfx950.py          from gfx950_standalone import dualwave
fmha_dualwave_gfx950.py      from gfx950_standalone import buffer_ops, dualwave
fmha_bwd_dkdv_gfx950.py      from gfx950_standalone import buffer_ops, dualwave
fmha_bwd_dkdv_m16_gfx950.py  from gfx950_standalone import buffer_ops, dualwave
fmha_bwd_dq_gfx950.py        from gfx950_standalone import buffer_ops, dualwave
fmha_bwd_dq_m16_gfx950.py    from gfx950_standalone import buffer_ops, dualwave
```

`kernels_common`, `layout_utils`, `mem_ops` and `utils` are re-exported upstream
but **taken by nothing**, so ours does not re-export them. Add one only when a
vendored file imports it. Ours does **no** `sys.path` surgery —
`flyc_bootstrap.py` already owns that, and doing it a second time from a
different anchor is how the two would drift.

`dualwave` resolves per environment: the real `kernels.attention.flash_attn_utils`
at build time, `flyc_polyfill` in the generator. `buffer_ops` resolves **lazily**
(PEP 562 module `__getattr__`) and deliberately so: it is device-side only, and
every file that imports it also imports flydsl at module scope, so none is
reachable from the generator. Importing it eagerly would break the generator over
a name the generator never uses; deferring it keeps `kernels.common`'s own
`ImportError` and traceback intact at the point of use.

## FakeTensor contract additions

None yet. If `python/flyc_compile.py`'s `FakeTensor` needs a new attribute
because `_launch` in `flash_attn_func_gfx1201_aiw.py` reads one unconditionally
(known hazard: the last two FlyDSL commits added an unconditional `Q.device`
read), record the addition here:

- `Q.device` (and by extension every duck-typed tensor's `.device`) — required
  as of the vendored commit; `_launch` passes it to `abi.dropout_args(...,
  Q.device, stream)` unconditionally (`flash_attn_func_gfx1201_aiw.py:2284`),
  even when dropout is disabled.

**gfx950 needs none, and `7cd69444` is where that could have changed.** The
pointer rewrite introduced `wire_ptr(t)` in `fmha_dualwave_gfx950.py`, which
reads `t.dtype` (stringified, keyed through `_TORCH_DTYPE_TO_FX`) and
`t.data_ptr()` — and `FakeTensor` has no `.dtype`. It does not matter, because
`wire_ptr` is only reached from `_args`, and `_args` is only reached from
`plan()` / `__call__`, the torch-facing JIT entry points. The AOT driver enters
at `@flyc.jit launch_flash_attn_func_gfx950`, whose six tensor operands are now
`fx.Pointer`, so `flyc_compile.synthesise_args` builds them from the annotation
alone and no descriptor is consulted. Re-check this on re-sync by confirming
`wire_ptr` has not migrated into the `@flyc.jit` launcher body.

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

## Re-sync cost, measured at `7cd69444` (gfx950)

The first gfx950 re-sync, and the one the arch was waiting on: `7cd69444`
*"[Kernel] gfx950: tensor operands become fx.Pointer, and the kernarg loses its
descriptors"*. Before it, `flyc_compile._operand_for` raised on `fx.Tensor` and
the C++ shim had no by-value-descriptor concept, so gfx950 could not be built at
all.

| | |
|---|---|
| upstream commits spanned | 1 (`70b2dbc5`..`7cd69444`) |
| vendored files changed upstream | 4 of 12 — `flash_attn_func_gfx950.py` (94 lines), `fmha_bwd_dkdv_gfx950.py` (80), `fmha_bwd_dq_gfx950.py` (80), `fmha_dualwave_gfx950.py` (+101, the new `wire_ptr`/`wire_view` pair) |
| our re-wiring | **none** — still zero import rewrites; only the 1c deletion, re-derived |
| polyfill changes needed | none — `flash_attn_utils.py` is byte-identical across the span, so `DualwaveSwpTraits` keeps its lines and its `6750ff4e…` |
| ATI description changes needed | **none** — the kernarg order did not move (below) |
| Gate B | both halves re-run and unchanged: generator side gives `fwd block_m=256`, `dkdv block_kv=64`, `dq block_m=64`, `GRID_AXIS_ORDER=0/0/0` at head_dim 128 with `flydsl.*` and `kernels.*` blocked; build side imports all three kernels and the 78-field `DualwaveSwpTraits` equivalence check passes |

**The kernarg order did not move, which is the thing that had to be checked.**
The rewrite touched every tensor operand's annotation, so it is exactly the
shape of change that reorders a kernarg. It did not: the parameter *names* are
unchanged and so are their positions. `fx.Tensor` → `fx.Pointer` in place, six
slots in the forward and nine in each backward.

| | gfx950 | gfx1201 | |
|---|---|---|---|
| forward | 46, or **43** with `Workspace` / `BlockTable` / `block_table_stride` folded away | **43** | identical name-for-name *and* annotation-for-annotation |
| backward dK/dV | **50** | **50** | identical name-for-name |
| backward dQ/dB | **50** | **50** | identical name-for-name |

**Zero `fx.Tensor`** survives in any of the three kernel defs or their
launchers. Two *spellings* remain, both in `flash_attn_func_gfx950.py`:

```python
_WS_ANN = fx.Tensor if _WS_RUNTIME else fx.Constexpr   # _WS_RUNTIME = SPLITK or DEBUG_LAZY_COUNTS
_BT_ANN = fx.Tensor if _BT_RUNTIME else fx.Constexpr   # _BT_RUNTIME = PAGED
```

Both resolve to `fx.Constexpr` for every build AOTriton makes — the descriptions
pin `num_kv_splits=1` and no paging — which is the constexpr fold (item C) and
is what folds 46 down to 43. That fold surviving the rewrite was an explicit
obligation on the FlyDSL side; it did.

**One upstream claim to read carefully.** The new comment above the kernarg says
the forward is 296 bytes and is *not* byte-identical to
`flash_attn_func_aiw_kernel`, which "carries `batch_size` on the wire" and puts
`sm_scale` last. That is measured against the copy of
`flash_attn_func_gfx1201_aiw.py` sitting on the **gfx950 branch**, which is the
pre-`67a3ace0` 44-parameter version. Against the gfx1201 we actually vendor
(`caee9257`, 43 parameters, no `batch_size`, `sm_scale` before the strides) the
two agree exactly. Do not act on the upstream comment without checking which
revision it is comparing to; the two branches carry different `aiw` files.

296 vs the 292 quoted elsewhere for gfx1201 is padding, not a layout difference:
the declared fields total 292 and `sm_scale` (f32) meets the fifteen i64 strides,
so one 4-byte hole is inserted. Both kernels have it.

## Re-sync procedure

**Re-sync one arch at a time.** The two lists come from different branches; the
three shared `*_gfx1201` files belong to gfx1201's list, and pulling them
forward moves both arches at once.

1. Copy every file in the "Vendored files" list for the arch you are syncing
   from `<flydsl checkout>/kernels/attention/parity/` into `modules/flash/flyc/`.
2. Reapply the edits recorded for that arch, then `diff` each file against its
   upstream original: the diff must be EXACTLY the recorded tables and nothing
   else.
   * **gfx1201** — the import rewrites (1a) and the torch-laziness edits (1b).
   * **gfx950** — nothing for eleven of the twelve files (`diff` must be
     **empty**, prove it, do not assume it), and only block 1 + block 2 of the
     combine-kernel deletion (1c) for `flash_attn_func_gfx950.py`. Re-derive
     those two blocks from the AST; the line numbers in 1c are for `70b2dbc5`.
2b. **Re-run Gate B** (both halves) — it is cheap, pure Python, needs no GPU and
   no cmake, and it is what catches a new flydsl import sneaking into the
   generate-time chain:
   * *generator side*, from `modules/flash/flyc/`, with a `sys.meta_path`
     finder that raises `ImportError` for any `flydsl.*` or `kernels.*`:
     `import fmha_tuning_gfx950, fmha_tuning_bwd_dkdv_gfx950,
     fmha_tuning_bwd_dq_gfx950` must succeed, and at `head_dim=128`
     `resolve()` must give `fwd block_m=256`, `dkdv block_kv=64`,
     `dq block_m=64`, `GRID_AXIS_ORDER=0` for all three;
   * *build side*, with flydsl available and `AOTRITON_FLYDSL_KERNEL_ROOT` set:
     `import flash_attn_func_gfx950, fmha_bwd_dkdv_gfx950, fmha_bwd_dq_gfx950`
     must succeed, which is also where the `DualwaveSwpTraits` equivalence
     check runs. A symbol missing from `kernels/common` goes into
     `flyc_polyfill.py` — **never** into a copy of `kernels/common/*`.
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
   failure. **gfx950 has no such mirror** and must not grow one: `block_m`
   (fwd, dQ), `block_kv` (dK/dV) and `GRID_AXIS_ORDER` are all flat resolved
   knob fields there.
5. Re-run Gate 1 and Gate 3 from `PLAN-PHASE1.md`.
6. Update the commit hash at the top of this file — **for the arch you synced**.

**A re-sync will not be picked up by an incremental build.** Editing a kernel
source invalidates no `.hsaco` and no `.aks2`: `v3src/CMakeLists.txt`'s
`add_custom_command` for each image lists only `DEPENDS aotriton_venv_flydsl`,
never the source, and the repack rule hangs off a phony target rather than the
JSON. So an incremental build after a re-sync regenerates the C++ shim against
the new ABI and keeps the old code objects — a silent wrong-kernarg launch
rather than an error.

This is a **build-system limitation, not a flyc one**: Triton kernels behave
the same way, which is why a Triton kernel change is also tested from scratch.
The standing workaround is to delete `CMakeCache.txt` and `<build>/v3src`
before rebuilding — cheaper than a full clean, and enough to force
regeneration. The real fix is to register Triton+ATI and FlyDSL+ATI as CMake
custom languages, at which point dependency tracking becomes CMake's job and
this whole paragraph goes away. Do not add ad-hoc `DEPENDS` entries in the
meantime: they would catch a direct edit and still miss a transitive one (a
helper module the kernel imports), which is the failure mode that is hardest
to notice.

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
