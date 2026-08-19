# Phase 2 — flyc as a real AOTriton v3 backend

Phase 1 ended with an artifact: `aotriton.images/amd-gfx120x/flash/flyc_attn_fwd.zip`,
288 entries, built by `ninja`. Nothing dispatches to it. Phase 2 makes `op_attn_fwd`
able to *choose* it at runtime.

**Deliverable.** A metro backend on `op_attn_fwd` that runs the flyc forward kernel
followed by the Triton `debug_simulate_encoded_softmax`, selectable like any other
backend, with `torch.ops` reaching it end to end on gfx1201.

Three reasons that specific pairing is the target, all of them the user's:

1. `flyc_attn_fwd` implements `debug_simulate_encoded_softmax`'s contract precisely, so
   it should just work.
2. If it does not, it is a placeholder that still exercises the plumbing.
3. It tests **inter-op between hsacos produced by two different DSLs** inside one metro —
   which nothing has ever done here.

**A constraint on the whole phase, not just Task 2.** The code generator
(everything under `python/codegen/`, run via `aotriton.generate` at *configure*
time) must not **invoke the FlyDSL builder or compiler**. Only
`python/flyc_compile.py`, invoked by `ninja` at *build* time, may do that. This
is not a style preference — PLAN.md 6.3 already says it ("The generator never
calls it; `aotriton.flyc_compile` does, at build time, in the venv that has
flydsl"). Note what the rule is *not*: the build-directory venv the generator
runs in already has flydsl installed (`v3src/CMakeLists.txt` launches
`aotriton.generate` from `VENV_BIN_PYTHON`/`VIRTUAL_ENV=${VENV_DIR}`), so
*importability* was never the problem and is not something this phase needs to
guarantee or assert on — the generator staying flydsl-free falls out of the
design below as a beneficial side effect, not a requirement it exists to
satisfy. Task 2 exists to give the generator a way to read FlyDSL's tuning
knobs without invoking the builder; Task 3 is where that reading actually
happens. An earlier draft of Task 2 got this backwards and an implementation
followed it faithfully — see Task 2's opening for what happened and how it was
corrected.

---

## 0. What Phase 1 already put in place

Facts, not plans. Each was measured.

| | |
|---|---|
| `Fly.compile` | 288 rows, 7 `;`-separated fields, `gfx1201` only, no duplicate outputs |
| flyc hsacos | 288 built by `aotriton_v3_flyc_compile` (an `ALL` target) |
| the zip | `aotriton.images/amd-gfx120x/flash/flyc_attn_fwd.zip`, 288 `STORED` entries |
| sidecar JSON | `compile_status`, `kernel_name`, `arch`, `num_warps`, `warp_size`, `shared`, `block_m`, `block_size` |
| `ir/flyc/kdesc.py` | `KernelDescription` — `CODEGEN_MODULE='flyc'`, `FILE_PFX='flyc'`, `ENUM_PREFIX='kFlyc_'`, `is_tunable=False`, empty `perf_cfields`/`func_cfields` |
| `ir/flyc/ksignature.py` | `KernelSignature` — `perf_section`/`copt_section` both `''`; perf fields **and** the `#P` name section deferred to Phase 2 by PLAN.md 6.2, see Task 2 |
| `ContextHelper` | stored on the spec, **read by nothing** (`ir/context_helper.py` says so explicitly) |
| discovery | `aot.flyc_kernels` + `parser.visit_flyc`, deliberately NOT `@ati.backend` |
| kernarg ABI | 45 launcher / 44 kernel params; launcher-minus-`stream` == kernel names in order |
| the description | 29 declared operands: 5 plain, 19 `wires_to=` renames, **5 `wires_to=ati.context_helper`** |

The 15 kernel params not in those 29 are `stride_*`, supplied by `strides='stride_q_*'`
wildcards on `@ati.tensor`.

---

## 1. The three backend shapes, side by side

Phase 2 is mostly "be Triton, minus autotune". This table is the design in miniature.

| | **triton** | **affine** | **flyc** (new) |
|---|---|---|---|
| generated files | `shim.<k>.{h,cc}` | `affine.<k>.{h,cc}` | `flyc.<k>.{h,cc}` |
| generator | `codegen/kernel.py` | `codegen/slim_affine.py` | `codegen/flyc.py` |
| per-functional tune file | `codegen/autotune.py` | none | `codegen/flytune.py` |
| has an hsaco to dispatch | yes | no (`.co`, hand-launched) | **yes** |
| `launch()` | generated, `kernel_on_device->invoke` | **hand-written** in csrc | generated, same `invoke` |
| `lookup_optimal()` | generated, LUT-driven | generated, calls hand-written `check_inputs_are_supported` | generated, degenerate LUT |
| perf/copt vocabulary | psel/copt grid + C struct | none | **schemaless `#P` string, no struct** (Task 2) |
| kernarg source | operator params struct | opaque cookie struct | params struct **+ context helpers** |

The last row is the one genuinely new thing. Triton's kernarg list *is* the operator's
operand list under apparel names; flyc's is a different list that only partly overlaps,
and five of its entries are host-computed.

---

## 2. Task 1 — `codegen/flyc.py`, the shim generator

`FlycShimGenerator(InterfaceGenerator)`, `PFX = 'flyc'`, modelled on
`codegen/kernel.py` (140 lines; this will be similar or smaller).

```python
class FlycShimGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('flyc.h')
    SOURCE_TEMPLATE = get_template('flyc.cc')
    PFX = 'flyc'

    def create_sub_generator(self, functional, df, sql):
        if functional.meta_object.is_functional_disabled(functional):
            return None, False
        return FlycTuneCodeGenerator(self._args, functional, df, sql, self._this_repo), True
```

`create_sub_generator` keeps the `@ati.disable` filter — the same predicate Phase 1 Task 5 already
uses to drop non-gfx1201, fp32 and off-ladder head dims. **A disabled functional must
produce no table entry**, or `lookup_optimal` will hand back a null `kernel_on_device`.

`write_shim_header` / `write_shim_source` mirror `kernel.py`'s, minus:

- `perf_fields` — empty at Phase 1's end; **Task 2 fills it**, so wire the template slot
  now rather than omitting it
- everything `AOTRITON_BUILD_FOR_TUNING`-gated, which can be emitted as the disabled arm
  from day one rather than omitted; keeping the `#if` shape means turning tuning on later
  is a template edit, not a generator rewrite

Reused **unchanged** from `InterfaceGenerator`: `codegen_archmod_number_body`,
`codegen_godel_number_body`, `codegen_tune_table_entry_declares`,
`codegen_tune_table_entries`, `codegen_declare_compiled_in_features`,
`codegen_define_compiled_in_features`. These are all functional-space machinery and
flyc's functional space is the operator's, so they work as-is.

**Files.**

| | path |
|---|---|
| NEW | `python/codegen/flyc.py` |
| MOD | `python/codegen/root.py` — instantiate `FlycShimGenerator` in the flyc loop; collect `shims` |

Today `root.py`'s flyc loop calls `fk.gen_functionals(...)` directly and writes
`Fly.compile` rows with no generator object. Task 1 replaces that with a real
`InterfaceGenerator`, exactly as the triton and affine loops do — the `Fly.compile`
emission moves into it (or stays in `root.py` and reads
`fsg.this_repo.get_data('hsaco')`, matching how triton does it; prefer the latter, it
keeps the rule-file writing in one place).

---

## 3. Task 2 — the flyc perf space (the Phase 1 deferral, now due)

**What happened here once, corrected up front.** An earlier draft of this task
said the generator should obtain the knob values by *invoking the FlyDSL
builder at generate time* — its exact words: "The values must come from the
builder's return instead: `flyc_attn_fwd(choices, hints)` already returns
`(built, sidecar)` ... `root.py` currently discards it. Task 2 keeps it and
threads it into the `KernelSignature`." That is a violation of the constraint
stated at the top of this document — the generator must not invoke the FlyDSL
builder — and an implementation followed it faithfully: it produced a
generator that ran 288 FlyDSL builds per configure, one per surviving
functional, discarding the built module every time (and, as an incidental
consequence of running a real build, importing flydsl 288 times too — a
symptom of the real problem, not the problem itself). The design below
(Design B, unchanged) was never the problem; only this one paragraph,
describing how its input reaches the generator, was wrong. The corrected version is the "Where the
values come from" section near the end of this task, and the mechanism that
actually calls anything now lives in Task 3, not here or in `root.py`.

**This is planned work, not a discovered gap.** PLAN.md 6.2 is titled *"Programmatic
tuning moves the perf space, it does not remove it"*, and `ir/flyc/ksignature.py` records
why both sections were left empty: the FlyDSL tuning model was unsettled and
`resolve_knobs` "looks nothing like a psel/copt grid", so Phase 1 declined to guess. 6.2's
instruction was to *"build the structure N-capable from the start and let N be 1 today"*.
PLAN.md also names the precise loose end — `BLOCK_M`/`BLOCK_SIZE` "have to reach the
context some other way ... but it is not designed". Designing it is this task.

**What forces it now.** `grid_calculator()` returns
`(num_head_q, cdiv(max_seqlen_q, BLOCK_M), batch)` with block `(BLOCK_SIZE, 1, 1)`.
`BLOCK_M` varies per functional — measured 256 at hd 64, 128 at hd 32 and hd 128 — so it
is neither a constant nor derivable host-side without forking `resolve_knobs`.

### Two candidate designs

**A — perf as a C struct.** Translate knobs into `perf_cfields`, emit
`kernel_image_perfs[]`, host reads `this->block_m`. What Triton does.

**B — perf as a schemaless string.** No perf struct. Store the perf/copt strings, ship a
dict-like accessor over a `string_view`, and let `ati.context_helper` read what it needs
out of it.

### Measured facts both designs have to live with

| | |
|---|---|
| `FmhaKnobs` shape | 23 fields, **every one `T \| None`**; 4 are `str`-typed (`v_lds_layout`, `sched_strategy`, `fp_mode`, `path_tag`) |
| Triton's psel by contrast | all int/bool (`PERSISTENT_TYPE=0;GRID_CU_MULTIP=2;BLOCK_M=32;…`) — Triton **never had to solve the string or the `None` case** |
| packed string is already emitted | `template/shim.cc:187`, **unconditional**, no `AOTRITON_BUILD_FOR_TUNING` guard |
| and already deduplicated | Triton `attn_fwd`: **63 distinct strings, 4,610 bytes total** |
| perf structs are **not** deduplicated | same kernel: **12,300 `kernel_image_perfs` entries** |
| the runtime already holds the string | `TritonKernel::ksig_psel_` / `ksig_copt_` (`string_view`), set by `delayed_init` from the packed string via `TritonKernelCluster`. Private — needs a one-line public accessor |
| what flyc's C++ *reads* today | **two** of the 23: `block_m`, `block_size`, for `grid_calculator()`. All 23 are still *stored* — the generator has no way to know which two matter |

### The three concerns, judged

**1. No C type for knob fields — real, but narrower than it looks.** It only bites if the
struct must model *all* knobs. It need not: the host reads two ints and never touches
`v_lds_layout` or `path_tag`. What does bite even on the int subset is `| None` —
`flat_work_group_size: int | None` and `shards: None` have no C spelling short of a
sentinel per field, and a sentinel that collides with a legal value is a silent wrong
grid. **Verdict: avoidable under A by restricting the struct to the host subset, at the
cost of the struct no longer being "the knobs".**

**2. Schemaless because arches differ — real, structural, and the strongest objection.**
`perf_cfields` is *one C struct type per `KernelDescription`*, shared across every arch in
`autotune_table[arch][functional]`. Triton gets away with this because its psel schema is
arch-*invariant* — the values differ per arch, the fields do not. Per-arch knob *sets*
cannot be expressed without a union of all arches, where every field is meaningless on
some arch. **Verdict: a type-system mismatch, not a sizing problem. Design A has no clean
answer.**

**3. Runtime bloat — real, and the scaling differs more than the constants.** Strings
dedupe because many images share a schedule; structs do not. Triton's own numbers show the
shape: 4,610 bytes of deduplicated strings against 12,300 perf structs. Design A is
`O(images x fields)`; Design B is `O(distinct schedules)`. At flyc's current
288 functionals x 1 image both are small — but only one stays small once FlyDSL's tuner
emits several images per functional, which is exactly the future 6.2 says to build for.

### Verdict: Design B, used directly

Design B wins on all three concerns, and on a fourth the evaluation surfaced that matters
more than any of them:

**Single source of truth.** `#P` is *already* the canonical identity of an image inside
the `.aks2`. Design A stores `block_m` twice — once in the `#P` string that names the
image, once in a C struct — with nothing enforcing agreement. A disagreement launches a
real kernel with the wrong grid, the worst failure class available here. Under B the
string is the truth and the host reads a projection of it.

And step (b) of the proposal is **already implemented**: the packed string is emitted
unconditionally today and the runtime already carries `ksig_psel_`. Design B adds a parser
and an accessor, not storage.

Two genuine costs, both mitigable:

- **Parse cost on the launch path.** `grid_calculator()` runs per launch; string scanning
  there is wrong. Parse once in `lookup_optimal()` — it already runs exactly once before
  launch, already touches the selected kernel, and the context already has mutable members
  (Task 5 adds more) to cache into.
- **Loss of compile-time name and type checking.** `this->block_m` would be a compile
  error when wrong; `perf().get_int("block_m")` is a runtime miss. Accepted, not
  engineered away — see below.

**Do not generate per-key accessors.** An earlier draft proposed emitting
`int32_t block_m() const { return perf_.get_int("block_m"); }` to recover compile-time
names. Rejected, for a reason that generalises: **the code generator cannot know which
keys matter.** There is no signal in the description saying `block_m` is load-bearing and
`path_tag` is not — that judgement lives in the hand-written C++, which the generator does
not read. So it would have to emit **all 23**, of which two are used, and every future
knob would add another dead accessor. Keeping all 23 fields in the string and none of them
in generated C++ is the simpler contract, and it is the only one the generator can
actually implement.

That also settles the "why not `ati.expr`" comparison, which an earlier draft got
backwards. The root reason `context_helper` was chosen is not that a missing symbol links
loudly — that is a pleasant side effect. It is that **`ati.expr` would need a transpiler**:
a C++ expression embedded in a Python description has to be parsed, validated and rendered
by machinery this project would then own. `context_helper` needs none of that, and leaves
control in hand-written C++. A schemaless accessor read from hand-written C++ sits on the
same side of that line — it adds no transpiler and takes no control away.

So the typo exposure is real and simply accepted, bounded by two things: the hand-written
surface is six functions, and the accessor should **assert on a missing key**, naming it,
rather than returning a zero that becomes a silently wrong grid.

### The accessor: `class Schemaless`

| | path |
|---|---|
| NEW | `include/aotriton/_internal/schemaless.h` |
| NEW | `v3src/schemaless/schemaless.cc` |

**Grammar**, measured from the packed strings actually emitted today:

```
section := pair (';' pair)*
pair    := key '=' value
value   := Python repr — 0 | -1 | True | False | None | transposed | auto
```

Real examples: `PERSISTENT_TYPE=0;GRID_CU_MULTIP=2;BLOCK_M=32;PRE_LOAD_V=True;NUM_XCDS=8`
(Triton, today) and, for flyc, `block_m=256;…;v_lds_layout=transposed;flat_work_group_size=None`.

```cpp
namespace AOTRITON_NS {

// A borrowed, read-only view over a ';'-separated 'k=v' string. Holds no
// storage: the backing text is the compiled-in packed_string, which has static
// storage duration. Never construct from a temporary std::string.
class Schemaless {
public:
  constexpr explicit Schemaless(std::string_view text = {}) noexcept : text_(text) {}

  bool contains(std::string_view key) const noexcept;
  std::optional<std::string_view> find(std::string_view key) const noexcept;

  // Return the parsed value. On a missing key, an unparsable value ("None"),
  // or a range error: return `dflt` if given, else log the key and assert.
  int64_t          get_int (std::string_view key, std::optional<int64_t> dflt = std::nullopt) const;
  bool             get_bool(std::string_view key, std::optional<bool> dflt = std::nullopt) const;
  std::string_view get_str (std::string_view key, std::optional<std::string_view> dflt = std::nullopt) const;

private:
  std::string_view text_;
};

}
```

**Why the parser has to be built in rather than `strtol`/`sscanf`.** This is the security
point and it is concrete, not theoretical: **`std::string_view` is not NUL-terminated**,
and the psel views point *into the middle* of one concatenated `packed_string` array
(`TritonKernelCluster` hands out `packed_string + meta.psel_offset`). Calling
`atoi`/`strtol`/`sscanf` on `.data()` reads until it finds a NUL — which means it reads
into the *next kernel's* psel string. That is an out-of-bounds read that returns a
plausible wrong number instead of crashing, i.e. the worst possible failure here: a
silently wrong launch geometry.

So the parser is `std::from_chars(v.begin(), v.end(), out)` — pointer-bounded, no locale,
no allocation, C++20 is already the project standard. Rules:

- **Bounded**: never reads outside the value's `[begin, end)`.
- **No trailing garbage**: require the returned `ptr == v.end()`, so `256abc` is a parse
  failure rather than `256`.
- **Overflow is a failure**: `from_chars` reports `errc::result_out_of_range`; treat it as
  a miss, do not truncate.
- **Booleans are exact**: only `True` and `False` (Python repr, as measured). Not `1`/`0`,
  not case-insensitive — strictness costs nothing and a typo should fail loudly.
- **`None` is a parse failure, by design.** Every one of the 23 `FmhaKnobs` fields is
  declared `T | None` — verified, not sampled — so *any* key can render as `None`, and
  three do at hd 64 (`flat_work_group_size`, `shards`, `sched_strategy`). A knob being
  absent is normal, not exceptional. That is what `dflt` is for, and why the argument is
  load-bearing rather than cosmetic.
- **No allocation, `noexcept` lookups**: 23 keys scanned linearly beats a map here, and
  the result is cached once in `lookup_optimal()` anyway.
- **Duplicate keys: first wins**, documented. A generator that emits a duplicate has a bug;
  the accessor should be deterministic about it rather than order-dependent.

**Construction site.** `lookup_optimal()` sets `perf_ = Schemaless(kernel_on_device->psel())`
once, after the image is chosen and before any launch — so `grid_calculator()` never
parses on the launch path.

**Build wiring — a real gotcha.** `v3src/CMakeLists.txt:450` uses
`aux_source_directory(. CC_FILES)`, which is **not recursive**. A new `v3src/schemaless/`
directory is silently ignored, and the failure surfaces as a link error for
`Schemaless::get_int` much later. Add the directory explicitly alongside `CC_FILES` in
`add_library(aotriton_v2 ...)`.

**Unit tests** (no GPU, so they belong in the ordinary suite): missing key; `None`;
trailing garbage; overflow past `int64_t`; empty section; a key that is a prefix of
another (`block_m` vs `block_mask`) — the scan must match on the full key, not a prefix;
duplicate keys; and a view deliberately built mid-`packed_string` to prove no read runs
past its `\0`.

### `#P` content: the full knob set, not the host subset

Correcting an earlier draft of this task, which said perf fields and `#P` "move together"
as the same two values. They are related, not identical:

- **`#P` must make an image unique within its functional.** Two images differing only in
  `waves_per_eu` must not collide, so `#P` carries the whole distinguishing knob set
  (~23 `k=v` pairs, ~250 chars, deduplicating to a handful of distinct strings).
- **The host reads a projection** — two of those keys today.

Design B makes that asymmetry free: the projection is a lookup, not a second copy. Under
Design A it would have been a second, hand-maintained subset.

### Mechanics

- `ir/flyc/kdesc.py` keeps `perf_cfields = []` — Phase 1's value is already right
- `ir/flyc/ksignature.py`'s `perf_section` renders the knob dict as `k=v;k=v`;
  `copt_section` stays `''`
- `codegen_kernel_image_perfs` emits an empty list and `codegen_perf_assignment` emits
  nothing — both reused unchanged, they simply have nothing to say
- NEW runtime: `class Schemaless` (see above), plus a public `psel()` on `TritonKernel`
- `grid_calculator()` reads `perf().get_int("block_m")`; no generated accessors
- `sidecar` — the dict `perf_section` renders — is **not** produced here. It comes from
  Task 3's `FlycTuneCodeGenerator` calling the description function once per functional
  and keeping only the second element of its return. See "Where the values come from"
  below for why that call is safe, and Task 3 for where it happens

**Where the values come from — the one real trap.** The sidecar `<hsaco>.json` has both
(Phase 1 Task 7), and `autotune.py:49-58` has precedent for reading it. **Do not.**
Generation runs at *configure* time; the sidecar appears at *build* time. Reading the file
would work on a second configure and mysteriously fail on a clean one.

**The values come from calling the description function — but never the builder it
returns.** `flyc_attn_fwd(choices, hints)` returns `(build, sidecar)` (commit `e3d2b370`):
`sidecar` is `asdict(knobs)`, plain JSON-serialisable data, and `sidecar['block_m']` is
exactly the number `grid_calculator()` needs. `build` is a zero-argument closure that
constructs the actual FlyDSL module; the only import of the flydsl-bearing kernel module
(`flash_attn_func_gfx1201_aiw`) lives lexically inside it. `flyc_attn_fwd` itself, at call
level, imports only `fmha_tuning_gfx1201` — a module that imports nothing but
`dataclasses`. So:

- **Task 3's generator** (`FlycTuneCodeGenerator`) calls `flyc_attn_fwd(choices, hints)`
  once per functional, keeps `sidecar`, and **never calls `build`** — that is the rule this
  task exists to satisfy. A pleasant side effect, not the goal itself: because `build`'s
  body is where the flydsl-bearing import lives, and the generator never runs that body,
  flydsl happens not to be imported by this call chain either — the generator's venv is
  not required to lack flydsl, and nothing here asserts that it does.
- **`python/flyc_compile.py`**, run by `ninja` at build time, calls the same
  `flyc_attn_fwd(choices, hints)`, then calls `build()` to obtain the real module and
  drives it to an hsaco.

This is what the earlier, wrong draft of this paragraph got backwards. It is quoted in
full at the top of this task; the short version is that it treated the first element of
the return as an already-built module (`built`) and told `root.py` to keep and thread it,
which is exactly invoking the builder at generate time. The fix was not a new mechanism —
Design B, `#P`, and `class Schemaless` below are all unchanged — it was moving one import
from call level into the closure in `modules/flash/aot/flyc_attn_fwd.py`, so that a caller
which does not invoke the callable has no path to flydsl at all — a structural guarantee
about *this one import*, not a claim that the generator's environment is or must be
flydsl-free in general.

**Two build-system changes existed only to make the wrong version work, and are being
reverted with it:**

- **A configure-time flydsl install in the top-level `CMakeLists.txt`.** Invoking the
  builder from inside the generator meant flydsl had to be importable by the time
  `aotriton.generate` ran at *configure* time, so the `flydsl-compiler.txt` pip install
  was pulled out of the build-time `aotriton_venv_flydsl` custom target (realised only
  when `ninja` later builds it) and run as a configure-time `execute_process` instead. The
  generator no longer invokes the builder, so it has no need for flydsl to be ready that
  early; the install goes back to being a build-time target, realised when something
  that actually needs it depends on it (the `Fly.compile` loop in `v3src/CMakeLists.txt`,
  which needs flydsl at build time regardless).
- **`AOTRITON_FLYDSL_KERNEL_ROOT` passed into the generator's own `cmake -E env` block**
  (the one wrapping `-m aotriton.generate`). This existed so that `flyc_bootstrap.setup()`
  — called *from inside the generator process*, to prepare the environment for building a
  real FlyDSL module — could find the pinned FlyDSL source checkout. The generator no
  longer calls `flyc_bootstrap.setup()` (there is nothing to prepare when nothing invokes
  the builder), so the variable has no reader there and is dropped from that block. It
  still belongs, unchanged, in `v3src/CMakeLists.txt`'s `Fly.compile` loop, where it is
  passed to `aotriton.flyc_compile` at build time (see `## 0` and Task 6 in
  PLAN-PHASE1.md) — that use was always correct and is not affected by any of this.

**What the generator still needs is a path, not flydsl, and that is fine.**
`flyc_attn_fwd`'s call-level import — `from fmha_tuning_gfx1201 import ...` — is a bare
import that only resolves if `modules/flash/flyc/` (the vendored kernel directory) is on
`sys.path`. `python/flyc_compile.py` already does exactly this for itself, from data it
already has: `kernel_dir = str(Path(node.module_path).parent)`, then
`sys.path.insert(0, kernel_dir)`. Task 3's generator does the same thing from the same
kind of data — `kdesc.MODULE_PATH.parent`, a path the linker resolves at parse time with
no import involved (`decorators/flyc.py`'s `FlycKernelSpec`). No CMake variable is needed
for this at all: unlike the FlyDSL source checkout (an external pin, hence
`AOTRITON_FLYDSL_KERNEL_ROOT`), the vendored kernel directory is a path inside this repo
that the linker already knows.

**Why a deferred closure, not just "don't call build yet" as a house rule.** The shape
matters beyond today's single-image case, because it is also what the eventual autotune
extension needs. Today `resolve_knobs` ignores `hints`, so one `(choices, hints)` call
produces exactly one `(build, sidecar)` pair. Once FlyDSL's schedule starts reading
`hints` (a seqlen-dependent schedule, PLAN.md 6.9.2), the same call can legitimately
produce **several** `(knobs, build)` pairs for one functional — several candidate
schedules, each with its own knob set and its own deferred builder. On the codegen side
that is N `sidecar`s instead of one, i.e. N `#P` strings registered against the same
functional's `.aks2` — exactly the N-capable layout PLAN.md 6.2 already asked for, and
what Design B is priced for (`#P` deduplicates identical schedules; a struct-per-image
design would not, see the measured facts above). On the compile side, `flyc_compile.py`
calls the one `build` it was asked for, selected by whatever key distinguishes the N
pairs. Nothing about *who is allowed to call what* changes: the generator still reads N
sidecars and calls zero builds; the compile driver still calls exactly one build.

Rejected alternatives: a separate compiled-in table keyed by godel number (duplicates the
perf mechanism for one case); recomputing `resolve_knobs` in C++ (forks knob logic into a
second language, and the description explicitly warns against re-deriving what it already
fixed).

**Files.**

| | path |
|---|---|
| MOD | `python/template_instantiation/ir/flyc/ksignature.py` — `perf_section` renders the knob dict passed in as `sidecar` |
| — | `python/codegen/root.py` — **no change**; `root.py` never called the builder and does not gain the sidecar either — Task 3's `FlycTuneCodeGenerator` does (see below) |
| NEW | `include/aotriton/_internal/schemaless.h` |
| NEW | `v3src/schemaless/schemaless.cc` |
| MOD | `v3src/CMakeLists.txt` — add `schemaless/` to the library sources (`aux_source_directory` is not recursive) |
| MOD | `include/aotriton/_internal/triton_kernel.h` — public `psel()` accessor |
| — | `python/template_instantiation/ir/flyc/kdesc.py` — `perf_cfields` stays `[]` (**unaffected** by this task); Task 3 adds `builder_fn`/`hints()` to the same file for its own, unrelated reason |
| — | `python/codegen/autotune.py` — **no change** |

**Gate.** Three checks:

- the rebuilt `flyc_attn_fwd.zip` still has exactly the same **288 ZIP entry names** as
  Phase 1's — perf must not leak into the functional layer
- every `.aks2` internal entry name carries a **non-empty `#P`**, and within any one
  `.aks2` the entry names are unique
- `perf().get_int("block_m")` returns the value the description's `sidecar` reported for that
  functional, checked at all three head dims (Phase 1 measured 256 at hd 64, 128 at hd 32
  and hd 128), and all 23 knobs are present in `#P`. This proves the string round-trips

---

## 4. Task 3 — `codegen/flytune.py`, the perf-space generator

`FlycTuneCodeGenerator(BaseTuneCodeGenerator)`, one instance per surviving functional,
constructed by Task 1's `FlycShimGenerator.create_sub_generator`. It emits the `.cc` table
entry Task 1's shim includes — the flyc analogue of `autotune.py`'s per-functional table
entry, degenerate compared to Triton's because flyc has no LUT, no binning, and (Phase 2)
exactly one hsaco candidate per functional (`## 1`'s side-by-side table).

**This is the one class that calls the description function, and it is the class that has
to honor the constraint stated at the top of this document and repeated in Task 2.** Its
constructor:

```python
choices = {name: tc.triton_compile_signature for name, tc in f.resolved.items()}
hints = kdesc.hints()                        # the @ati.flyc.hints dataclass's defaults;
                                              # no CLI override exists at generate time
kernel_dir = str(kdesc.MODULE_PATH.parent)
if kernel_dir not in sys.path:
    sys.path.insert(0, kernel_dir)           # same derivation flyc_compile.py uses on itself
build, sidecar = kdesc.builder_fn(choices, hints)   # NEVER call build()
self._sig = KernelSignature(f, sidecar=sidecar)
```

`choices` and `hints` are built the same way `python/flyc_compile.py`'s `do_compile` builds
them, because they must resolve to the identical knob set: the whole point of Design B is
that this generate-time `#P` string agrees with what `flyc_compile.py` resolves
independently at build time, and Task 2's Gate checks exactly that agreement.
`kdesc.MODULE_PATH` is already resolved by the linker at parse time
(`decorators/flyc.py`'s `FlycKernelSpec`), with no import involved, so finding it costs
nothing and needs no new CMake variable.

**Do not call `flyc_bootstrap.setup()` here.** That function resolves `ROCM_PATH`, stubs
`torch`, and points `sys.path` at the pinned FlyDSL *source checkout* — all of it exists
solely to make `flydsl` importable. This constructor never imports flydsl (it calls
`builder_fn`, never the `build` it returns), so none of that applies. Calling `setup()`
anyway would silently re-introduce the same class of mistake Task 2 corrects, just moved
one file over — and would also drag `AOTRITON_FLYDSL_KERNEL_ROOT` back into being something
the generator needs, which Task 2 explicitly removes.

The rest of `generate()` — `write_flytune_src`, `codegen_compact_kernels` (registers
`KernelSignature.perf_section`/`copt_section` into the per-kernel packed-string registry
and the blake2b-hashed image entry), `codegen_deduplicated_pp_args_function_index` (builds
the `pp_args` body from Task 5's `iter_launch_arguments`/`iter_context_helpers`) — touches
no flydsl-adjacent code and is unaffected by any of the above.

**Files.**

| | path |
|---|---|
| NEW | `python/codegen/flytune.py` |
| MOD | `python/codegen/linker.py` — `_build_flycs` threads the description function itself (`builder_fn`) and its `@ati.flyc.hints` class (`hints_cls`) from `FlycDecl` onto `KernelDescription`, alongside the `tensors`/`scalars` specs Task 5 also needs there |
| MOD | `python/template_instantiation/ir/flyc/kdesc.py` — stores `builder_fn`/`hints_cls`; `hints()` returns an instance of the latter |
| — | `python/codegen/basetune.py` — no change expected; confirm `_df=None` is tolerated |

---

## 5. Task 4 — templates

| | path | from |
|---|---|---|
| NEW | `python/codegen/template/flyc.h` | copy of `shim.h`, drop perf struct + tuning members |
| NEW | `python/codegen/template/flyc.cc` | copy of `shim.cc`, see below |
| NEW | `python/codegen/template/flytune_table_entry.cc` | copy of `autotune_table_entry.cc` |

Three deltas in `flyc.cc` versus `shim.cc`, all in `launch()`:

1. **No `TritonAuxiliaryArguments`.** Triton's `pp_args` appends
   `CAST(&aux.global_scratch)` and `CAST(&aux.profile_scratch)`
   (`autotune.py:207-208`) because Triton kernels take two trailing scratch pointers.
   The flyc kernel takes 44 arguments and none of them is scratch. The `PP_FUNC` typedef
   loses the `aux` parameter.
2. **`invoke()` is reused as-is.** Checked: `TritonKernel::invoke`
   (`include/aotriton/_internal/triton_kernel.h:69`) takes
   `(kernel_name, flatzip_path, aks2_entry, func_name, arch_name, grid, args, stream)` —
   nothing in that signature is Triton-specific, and the **block size comes from the aks2
   directory entry** (`block_threads = num_warps * warp_size`), not from the caller. This
   is why Task 7 of Phase 1 had to write `num_warps` into the sidecar; that fix is what
   makes flyc images loadable by this path.
3. The `constexpr std::string_view triton_kernel_name` local should be renamed; it is
   passed to `invoke` purely for logging.

---

## 6. Task 5 — consuming `ContextHelper` in `pp_args`

The heart of the design. `ir/context_helper.py` already specifies the semantics; Phase 2
implements them.

Triton builds its kernarg vector from `kdesc.iter_launch_arguments()`, each yielding a
`LaunchArg(aname, kind, expr)` where `expr` is a rendered C++ expression like
`params.Q->kparam_data_ptr()`. flyc needs the same iterator with one extra case:

| `wires_to=` | count | emitted expression |
|---|---|---|
| absent (name == operand) | 5 | `params.<name>…` as today |
| a plain operand name | 19 | `params.<apparel>…` as today |
| `ati.context_helper('f')` | **5** | `CAST(&<tmp>)` where `<tmp>` holds `this->f()` |

The five are `flyc_varlen_bits`, `flyc_batch_size`, `flyc_num_seqlens`,
`flyc_idropout_p`, `flyc_dropout_scale`.

**A context helper returns a value, and the kernarg vector holds pointers.** Triton's
scalars are `CAST(&params.<field>)` — the address of a field that outlives the call. A
helper's return value has no such home. Two options:

- **(a) locals in `pp_args`, and pass the vector by value.** `pp_args` currently returns
  `std::vector<void*>` by value, and `launch()` passes it to `invoke` before the locals
  die — but the *pointed-to* values would die at `pp_args` return. **Unsafe. Rejected.**
- **(b) mutable scratch members on the context.** Add one `mutable` field per helper to
  the generated context struct, have `pp_args` assign then take the address. Lifetime is
  the context's, which outlives `launch()`. **Recommended.**

Option (b) also gives a natural place to cache: helpers are called once per launch, not
once per use. Make `pp_args` take the context (`const FlycAttnFwdContext&`) rather than
just `params`, since it now needs both.

Where the iterator lives: `ir/flyc/kdesc.py` needs its own
`iter_launch_arguments()`. It cannot inherit Triton's — that one is built from
`self._axes_all` and Triton's apparel machinery. flyc's is built from the description's
declared operand list, in declaration order, which Phase 1 already verified matches the
kernel's parameter order.

**Files.**

| | path |
|---|---|
| MOD | `python/template_instantiation/ir/flyc/kdesc.py` — `iter_launch_arguments`, `list_functional_params`, context-helper declarations |
| MOD | `python/codegen/flyc.py` — `codegen_kernel_arguments` (flyc flavour of `kernel.py:125`) |
| MOD | `python/codegen/template/flyc.h` — declare the 5 helpers + their scratch members |

---

## 7. Task 6 — the hand-written C++ (`modules/flash/csrc/flyc_attn_fwd.cc`)

Generated header declares, author implements. Exactly the split
`AttnFwdContext::grid_calculator()` already uses (`modules/flash/csrc/attn_fwd.cc`), and
`AiterFmhaV3FwdContext::check_inputs_are_supported()` uses for affine.

```cpp
namespace AOTRITON_NS::v3::flash {

int32_t FlycAttnFwdContext::flyc_varlen_bits()   const;  // packs the varlen encoding
int32_t FlycAttnFwdContext::flyc_batch_size()    const;  // params->Q->size(0)
int32_t FlycAttnFwdContext::flyc_num_seqlens()   const;  // packed-sequence count
float   FlycAttnFwdContext::flyc_idropout_p()    const;
float   FlycAttnFwdContext::flyc_dropout_scale() const;

dim3    FlycAttnFwdContext::grid_calculator()    const;
}
```

Return types are **not** declared twice: each comes from the `@ati.scalar` type on the
same description line (`'i32'` → `int32_t`), which is why the description does not repeat
them.

Six functions is the whole hand-written surface. Two of them are one-liners
(`flyc_batch_size`, `flyc_dropout_scale`); the description's own comment accepts that cost
in exchange for one mechanism in one file.

**The `num_seqlens`/`batch_size` pair carries a known hazard.** Phase 1 recorded a
follow-up that was never closed: the mapping must be checked against `flyc_varlen_bits`
for the `< 0` padded case. The three-way `Num_seqlens` encoding is exactly the kind of
thing a C++ expression in the description could not have documented, which is why these
are functions. **Write that check as a comment at the point of decoding, and add a unit
test** — this is the single most likely source of a silent wrong answer in Phase 2.

**Files.**

| | path |
|---|---|
| NEW | `modules/flash/csrc/flyc_attn_fwd.cc` |

No CMake change: `v3src/CMakeLists.txt` globs `modules/<family>/csrc/**.cc` with
`CONFIGURE_DEPENDS`.

---

## 8. Task 7 — metro wiring

Two small parser changes plus a description.

`_node_kind` (`parser.py:149`) gains a `FlycDecl` branch returning `'flyc'`, so a flyc def
can appear as a backend ref. The existing `visit_flyc` was written for the
`aot.flyc_kernels` list and takes a bare def; the backend path passes a `Backend` record
(`b.index`, `b.obj`, `b.name`) and must return `(b.index, 'flyc', name)`. Keep both entry
points during the transition: `functionals_of=` still resolves the functional space, and
`flyc_kernels` can stay until the backend path is proven, then be deleted in one commit.

The metro plan must accept a flyc step. `_iter_plan_subkernels` resolves each call by name
against `self.aot`, and `_record_kernel` assumes a Triton `KernelSpec` — that assumption
needs relaxing so a step can be a flyc def.

Then, in `modules/flash/aot/__init__.py`:

```python
@ati.start
@ati.metro_kernel
def metro_fwd_flyc(params):
    flyc_attn_fwd(params)
    if params.encoded_softmax.data_ptr() != 0:
        debug_simulate_encoded_softmax(params)
```

and on the operator, a third backend below the existing two:

```python
@ati.backend(2, metro_fwd_flyc, 'flyc')
@ati.backend(1, aiter_fmha_v3_fwd, 'aiter')
@ati.backend(0, metro_fwd, 'triton')
```

The generated launcher will be the same shape as `launcher_for_kMetro_Triton`
(`iface.op_attn_fwd.cc:162`): construct both contexts, `lookup_optimal` both, then
`launch` both in order. **That is the inter-op test** — one metro, two hsacos, two DSLs,
one stream, and `debug_simulate_encoded_softmax`'s `launch_condition` already carries the
`encoded_softmax != nullptr` guard.

**Files.**

| | path |
|---|---|
| MOD | `python/codegen/parser.py` — `_node_kind`, `visit_flyc` as a backend visitor, metro step relaxation |
| MOD | `python/codegen/linker.py` — build flyc kdescs for backend refs, not only from `flyc_kernels` |
| MOD | `modules/flash/aot/__init__.py` — `metro_fwd_flyc`, `@ati.backend(2, ...)` |

---

## 9. Task 8 — operator-side fallout

Adding a backend widens things that are sized by backend count:

- `BackendEnum` gains `kMetro_Flyc`; `launcher_table[BackendEnum::Max]` grows a row
- `optune_table[][576]` gains entries — the operator's own tuning selects *which backend*,
  and with no operator-level tuning data for flyc the default stays `kMetro_Triton`
- the operator params struct is a **union over backends**; flyc contributes
  `func_cfields = []` today, so it should widen nothing. **Verify rather than assume** —
  if flyc ever needs an operand the Triton path lacks, it lands here

`ENUM_PREFIX = 'kFlyc_'` is already set on `ir/flyc/kdesc.py`.

---

## Gates

**Gate A — shim generates.** Configure with `-DAOTRITON_DEBUG_SKIP_TRITON_KERNELS=ON`.
`v3src/flash/flyc.flyc_attn_fwd.{h,cc}` exist; `flytune.flyc_attn_fwd/` holds one `.cc`
per surviving functional; the count equals `wc -l Fly.compile`.

**Gate B — it compiles.** `ninja` builds `libaotriton_v2.so` with the flyc shim and
`csrc/flyc_attn_fwd.cc` linked. A missing context helper is a link error naming the
symbol — a useful property, though not the reason `context_helper` exists (that reason is
avoiding a transpiler; see Task 2).

**Gate C — the kernarg vector is right.** Unit-test `pp_args` against a known
`OpAttnFwdParams`: 44 entries, in kernel order, with the 5 helper slots holding the
helper results. This is the one place a silent wrong answer is cheap to produce and
expensive to find, because the kernel will happily run on a misordered kernarg buffer.

**Gate D — dispatch.** With Triton kernels skipped, `op_attn_fwd` on gfx1201 selects the
flyc backend and returns `hipSuccess`. Log at `LOG_DEBUG` and confirm
`flyc_attn_fwd lookup_optimal: kernel_index = 0`.

**Gate E — inter-op, the actual goal.** A full build (Triton kernels included), then
`torch.ops` SDPA forward with `encoded_softmax` non-null, on the flyc backend. Both
hsacos launch on one stream. Compare against the Triton backend's output.

Task 2 carries its own gate inline (hsaco entry names must not move); it is a
generate-time check and belongs with the task rather than in this sequence.

Gates A–C need no GPU. D and E do, and this machine now has four gfx1201.

---

## Open questions, honestly

1. **`num_seqlens` / `batch_size` for the `< 0` padded case.** Carried over unclosed from
   Phase 1. Needs checking against `flyc_varlen_bits` before Gate E is meaningful.
2. **Does `debug_simulate_encoded_softmax` read anything the flyc forward does not
   write?** Goal (a) asserts the contract matches. Unverified here. If it does not, goals
   (b) and (c) still hold — the metro is then a plumbing test with a wrong number in
   `encoded_softmax`.
3. **hd 48 nondeterminism.** 4 of 288 images can be missing after a single build pass
   (`c98f0084`). Harmless in Phase 1 because nothing dispatched; in Phase 2 a missing
   image is a runtime `hipErrorSharedObjectSymbolNotFound`. Either retry until the zip is
   complete, or make the operator fall back when `lookup_optimal` fails — the latter is
   probably wanted anyway.
4. **Naming.** `flyc.<kernel>.cc` versus reusing `shim.<kernel>.cc`. Chose the former,
   matching `affine.<kernel>.cc`, so `ls v3src/flash/` shows the backend of each file.

---

## Suggested order

**1 → 2 → 4 → 3 → 5 → 6 → 7 → 8.**

Task 2 (perf fields) comes second on purpose: Task 3 emits `kernel_image_perfs[]` from it
and Task 6's `grid_calculator()` reads it, so deferring it a second time would mean
touching both again. Templates (4) before flytune (3) only because the template fixes the
slot names the generator fills.

Tasks 1–5 are generator work, gated by A–C, and need no GPU. Tasks 6–8 are integration and
need the hardware, which this machine now has.

The riskiest step is **Task 5**. Everything before it copies an existing shape; everything
after it depends on the kernarg vector being exactly right, and a misordered one runs
happily and returns wrong numbers.
