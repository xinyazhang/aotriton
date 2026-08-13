# JIT → AOT: what actually happens between `build_flash_attn_func_aiw_module_primary` and a `.hsaco`

Written to support a design decision: **what belongs inside `def flyc_attn_fwd`, and how much
of the rest the AOT compiler should automate.** Everything below is measured against
flydsl 0.3.1 in `/home/xinyazha/.venvs/nogpu`, gfx1201, no GPU.

---

## 1. The three layers, and which one we actually want

```
build_flash_attn_func_aiw_module_primary(meta, knobs)
│
│   TRACE-TIME SPECIALISATION.  Reads FmhaInputMetadata + FmhaKnobs and closes over
│   ~25 const_expr values (BLOCK_M, BLOCK_DMODEL, CAUSAL_TYPE, V_LDS_LAYOUT,
│   K_PREFETCH_DIST, SHARDS, …).  This is the ONLY place the emitted kernel is decided.
│
├── @flyc.kernel flash_attn_func_aiw_kernel(...)   ← the device kernel; kernarg ABI
├── @flyc.jit    launch_flash_attn_aiw(44 params)  ← LAYER WE WANT
└── returns      _launch(Q, K, V, O, batch, seqlen, …)   ← LAYER WE DO NOT WANT
        │
        │   HOST MARSHALLING.  Exists to turn real torch tensors into the 44 values.
        │   _prep/strides_of, lse_args, resolve_window, varlen_args, _bias_args,
        │   dropout_args, _resolve_scale.  Every output is a RUNTIME argument.
        └── abi.run_compiled(cache, launch_flash_attn_aiw, *args) → flyc.compile(...)
```

The AOT job is to reach `launch_flash_attn_aiw` and compile it. The middle layer is the
whole source of the FakeTensor problem: it wants tensors because its callers have tensors.

**Measured:** every value `_launch` computes is a runtime kernel argument, so none of it
reaches the binary. Driving `launch_flash_attn_aiw` directly with 44 type-correct dummies
produces a **byte-identical** hsaco to the full marshalling path
(`bc6d0fca66a0c2d9f476` both ways).

---

## 2. What the compiler actually requires

`launch_flash_attn_aiw` — 44 parameters, and only their **types** matter:

| annotation | count | AOT value |
|---|---|---|
| `fx.Pointer` | 14 | `flyc.from_c_void_p(fx.Uint8, 0)` |
| `fx.Int32` | 11 | `0` |
| `fx.Int64` | 16 | `0` |
| `fx.Float32` | 2 | `0.0` |
| `fx.Stream` | 1 | `fx.Stream(None)` |

```
Q K V O L Bias seqinfo_q0 seqinfo_q1 seqinfo_k0 seqinfo_k1
batch_size varlen_bits max_seqlen_q max_seqlen_k window_left window_right
philox_seed_ptr philox_offset1 philox_offset2 philox_seed_output philox_offset_output
idropout_p dropout_scale num_head_q num_head_k hdim_qk hdim_vo
stride_{q,k,v,o}_{batch,head,seq} stride_b0 stride_b1 stride_b2 sm_scale_v stream
```

`PointerJitArg.__init__` (`flydsl/compiler/jit_argument.py:592`) takes an element type and an
`int | c_void_p | None`; `__get_ir_types__` derives the IR type from `element_type` alone. No
tensor is involved at any point. `int`/`float` map to `Int32`/`Float32` through
`JitArgumentRegistry`.

**So the arg list is derivable from `inspect.signature(jf.func)` with a 5-entry lookup
table** — no kernel-specific knowledge at all.

---

## 3. What `jf(*args)` does, step by step

`JitFunction.__call__`, `flydsl/compiler/jit_function.py`:

| step | AOT relevance |
|---|---|
| `_ensure_sig()` — resolve signature, freeze `GPUTarget` | needed |
| `_build_full_cache_key(...)` — env + target + per-arg `cache_signature()` | wasted work, harmless |
| `ensure_compile_runtime_pairing_from_env()` | env-only; explicitly avoids constructing a device runtime |
| in-process / disk cache lookup | disabled via `FLYDSL_RUNTIME_ENABLE_CACHE=0` |
| `convert_to_jit_arguments(sig, bound)` → `jit_args`, `dsl_types`, `ir_types` | **the only place args are consumed** |
| build `ir.Module`, `gpu.module` with `backend.gpu_module_targets()` | needed |
| trace `self.func(**named_args)` — runs the `@flyc.jit` body, emits the launch op | needed |
| `MlirCompiler.compile(module, arch=backend.target.arch)` — full pass pipeline incl. `gpu-module-to-binary{format=fatbin}` | **this is the compile** |
| `CompiledArtifact(compiled_module, name, original_ir, …)` | holds both IRs |
| `if env.compile.compile_only: return None` | returns before ExecutionEngine init → no HIP needed |

Then the driver extracts from `jf._last_compiled[1]`:
- `._ir_text` → post-lowering module → walk for `gpu.binary` → `gpu.ObjectAttr(...).object` → the ELF.
  Two objects appear (`#rocdl.target<chip=…>` plus the `no_wave64` one `rocdl-attach-target`
  adds); they are byte-identical.
- `._source_ir` → pre-lowering module → `gpu.func`'s `known_block_size` → BLOCK_SIZE.

`flyc.compile()` is **not** usable: it ends in `artifact._get_func_exe()`, which builds an
ExecutionEngine and needs HIP.

---

## 4. What determines the artifact, and what provably does not

Measured, not assumed:

| input | affects the hsaco? | evidence |
|---|---|---|
| `FmhaInputMetadata` + `FmhaKnobs` | **yes** — this is the whole specialisation | by construction |
| target arch (`ARCH`) | **yes** | |
| tensor shapes / strides / layout | no | 4 shape+layout combos → identical bytes |
| `window=` value | no | `(512,0)` vs `(1024,7)` → identical bytes |
| all 44 argument *values* | no | annotation-derived zeros → identical to the marshalled path |
| `ROCM_PATH` location | no | full venv / 143 MB tree / cross-venv → identical bytes |
| env vars in the tuning module | no | `grep environ|getenv` over the tuning path returns nothing |

The practical consequence: **the AOT compiler cannot get the artifact wrong by choosing bad
argument values.** It can only get it wrong by choosing the wrong `(meta, knobs)`.

---

## 5. The decomposition: who must decide what

Four decisions, and only two are kernel-specific.

| # | decision | kernel-specific? | who |
|---|---|---|---|
| 1 | functional → `(meta, knobs)` | **yes** — FlyDSL vocabulary, ladder rules, `resolve_knobs` vs `plan` | description |
| 2 | which builder, called how | **yes** — builders are not required to share an API | description |
| 3 | launcher → 44 typed dummies | **no** — derivable from `inspect.signature` | compiler |
| 4 | compile, extract ELF + block_size, write sidecar | **no** | compiler |

Steps 3 and 4 are the same for any `@flyc.jit` launcher. Steps 1 and 2 differ per kernel and
per arch, which is exactly why the plan put the builder call in the description body.

**The remaining seam is between 2 and 3: how does the compiler get the `JitFunction`?**
`build_..._primary` returns `_launch`, a closure; `dir()` on it shows only `compile` and the
five `varlen_*` helpers. Options:

- **(a) upstream, one line** — `_launch.jit_function = launch_flash_attn_aiw`. The minimal
  form of a real AOT entry point.
- **(b) closure walk** — `[c.cell_contents for c in built.compile.__closure__ if isinstance(c.cell_contents, JitFunction)]`.
  Verified working; reaches into internals.
- **(c) description returns it** — the body already has the builder; if FlyDSL exposes the
  launcher some other way, the body hands back `jf` directly and the seam disappears.

---

## 6. The automation question

How much should the AOT compiler do for the description? Three coherent positions.

### Minimal — the description hands over a `JitFunction`

```python
def flyc_attn_fwd(f, hints):
    meta, knobs = ...              # decisions 1
    built = build_..._primary(meta, knobs)   # decision 2
    return jit_function_of(built), asdict(knobs)
```

Compiler does 3 and 4. **Pro:** smallest contract; the compiler never guesses. **Con:** every
description repeats the "get the JitFunction" incantation, so option (b) above leaks into N
descriptions instead of one.

### Middle — the description hands over the builder result; the compiler traces

```python
def flyc_attn_fwd(f, hints):
    meta, knobs = ...
    return build_..._primary(meta, knobs), asdict(knobs)
```

Compiler recovers the `JitFunction`, synthesises args from annotations, compiles. **Pro:** the
incantation lives in one place; descriptions stay declarative. **Con:** the compiler must know
*how* to recover a `JitFunction` from a builder's return — one assumption about FlyDSL's shape,
but exactly one, and it dies the day (a) lands.

### Maximal — the description hands over `(meta, knobs)`; the compiler builds too

**Rejected.** It would need to know which builder to call and with what signature, and the
plan already established that FlyDSL builders are not required to share an API. This is the
position that reintroduces per-kernel knowledge into the compiler.

### What the evidence favours

The middle position, because §4 shows the compiler's synthesised arguments **cannot** produce
a wrong artifact — the only failure mode is a missing/renamed annotation type, which raises
rather than silently mis-compiles. That is a cheap, loud failure, which is what makes
automating step 3 safe. Step 1 stays in the description precisely because it *can* be wrong
silently (choose the wrong tile and you get a valid hsaco for the wrong functional).

Rule of thumb this suggests: **automate what fails loudly, keep in the description what fails
quietly.**

---

## 7. What this deletes from the current implementation

`FakeTensor`, `_FAKE_SHAPE`, the `abi.run_compiled` monkeypatch (never restored today),
`import fmha_abi_gfx1201` — the last `fmha_*` import in the "kernel-agnostic" driver — the
`CAUSAL_TYPE=3` window fix, `philox_seed=None`, and reading `functional.choices` inside the
driver. Every one was a symptom of entering at `_launch` instead of the `JitFunction`, not a
separate problem.

## 8. Open

- Which of (a)/(b)/(c) in §5, and therefore minimal vs middle in §6.
- If (a): worth asking for `jit_function` *and* `block_size` on the same pass, since
  `known_block_size` is currently recovered by re-parsing `_source_ir`.
- Does anything else need the pre-lowering IR? If not, and FlyDSL exposed BLOCK_SIZE, the
  driver would only ever touch `_ir_text`.

---

# Plan: make the compiler kernel-agnostic

Supersedes §6-§7 above on two points, both raised in review.

## Correction 1 — FakeTensor is structural, not a workaround

§7 claimed FakeTensor could be deleted. Wrong, and the reason is worth pinning down.

`MemRefJitArg.__init__` (`jit_argument.py:258-268`) requires `element_bits`, `shape`,
`strides`, `dtype`. None of those is derivable from a parameter's annotation — `fx.Tensor`
tells you the *kind*, not the rank, extents or element type. So the moment a `@flyc.jit`
launcher takes an `fx.Tensor`, the compiler needs a descriptor object, and that is exactly
what FakeTensor is.

Today's gfx1201 launcher is all `fx.Pointer` (14 of them), which is why annotation-derived
zeros happen to suffice and why the measurement in §4 held. That is a property of *this*
kernel, not of the design. FakeTensor stays.

What was genuinely wrong is *where* it was used: as a prop for the `_launch` host-marshalling
layer, which is what dragged in `_prep`, `resolve_window`, `philox_seed=None` and the rest.
It should exist as the compiler's tensor-argument descriptor instead.

Open, for when `fx.Tensor` arrives: rank and dtype certainly reach the memref IR type;
whether concrete extents/strides do depends on whether the layout is static. Measure before
assuming a descriptor's numbers are as inert as a pointer's value.

## Correction 2 — the body takes `choices`, not a `Functional`

`flyc_compile` has only `--signature` text. A `Functional` cannot be rebuilt from it: it
carries `arch`, `arch_number`, `godel_number`, a `meta_object` back-reference and the axis
table, all of which come from the linked IR the driver does not have. The current
`_FunctionalStandIn` (`flyc_compile.py:100`) fabricates `.arch` + `.choices.<NAME>` and
nothing else — a stand-in that will drift the moment a description touches anything else on
`f`, and drift silently.

So pass the thing that genuinely round-trips through text:

```python
def flyc_attn_fwd(choices, hints):
    tile = choices['BLOCK_DMODEL']
    ...
```

`choices` is `parse_kv(args.signature, sep=' ')` — `{name: literal}`, values via
`ast.literal_eval`. Nothing else, no fabricated object.

**Arch is not needed by the body.** Verified: the only `f.arch` reader is
`_flyc_fwd_disabled`, and disable predicates run **generator-side**, where a real `Functional`
exists. The asymmetry is honest and worth stating in the contract:

| callback | phase | receives |
|---|---|---|
| `@ati.disable(when=…)` | generator (has the linked IR) | a real `Functional` |
| the `@ati.flyc.kernel` body | driver (has only text) | `choices`, `hints` |

If a body ever does need arch, add it as an explicit third parameter from `--target` rather
than smuggling it into `choices` — `choices` should mean "the functional's axis values" and
nothing more.

Spelling: a plain `dict` is the most honest representation of "parsed from text". If
`choices.BLOCK_DMODEL` reads better than `choices['BLOCK_DMODEL']`, a thin frozen mapping
supporting both is fine — but it must not grow into a `Functional` impersonation again.

## The contract

```python
# description
def flyc_attn_fwd(choices: Mapping[str, object], hints: FlycFwdHints):
    """-> (built, sidecar_dict)"""

# compiler, kernel-agnostic from here on
jf   = jit_function_of(built)          # the seam -- see below
args = synthesise_args(jf)             # table-driven from inspect.signature
jf(*args)                              # COMPILE_ONLY -> returns None
hsaco, block_size = extract(jf._last_compiled[1])
```

`synthesise_args` — the whole of the compiler's kernel knowledge:

| annotation | value | needs description input? |
|---|---|---|
| `fx.Pointer` | `flyc.from_c_void_p(fx.Uint8, 0)` | no |
| `fx.Int32` / `fx.Int64` | `0` | no |
| `fx.Float32` | `0.0` | no |
| `fx.Stream` | `fx.Stream(None)` | no |
| `fx.Tensor` | `FakeTensor(shape, strides, dtype)` | **yes** |
| anything else | raise, naming the parameter and its annotation | — |

## Where a Tensor descriptor's metadata comes from

Not invented by the compiler. The ATI description **already declares it** — the Phase-2 ABI
block has `@ati.tensor('Q', 'T_io', rank=4, strides='stride_q_*')`, i.e. rank, dtype variable
and stride names per operand. When `fx.Tensor` support lands, `synthesise_args` reads those
declarations off the `FlycDecl` rather than taking shapes from anywhere new.

Until then the `fx.Tensor` row raises `NotImplementedError` naming the parameter and pointing
at this mechanism. That is deliberate: per §6's rule — automate what fails loudly, keep in the
description what fails quietly — a missing descriptor must stop the build, not guess a shape.

## Steps

1. **Contract change.** `flyc_attn_fwd(choices, hints)`; delete `_FunctionalStandIn` and
   `_Choices` from `flyc_compile.py`; pass `parse_kv(args.signature, sep=' ')` straight in.
   Update `modules/flash/aot/flyc_attn_fwd.py`'s body (`f.choices.X` → `choices['X']`).
   `_flyc_fwd_disabled` is untouched — it keeps taking a `Functional`.
2. **`jit_function_of(built)`** — one function, the only place that knows how to reach a
   `JitFunction` from a builder's return. Closure walk today (verified working), one-line
   swap if FlyDSL exposes `_launch.jit_function`.
3. **`synthesise_args(jf)`** — the table above, from `inspect.signature(jf.func)`.
4. **Retire the marshalling path.** Delete `_trace_fmha_launch`, the `abi.run_compiled`
   monkeypatch, `import fmha_abi_gfx1201`, the `CAUSAL_TYPE=3` window special-case,
   `philox_seed=None`, and `_FAKE_SHAPE`-as-launch-shape. Keep `FakeTensor` as the descriptor
   for step 3's Tensor row.
5. **Tighten the gate.** Gate 3's kernel-agnosticism check greps for `fmha_tuning_gfx1201`
   and `attn_fwd`; it passed while `import fmha_abi_gfx1201` sat in the driver. Make it
   `fmha_*`, and add "no `flash`/`attn` module import".
6. **Re-verify.** hsaco must stay byte-identical (`bc6d0fca66a0c2d9f476` at hd 64 f16
   non-causal); rerun the 12-combination sweep; suite at 192 passed / 7 skipped.

## Still open

- The `jit_function_of` seam: one line upstream in FlyDSL versus the closure walk.
- Whether `block_size` keeps needing a `_source_ir` re-parse, or FlyDSL exposes it alongside.
- Whether a Tensor descriptor's extents affect the artifact (measure when it matters).

---

# Survey: is `jit_function_of` universally applicable?

AST survey of every module-level builder under `../FlyDSL/kernels/` that contains a nested
`@flyc.jit`. 60 files carry `@flyc.jit` at all (152 occurrences, 118 of them nested inside a
builder — so the nested pattern is the norm, not a gfx1201 quirk). 69 module-level builders:

| what the builder returns | count | `jit_function_of`? |
|---|---|---|
| the `@flyc.jit` function itself (1 jit in scope) | 38 | identity — `built` *is* the JitFunction |
| the `@flyc.jit` function itself (2 jits in scope) | 9 | identity |
| a wrapper closing over exactly **1** jit | 18 | closure walk, unambiguous |
| a wrapper closing over 2, 3, or 9 jits | 3 | **ambiguous** |
| no return | 1 | n/a (`p2p_scatter_epilog`, a fragment helper) |

**65 of 69 (94%) are unambiguous**, and the two mechanisms are trivial:

```python
def jit_function_of(built):
    if isinstance(built, JitFunction):        # 47 builders
        return built
    found = _search(built)                    # closure cells, tuple/list items, dict values
    if len(found) == 1:                       # 18 builders (incl. gfx1201)
        return found[0]
    raise ...                                 # 3 builders -- see below
```

`_search` needs to look past closures alone: `pa_decode_swa.compile_pa_decode_sw_reduce`
returns `{'launch': <jit>, 'kernel': <kernel>}` and
`moe_sorting_kernel._compile_moe_sorting_multiphase` returns a 9-tuple of launches.

## The 6% that cannot work, and why that is fine

The four exceptions are not near-misses — they are **multi-kernel builders**:

- `custom_all_reduce_kernel.make_allreduce_kernels` → dict of 3 launches
- `moe_sorting_kernel._compile_moe_sorting_multiphase` → tuple of 9 launches
- `flash_attn_gfx950.build_flash_attn_dualwave_swp_module` → wrapper over 2
- `mega_moe_stage2.p2p_scatter_epilog` → returns nothing; a fragment, not a builder

For these there is no *one* JitFunction to find, so no amount of cleverness in
`jit_function_of` helps. One builder produces several kernels, hence several hsacos — which
in AOTriton's vocabulary is a **metro**, not a single kernel, and is Phase 2+ territory. The
honest move is to raise, naming how many JitFunctions were found and pointing at the
multi-kernel case, rather than silently picking the first.

If a multi-kernel FlyDSL backend is ever wanted, the fix is a plural contract
(`jit_functions_of` → sequence, one hsaco each) — a deliberate extension, not a patch.

## Verdict: the middle design

`jit_function_of` is universal for every single-kernel builder in the tree, which is what a
`@ati.flyc.kernel` description is. Adopt the **middle** position from §6:

```python
def flyc_attn_fwd(choices, hints):
    meta, knobs = ...                              # decision 1: kernel-specific
    return build_..._primary(meta, knobs), {...}   # decision 2: kernel-specific
# compiler does the rest: jit_function_of -> synthesise_args -> compile -> extract
```

The 47 builders that already return their JitFunction make this cheaper than expected — for
most kernels `jit_function_of` is `isinstance` and nothing more. The closure walk is the
minority path, and it stays confined to one function that a one-line FlyDSL change would
retire.

## The sidecar is deliberately opaque

Not `asdict(knobs)` — that is only what *this* description happens to produce. Other builders
take entirely different arguments (`build_layernorm_module(N, dtype_str, …)`,
`make_dispatch_jit(...)`, `compile_pa_decode_sw_reduce(...)`), and there is no common shape to
impose.

So the contract is: **the second return value is a JSON-serialisable dict of whatever the
description considers necessary to reproduce this build.** The compiler does not read it, does
not validate its keys, and only writes it into the sidecar. `flyc_attn_fwd` fills it with
`asdict(knobs)` because that is what reproduces an FMHA build; a GEMM description would put
its tile shape and split factor there.

Two consequences worth stating in the contract:

- Anything the *build system* needs to consume — `block_m`, `block_size` for
  `grid_calculator()` — must be lifted to named, top-level sidecar keys by the compiler, not
  left inside the opaque blob. Opaque means "the compiler does not interpret it", not "nobody
  ever reads it".
- It must be JSON-serialisable. A frozen dataclass is not; `asdict()` of one is. The compiler
  should fail loudly on a non-serialisable value rather than dropping it.

---

# Which ABI do the decorators describe? (resolved)

## `@flyc.jit` is host code, and we strip it

The compiled module's host half — the `llvm.func` with `emit_c_interface` that the `@flyc.jit`
launcher lowers to — is discarded. We keep only `gpu.binary`. AOTriton loads the hsaco and
launches the kernel symbol with a kernarg buffer it fills itself, so the ABI that survives
into the artifact is the **`@flyc.kernel`** one. Decorators describe that.

An earlier draft argued for pointing them at `@flyc.jit` because that is the signature the
compiler calls. That conflates two jobs:

| job | source of truth | needs decorators? |
|---|---|---|
| drive the compile (which args to pass) | `inspect.signature(jf.func)` — the launcher | **no**, fully automated |
| describe the kernarg layout (Phase 2 shim) | the `@flyc.kernel` signature | yes |

The launcher never needs declaring because it is introspected at the moment it is used.

## The two ABIs are not the same, and we are choosing to assume they are

Measured over 52 single-kernel/single-jit builders in `../FlyDSL/kernels/`: names coincide
exactly in only **21**. The differences are systematic, not random:

- launcher-only launch-geometry args — `batch_size`, `m_in`, `grid`, `grid_blocks`,
  `num_tokens` — consumed to compute `grid=`, never passed to the kernel
- kernel-only trace-time args — `tiled_mma`, `tiled_copy_g2s`, `_Pad0`
- `_ptr`-suffix renames — `bt` → `block_tables_ptr` (whole files at shared=0 for this reason)

For gfx1201 specifically: 43 vs 43 params, 41 shared, differing by `batch_size`/`num_seqlens`
and `sm_scale_v`/`sm_scale_arg`, plus `stream`, plus a swap at indices 10/11.

**Working assumption: for the SDPA kernels we control, treat the two as the same.** That is a
choice, not a discovered truth — record it so nobody later reads it as a FlyDSL guarantee. If
a future flyc kernel breaks it, the fix is a per-description name map, not a general one.

## BLOCK_SIZE: use the declared value, not the ELF bound

`.max_flat_workgroup_size: 512` in the hsaco metadata is **not** logically BLOCK_SIZE. It is a
*bound*, and `.reqd_workgroup_size` is emitted **empty**, so the exact launch geometry is not
in the artifact at all. It coincides here only because flydsl derives
`rocdl.flat_work_group_size = "512,512"` (min == max) from `known_block_size`; omit that
decorator argument and the bound falls back to a default while the real launch block does not.

The concrete clue is the declared value, read off the `KernelFunction`:

```python
kf._known_block_size      # [512, 1, 1] at hd 64;  [256, 1, 1] at hd 128
```

reachable from the `JitFunction`'s closure (**not** from the builder's returned wrapper — the
`KernelFunction` is not in `built.compile`'s closure, only in `jf.func`'s). flydsl validates
the actual launch against it in `KernelLauncher._check_block_vs_known`, so it is authoritative
rather than advisory.

This also retires the `_source_ir` re-parse: no MLIR text scraping for `known_block_size`,
just an attribute read.

## What the hsaco metadata IS good for

Even though it cannot give BLOCK_SIZE, the AMDGPU note carries the full kernarg layout —
43 entries of `.offset` / `.size` / `.value_kind` / `.address_space`, plus
`.kernarg_segment_size: 292` and `.group_segment_fixed_size: 8960` (the LDS figure already in
the sidecar). Per-argument *names* are absent (one `.name:`, the kernel's own), so operand →
slot identity still comes from the kernel signature. Phase 2 can therefore validate its
generated kernarg vector against the artifact rather than trusting the declarations — worth
doing, since the declarations in this very file were wrong until an AST diff caught them.

## Correction: `flyc.compile` cannot be called

Targeting the jit *layer* is right; that particular entry point is not available.
`_compile_impl` ends in `artifact._get_func_exe()`, which builds an ExecutionEngine and needs
HIP. `COMPILE_ONLY` short-circuits `JitFunction.__call__`, not the `flyc.compile` wrapper. So
the driver calls the `JitFunction` directly.

---

# One descriptor: FakeTensor everywhere (supersedes the arg table above)

## Why

AOTriton's Interface is tensor-shaped. Every pointer-backed operand in an ATI description is
an `@ati.tensor`, including the ones that are logically scalars — the philox seed and offsets
are `@ati.tensor([...], 'T_u64', rank=0)`. So the description already speaks in tensors, and
making the compiler speak two dialects (FakeTensor for `fx.Tensor`, bare null pointers for
`fx.Pointer`) means translating between them for no reason.

Let **`fx.Pointer` accept a FakeTensor too**. Then there is exactly one operand descriptor,
and the description never has to know which annotation the launcher happened to use.

## Safe, because the element type is inert

`fx.Pointer`'s element type reaches only `PointerJitArg.__get_ir_types__` →
`PointerType.get(ir_type, addr_space, alignment)`, i.e. the *host* function's signature, which
we discard. Measured — same functional, three element types:

```
Uint8    bc6d0fca66a0c2d9f476
Float16  bc6d0fca66a0c2d9f476
Int32    bc6d0fca66a0c2d9f476     identical
```

So a descriptor's real dtype can be handed to `from_c_void_p` freely. It cannot change the
artifact, and it keeps the declaration honest rather than laundering everything through
`fx.Uint8` the way the JIT host path does (`abi.ptr_arg`).

## The table

| launcher annotation | AOT value | from |
|---|---|---|
| `fx.Pointer` | `flyc.from_c_void_p(desc.dtype, 0)` | FakeTensor descriptor |
| `fx.Tensor` | memref jit arg | the *same* FakeTensor descriptor (`element_bits`, `shape`, `strides`, `dtype`) |
| `fx.Int32` / `fx.Int64` | `0` | — |
| `fx.Float32` | `0.0` | — |
| `fx.Stream` | `fx.Stream(None)` | — |
| anything else | raise, naming the parameter and its annotation | — |

A rank-0 FakeTensor covers the scalar-pointer operands (philox seed/offset in/out) with no
special case, exactly as the ATI declaration already spells them.

## The payoff

**The launcher's choice between `fx.Pointer` and `fx.Tensor` becomes a compiler detail.** If a
future FlyDSL kernel promotes `Q` from a raw pointer to an `fx.Tensor` — which
`fmha_common_gfx1201.py:137` says was measured and rejected for kernarg-size reasons, but is
the natural direction elsewhere in the tree — the description does not change. Only the
compiler's dispatch on the annotation does, and that is already a table lookup.

It also means FakeTensor stops being "the thing the FMHA host wrapper needs" and becomes what
it should have been from the start: **the compiler's representation of an operand**, one per
declared tensor, whatever the launcher does with it.

## Where the descriptors come from

The ATI declarations, which already carry rank, dtype variable and stride names:

```python
@ati.tensor('Q', 'T_io', rank=4, strides='stride_q_*')
@ati.tensor('philox_seed_ptr', 'T_u64', rank=0)
```

Phase 1 does not need them — every gfx1201 operand is `fx.Pointer`, and a descriptor with a
null address and an arbitrary dtype suffices, so the compiler can synthesise one per pointer
parameter without consulting the description at all. Phase 2 wires the declarations in, at
which point the descriptors carry real dtypes and ranks and the `fx.Tensor` row lights up.

Concrete extents remain the open question flagged earlier: rank and dtype certainly reach a
memref IR type; whether the *numbers* do depends on whether the layout is static. Measure when
the first `fx.Tensor` launcher appears, rather than assuming a descriptor's shape is as inert
as a pointer's element type turned out to be.

## Depends on

The kernel session aligning `@flyc.jit` and `@flyc.kernel` argument order. Once that lands,
"the two ABIs are the same" stops being the working assumption recorded above and becomes
true for the SDPA kernels, so one declaration set genuinely serves both the compile call and
the Phase-2 kernarg vector.
