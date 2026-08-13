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
