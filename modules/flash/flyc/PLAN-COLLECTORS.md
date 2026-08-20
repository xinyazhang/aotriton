# Unifying the Stage-2 collectors

**Scope note.** This is an ATI-wide refactor, not a flyc one. It lives here only
because the executive plans are kept together and dropped as a set before the
PR; nothing in it is specific to the flyc backend. It surfaced from the flyc
work because adding a fourth backend is what made the divergence visible.

---

## 1. The defect class

`start()` (`specs/finalize.py:262-272`) dispatches a stacked-`@` block five ways.
Four of those five partition the same spec vocabulary — `TensorSpec`,
`ScalarSpec`, `ChoiceVar`, `CiteSpec`, `Override`, `DisableSpec` — with four
independent `isinstance` ladders:

| stack | collector | partition lives in |
|---|---|---|
| triton | `describe()` (public Mode-B API) | `_partition`, `finalize.py:56-80` |
| affine | `collect_affine_decl(specs)` | inline ladder, `affine.py:60-90` |
| flyc | `collect_flyc_decl(placeholder, specs)` | inline ladder, `flyc.py:131-156` |
| operator | `collect_operator_decl(specs)` | inline ladder, `operator.py` |
| metro | — | inline in `_finalize_metro`, `finalize.py:277-286` |

The cost is not the repetition. It is that **every rule about the vocabulary is
restated per site, so the sites disagree and nothing detects it.** Three
measured instances, one of them a live bug:

1. **`affine.py:86-87` accepts a second `@ati.disable` and silently drops the
   first.** `elif isinstance(s, DisableSpec): disable = s` — no assertion. flyc
   carried the identical bug until the `cite`/`disable` rewrite added one. Two
   of four sites got it wrong independently; that is the signature of a rule
   with no home.
2. **Cardinality disagrees across sites for the same concept.** `_partition`
   accumulates `disables`/`cites` as unbounded lists, which overstates what a
   description can express — at most one of each is writable on any stack.
3. **Unknown-spec diagnostics disagree.** `_partition` collects `others` and
   lets `describe()` raise generically; affine and flyc raise inline with
   good, kind-specific "this stack accepts …" messages. The better diagnostic
   exists in two copies and is absent from the path most users hit.

Two further divergences that are latent rather than active:

4. **Accepted vocabulary is nowhere written down.** flyc takes `ChoiceVar` and
   `Override`; affine takes neither. Whether that is deliberate is not
   recoverable from the code — it is the absence of an `elif`.
5. **Signatures differ for no reason**: `collect_affine_decl(specs)`,
   `collect_operator_decl(specs)`, `collect_flyc_decl(placeholder, specs)`.

## 2. Why there is no `collect_triton_decl`

Worth recording, because it explains the shape and constrains the fix.

`describe()` is the *oldest* of the four and is **public API** — `ati.describe(
kernel, *specs)` is Mode B, documented at `finalize.py:11-16`. There is no
`ati.describe_affine`. So it cannot simply be renamed into the `collect_*`
family.

It also does one thing the others genuinely cannot: `kernel_params(kernel)` then
`_validate_completeness` — every signature parameter claimed exactly once.
Affine has no Python def; flyc's kernarg ABI is *declared* by the stack rather
than introspected. `_finalize_flyc` says so explicitly.

**But that asymmetry does not justify the current split.** `describe()` does
three separable things:

```
partition specs by type     <- common to all five
validate against signature  <- triton only
construct the Decl          <- common to all five
```

Only the middle step is Triton-specific. The boundary was drawn around the whole
function instead of around that step, and the partition got copied four times as
a result. **Collecting is neutral to the kernel type; only validation and
construction are not.**

## 3. Part 1 — the unification

### 3.1 `SpecBundle`: a dataclass, not a tuple

`_partition` returns an 8-tuple, unpacked positionally at its two call sites.
Adding a spec kind means editing every unpack; mis-ordering two same-typed
fields is silent.

```python
@dataclass
class SpecBundle:
    tensors:      list[TensorSpec]   = field(default_factory=list)
    scalars:      list[ScalarSpec]   = field(default_factory=list)
    overrides:    list[Override]     = field(default_factory=list)
    dtype_vars:   list[ChoiceVar]    = field(default_factory=list)
    tune_records: list               = field(default_factory=list)
    cites:        list[CiteSpec]     = field(default_factory=list)
    disables:     list[DisableSpec]  = field(default_factory=list)
    markers:      list               = field(default_factory=list)
```

Always lists, uniformly — the storage should not change shape with cardinality.
Callers that want the singular read it through one accessor that enforces the
rule:

```python
    def one(self, name):
        """The single spec in `name`, or None. Raises if there are several."""
```

so `bundle.one('cites')` and `bundle.one('disables')` are the *only* way a
0-or-1 concept is read. That single method is what makes defect 1 unrepeatable:
affine's missing assertion stops being something a site can forget, because no
site writes the check any more.

Prefer a dataclass over `NamedTuple`: the fields are mutable lists that
`resolve_cites` appends to, and a `NamedTuple` invites positional unpacking —
the exact habit being removed.

### 3.2 `partition(specs, accept, what)` — kernel-type-neutral

One function, no knowledge of triton/affine/flyc/operator/metro. What varies is
passed in:

```python
@dataclass(frozen=True)
class SpecKind:
    field:     str          # SpecBundle field to append to
    type:      type         # the spec class to match
    max_count: int | None = None   # None = unbounded; 1 = at most one
```

Each stack declares its vocabulary as a tuple of `SpecKind`, next to that
stack's `Decl`. `partition` walks `specs` once, appends by declared field, and
raises on (a) a spec matching no accepted kind — message built from `accept`, so
every stack gets the good diagnostic that only affine and flyc have today — and
(b) a count exceeding `max_count`, naming the stack and the kind.

This is where defects 1-4 are fixed at once: cardinality and vocabulary become
**data, declared per stack in one place**, rather than control flow restated per
stack in four.

The first `SpecKind` of each stack is its marker (`AffineKernelSpec`,
`FlycKernelSpec`, `OperatorSpec`), which `start()` already uses as the O(1)
discriminant; declaring it with `max_count=1` also folds in the three hand-rolled
"multiple markers in one stack" assertions.

### 3.3 Collectors shrink to construction

```python
def collect_flyc_decl(placeholder, specs):
    b = partition(specs, accept=FLYC_SPEC_KINDS, what='@ati.flyc')
    ...                                  # build FlycDecl from b
```

Uniform `collect_*(placeholder, specs)` signature across all of them —
placeholder ignored where unused, and flyc already needs it for
`inspect.getfile`/`__name__`.

`describe()` keeps its name and public contract, and becomes the only one with a
middle step:

```python
def describe(kernel, *specs, _validate=True):
    b = partition(specs, accept=KERNEL_SPEC_KINDS, what='ati.describe')
    ...                                  # annotation specs, appended to b
    if _validate: _validate_completeness(...)
    kernel.__ati_node__ = KernelSpec(...)
```

### 3.4 Metro gets a collector

`_finalize_metro` currently reads `UnionPrecedenceSpec` inline and mutates
`plan.precedence`. Give it `collect_metro_decl(placeholder, specs)` for
symmetry, so all five paths read the same and `start()` is five identical calls.
This is the smallest item and the one that makes the dispatch table honest.

### 3.5 What this does NOT change

State plainly, so a later reader does not "finish the job" wrongly:

* **`describe()` is not renamed** and does not lose validation.
* **The four `Decl` types do not merge.** `AffineDecl`, `FlycDecl`,
  `OperatorDecl`, `KernelSpec` describe genuinely different things. Only the
  partition step is shared.
* **Triton's `disables`/`cites` list fields on `KernelSpec` are not reshaped
  here.** `partition` will enforce `max_count=1`, so the lists become
  0-or-1-element by construction, but changing the field type touches
  `resolve_cites` and `build_kernel` and belongs in its own change.

## 4. Part 2 — adjacent opportunities, ranked

Ranked by (duplication is real) × (silent divergence is likely), not by size.

### 4.1 `_clone_spec` bypasses `__post_init__` — highest risk, smallest fix

`linker.py:80-97` builds a clone with `KernelSpec.__new__(KernelSpec)` and
hand-assigns ten attributes. **`__post_init__` never runs.** Today that is
survivable only because the clone hand-copies `source_path`, the one field
`__post_init__` derives. Any future field with a post-init default is silently
absent from every cloned spec — and cloning is what the linker builds from, so
the original would look correct in tests that read the spec directly.

`_flyc_kernel_spec` (`linker.py:100-128`), written in the Part 3 work, uses the
real constructor instead. **The two clone paths have already diverged in
construction style.** Converge on the constructor form and delete the `__new__`
version.

### 4.2 The three `_finalize_*` one-liners

`_finalize_affine`, `_finalize_flyc`, `_finalize_operator` are the same
statement three times (`placeholder.__ati_node__ = collect_X(...)`). Once 3.3
makes the signatures uniform, `start()`'s dispatch collapses to a
`{marker_type: collector}` table and the three functions disappear. Do this
**with** 3.3, not after — it is the payoff that makes the uniform signature
worth having.

### 4.3 `ir/*/ksignature.py` — partial, not full

Shared method names: `__init__`, `blake`, `perf_section`, `copt_section`,
`hsaco_entry_name` (triton 98 lines, flyc 61). But the flyc module's docstring
argues correctly that Triton's `num_warps`/`num_stages`/`waves_per_eu`,
`COMPILER_OPTIONS` and the gfx1250 double-warps workaround do not apply.

So unify **the frame, not the vocabulary**: `hsaco_entry_name` and `blake`
compose `;;#F;…;;#P;…;;#CO;…;;arch=…` identically and already both route through
`ir/lib/naming.py`. Lift those two to a small shared base; leave
`perf_section`/`copt_section` abstract. Resist merging further — the sections
genuinely differ, and forcing them together would reintroduce the Triton
vocabulary into flyc that the split deliberately removed.

### 4.4 `codegen/autotune.py` vs `codegen/flytune.py`

Shared: `__init__`, `all_signatures`, `codegen_compact_kernels`,
`codegen_deduplicated_pp_args_function_index`, `generate` (227 vs 172 lines).
Same shape, different signature sources — Triton's from the tuning dataframe,
flyc's from `builder_fn(choices, hints)`. Candidate for a shared base with
`all_signatures` abstract. Lower priority than 4.1-4.2; the divergence is
visible in the class names rather than hidden.

### 4.5 The shim templates — already filed, keep filed

`codegen/flyc.py:28-38` records the measurement: ~34 of ~200 lines differ in the
`.cc`, ~31 of ~110 in the `.h`, with `kctl.control_bits` appearing three times in
each. The TODO already names the parameterisable divergences (header name,
`PP_FUNC` signature, one `Pon` line, tune namespace).

**Leave it filed until flyc stops moving.** It is the largest item and the one
whose inputs are still changing; it is listed here only so the plan is complete.

## 5. Ordering

1. `SpecBundle` + `partition` + `SpecKind`, with `_partition`'s two call sites
   switched over. No behavior change intended.
2. affine/flyc/operator collectors onto `partition`. **Affine's behavior does
   change here** — a second `@ati.disable` becomes an error instead of a silent
   overwrite. That is the point, and it needs its own line in the commit
   message rather than riding along as a refactor side effect.
3. `collect_metro_decl`; `start()` becomes a dispatch table; the three
   `_finalize_*` go.
4. `_clone_spec` onto the constructor (4.1) — independent of 1-3, can go first
   if convenient.
5. 4.3 and 4.4 as separate later changes. 4.5 stays filed.

## 6. Gates

* Triton hsaco entry names **byte-identical** across steps 1-4. This is a
  description-layer refactor; nothing generated may move.
* flyc ZIP names and `#P` sections unchanged; 288 functionals, same godel
  numbers.
* Suite green at each step.
* A new test asserting a second `@ati.disable` raises on **every** stack kind —
  the regression that motivates the whole change, and the one thing no current
  test covers.
* `grep` finds one `isinstance(s, TensorSpec)` ladder in the tree, not four.
