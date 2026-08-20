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

### 3.1 Two layers, not one table

The partition is **hierarchical**. A common layer knows the vocabulary every
stack shares and nothing else; each stack has a specialised partition that
claims its own kinds and delegates the rest.

```python
def partition(specs) -> SpecBundle:
    """Claim every COMMON spec kind. Anything unclaimed lands in
    bundle.unrecognized -- not an error here, because this layer cannot know
    what the caller's stack additionally accepts."""

def partition_flyc(specs) -> FlycSpecBundle:
    b = partition(specs)               # common kinds claimed
    ...                                # claim FlycKernelSpec / FlycHintsSpec
                                       #   out of b.unrecognized
    b.reject_remaining('@ati.flyc')    # whatever is still unclaimed IS an error
```

This is better than the flat `accept=[SpecKind(...)]` table an earlier draft of
this document proposed. That version made each stack re-list the common kinds,
which trades four copies of a `isinstance` ladder for five copies of a
declaration table — an improvement, but the duplication survives in a new form.
Here the common vocabulary is written once, in code, and a stack declares only
what is *additional* to it.

`unrecognized` is the seam that makes this work: the common layer cannot decide
whether an unclaimed spec is an error, so it does not try. Only the outermost
caller knows, and only it raises.

### 3.2 Restrictions become explicit

A stack may accept less than the common vocabulary. Affine takes no
`@ati.tensor`, `@ati.scalar`, `@ati.type_var` or `@ati.derives` today — but that
is expressed as an *absent* `elif`, which is why defect 4 (accepted vocabulary
nowhere written down) exists at all.

Under the hierarchy the common layer claims them, so the specialised layer must
say so:

```python
    b.forbid('tensors', 'scalars', 'dtype_vars', 'overrides', what='@ati.affine')
```

An affine stack carrying an `@ati.tensor` then fails naming the kind and the
stack, instead of falling into a generic "unexpected spec" branch or, worse,
being silently accepted the day someone adds the field. **The restriction is
now a line of code that has to be deleted to change the rule**, rather than the
absence of one.

### 3.3 `SpecBundle`: a dataclass, and `cite`/`disable` are NOT lists

`_partition` returns an 8-tuple, unpacked positionally at its two call sites.
Adding a kind means editing every unpack; mis-ordering two same-typed fields is
silent.

```python
@dataclass
class SpecBundle:
    tensors:      list[TensorSpec]   = field(default_factory=list)
    scalars:      list[ScalarSpec]   = field(default_factory=list)
    overrides:    list[Override]     = field(default_factory=list)
    dtype_vars:   list[ChoiceVar]    = field(default_factory=list)
    tune_records: list               = field(default_factory=list)
    cite:         CiteSpec | None    = None      # AT MOST ONE
    disable:      DisableSpec | None = None      # AT MOST ONE
    unrecognized: list               = field(default_factory=list)
```

`cite` and `disable` are **singular fields, not one-element lists**. A second of
either is rejected by `partition` at decoration time, naming the stack. This is
the earliest point at which the rule can possibly be enforced — the specs are in
hand and nothing has been built yet.

Storing them singly is what makes the check unavoidable. A `list` field with a
`max_count` rule is still a list: every consumer can iterate it, so the rule has
to be re-checked (or forgotten) wherever the plural type suggests plural is
possible. That is precisely how `affine.py:86-87` came to drop a disable
silently. The type should not admit the state the rule forbids.

Prefer a dataclass over `NamedTuple`: the list fields are mutable and a
`NamedTuple` invites positional unpacking, the exact habit being removed.

### 3.4 Declared cardinality vs resolved cardinality

This is the part the current code conflates, and the reason the check ended up
deferred to the far end of the pipeline.

**At declaration, one. After cite resolution, many — legitimately.**
`cite.py:326` builds `cited_disables = [d for cs in cited_specs for d in
cs.disables]`, and a *whole-metro* cite resolves to every sub-kernel's spec. So
a kernel that declares no disable and cites a metro with three sub-kernels
inherits three. `BuiltKernel.disables` being a list is correct.

The mistake is using one field for both. The declared fields (`cites`/
`disables` on what 3.5 renames to `KernelDecl`) are
plural *because resolution writes into them*, and that plurality then reads
backwards as "you may declare several" — which nothing checks, at any stage.

Split them:

| | declared | after resolution |
|---|---|---|
| field | `KernelDecl.cite`, `KernelDecl.disable` | `KernelDecl.resolved_disables` |
| cardinality | 0-1, enforced at partition | 0-N, legitimately |
| written by | the collector, once | `resolve_cites` |

`BuiltKernel.disables` keeps its list and its meaning. `is_functional_disabled`
keeps iterating it.

A side benefit worth noting: `resolve_cites` currently **mutates**
`spec.disables` in place, which is a large part of why `_clone_spec` exists —
`_clone_spec`'s own docstring claims "the spec is the source of truth; the
linker builds from a copy". With declared and resolved separated, the declared
fields are never written after collection, so that claim becomes structurally
true rather than maintained by discipline. Cloning is still needed
(tensors/scalars/overrides/dtype_vars are still appended to), but the shrinking
of what resolution may touch is real and worth doing on its own merits.

### 3.5 One vocabulary: `Spec` is a record, `Decl` is a collection

The names should say which layer a type belongs to, and today one of them does
not. The rule the codebase already mostly follows:

* **`*Spec`** — the record ONE `@ati.*` decorator produces. `TensorSpec`,
  `ScalarSpec`, `CiteSpec`, `DisableSpec`, `Override`, `ChoiceVar`, and the
  three stack markers `AffineKernelSpec` / `FlycKernelSpec` / `OperatorSpec`.
* **`*Decl`** — the finalized per-stack collection of those records, attached
  as `fn.__ati_node__`. `AffineDecl`, `FlycDecl`, `OperatorDecl`.

By that rule **`KernelSpec` is misnamed**: it is a collection, not a record, and
it is the only collection not called `*Decl`. Rename it **`KernelDecl`**.

`SpecBundle` fits the rule as-is — it is a bundle of `Spec` records, and it is
not attached to anything, so it is not a `Decl`.

**The existing objection is answered by step 3.4, and only by it.** The class
docstring currently argues the name difference is deliberate:

> There is no separate KernelDecl because KernelSpec must be CLONED AND MUTATED
> during linking: cite resolution appends gap tensors/scalars/overrides onto a
> per-link mutable copy of this record. OperatorDecl / AffineDecl carry no
> unresolved cross-kernel references, so the linker reads them verbatim.

That was a real distinction when written. It is already weaker than it reads —
`FlycDecl` is a `Decl` and *is* adapted into a mutable per-link copy
(`_flyc_kernel_spec`) — and 3.4 removes what is left of it: once resolution
writes `resolved_disables` instead of mutating declared fields, the declared
record is passive exactly like the other three. **So the rename is not a
cosmetic sweep; it is the step that records a design change that already
happened.** Sequence it after 3.4, delete that docstring paragraph with it, and
say in the commit message which invariant made the old name obsolete.

Knock-on renames, all mechanical (48 bare `KernelSpec` references; the 16
`AffineKernelSpec` / `FlycKernelSpec` occurrences are markers and must NOT be
swept up by a blind substitution):

| now | after |
|---|---|
| `KernelSpec` | `KernelDecl` |
| `get_kernel_spec()` | `get_kernel_decl()` |
| `kdesc.kernel_spec` attribute | `kdesc.kernel_decl` |
| `cite._kernel_spec_of` | `cite._kernel_decl_of` |
| `tools.sancheck_kernel_spec` | `tools.sancheck_kernel_decl` |
| `build_kernel(kernel_spec)` param | `build_kernel(decl)` |
| `linker._clone_spec` | `linker._clone_kernel_decl` |
| `linker._flyc_kernel_spec` | `linker._flyc_kernel_decl` |

`describe()` keeps its name regardless — it is public API and names an action,
not a type.

### 3.6 Collectors shrink to construction

```python
def collect_flyc_decl(placeholder, specs):
    b = partition_flyc(specs)
    ...                                  # build FlycDecl from b
```

Uniform `collect_*(placeholder, specs)` across all of them — ignored where
unused, and flyc already needs it for `inspect.getfile`/`__name__`.

`describe()` keeps its name and public contract, and is the only one with a
middle step:

```python
def describe(kernel, *specs, _validate=True):
    b = partition_kernel(specs)
    ...                                  # annotation specs, appended to b
    if _validate: _validate_completeness(...)
    kernel.__ati_node__ = KernelDecl(...)
```

### 3.7 Metro gets a collector

`_finalize_metro` reads `UnionPrecedenceSpec` inline and mutates
`plan.precedence`. Give it `collect_metro_decl(placeholder, specs)` so all five
paths read the same and `start()` becomes a dispatch table. Smallest item; it is
what makes the dispatch honest.

### 3.8 What this does NOT change

* **`describe()` is not renamed** and does not lose validation.
* **The four `Decl` types do not merge.** `AffineDecl`, `FlycDecl`,
  `OperatorDecl`, `KernelDecl` describe genuinely different things; only the
  partition step is shared.
* **`BuiltKernel.disables` stays a list.** See 3.4 — that plurality is real.

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

1. `SpecBundle` + the common `partition()`, with `_partition`'s two call sites
   switched over. No behavior change intended.
2. The four specialised partitions (`partition_kernel`, `partition_affine`,
   `partition_flyc`, `partition_operator`) on top of it, each with its
   `forbid(...)` line. **Two behaviour changes land here and both need their own
   line in the commit message rather than riding along as refactor side effects:**
   a second `@ati.disable` becomes an error on every stack (today affine drops
   the first silently), and an `@ati.tensor` on an affine stack becomes an error
   naming the kind.
3. Split declared from resolved (3.4): `KernelDecl.cite`/`disable` singular,
   `resolved_disables` added, `resolve_cites` writing only the latter. This is
   the largest step and the one that touches the shared Triton path, so it
   stands alone.
4. Rename `KernelSpec` -> `KernelDecl` and its knock-ons (3.5). Mechanical, but
   it must come AFTER step 3 -- that is the step that makes the name true, and
   the commit message should say so rather than presenting it as tidying. Watch
   that `AffineKernelSpec`/`FlycKernelSpec` are markers and survive unchanged.
5. `collect_metro_decl`; `start()` becomes a dispatch table; the three
   `_finalize_*` one-liners go.
6. 4.3 and 4.4 as separate later changes. 4.5 stays filed.

Item 4.1 (`_clone_spec` via the constructor) is **done** — it was promoted onto
the critical path by the `KernelSpec.name` change, since adding a
`__post_init__`-derived field to a class whose clones bypassed `__post_init__`
would have dropped it from every clone.

## 6. Gates

* Triton hsaco entry names **byte-identical** across every step. This is a
  description-layer refactor; nothing generated may move. Use the CLI
  (`--selective 'flash/triton/attn_fwd'`, 872 files) and diff with embedded
  build paths normalised — an internal-API check does not exercise
  `unique_path` and will miss an identity regression, as it did once already.
* flyc: `--selective 'flash/flyc/flyc_attn_fwd'`, 584 files, byte-identical;
  288 functionals with unchanged godel numbers.
* Suite green at each step.
* **A test asserting a second `@ati.disable` raises on every stack kind**, and a
  second `@ati.cite` likewise. This is the regression that motivates the change
  and nothing covers it today.
* A test asserting an `@ati.tensor` on an affine stack raises — the `forbid`
  path, which is new behaviour rather than a restored invariant.
* A test that a whole-metro cite still inherits N disables (3.4's plural side),
  so the singular declaration does not quietly cap resolution.
* `grep` finds one `isinstance(s, TensorSpec)` ladder in the tree, not four.
* After the rename: no bare `KernelSpec` remains, `AffineKernelSpec` and
  `FlycKernelSpec` still exist with 8 occurrences each, and every collection
  attached as `fn.__ati_node__` is named `*Decl`.

---

## 7. How this work is carried out

### Branch

```
git switch -c flyc-codegen-rewrite.unify-collectors flyc-codegen-rewrite
```

A **dot**, not a slash. Git refuses the slash form outright:

```
fatal: cannot lock ref 'refs/heads/flyc-codegen-rewrite/unify-collectors':
'refs/heads/flyc-codegen-rewrite' exists; cannot create ...
```

A ref is a file, so a branch name cannot also be a directory prefix while the
parent branch exists. The dot carries the same "child of" reading without
creating a ref-path conflict.

### Loop

1. Branch from `flyc-codegen-rewrite` HEAD, as above.
2. Implement on the branch, following the §5 ordering.
3. Review happens on the branch; fixes land as **incremental commits on top**,
   never as rewrites of earlier ones. Anything already reviewed stays
   addressable by sha.
4. When complete: **squash merge** back into `flyc-codegen-rewrite`.

### Why squash, and what that means for §5 and §6

The unification is atomic in effect — a partial application leaves two
partitioning schemes live at once, which is worse than either. And the point of
the change is deduplication, which reads as one diff and does not read at all
when spread across six commits plus review fixes.

So the granularity in §5 is for **doing and reviewing**, not for the final
history. That is not a licence to skip it:

* Every §5 step still lands as its own commit on the branch, and every §6 gate
  still runs at each step. Bisecting a byte-identity failure across the whole
  unification is exactly the situation the step boundaries exist for, and the
  branch is where that bisect would happen.
* The two steps with intended **behaviour** changes — step 2 (a second
  `@ati.disable` becomes an error; `@ati.tensor` on an affine stack becomes an
  error) and step 3 (declared/resolved split) — must each be a distinct commit
  on the branch, so review can see them separately from the mechanical moves.
* The squash message must carry those behaviour changes explicitly. A squash
  that reads "unify the collectors" and silently contains two new error
  conditions is a worse artifact than the six commits it replaced. List them.

### Before the squash

* Full §6 gate set green on the branch tip, not merely at the last step.
* `git diff flyc-codegen-rewrite...HEAD` reviewed as one diff — that is what the
  merge will look like, and it is the first time the deduplication is visible
  as a whole.
* Confirm the branch is a fast-forward candidate (no unrelated commits landed
  on `flyc-codegen-rewrite` meanwhile); if any did, rebase the branch first so
  the squash diff is only this work.
