# PON — Plain Object Notation, and the `builder_fn(choices, …)` interface

Two related interface cleanups. Both are about the same thing: one spelling for
one concept, instead of several that agree by luck.

---

## Part 1 — the `k=v` list

`k=v;k=v` entered the codebase in Tuner v3.5 as the wire protocol between
`exaid` and `testrun`. It has since spread, and it is now written by six pieces
of code and read by three, with **no shared implementation** and — the part that
matters — **two incompatible dialects**.

### Survey

**Readers**

| where | how | safe? |
|---|---|---|
| `v3python/tune/utils.py:43` `parse_python` | `split(';')` then **`eval(v)`** | **no** — arbitrary code execution |
| ↳ callers | `tune/flash/module.py:37` `FlashEntry.parse_text`, `:73` `FlashInputMetadata.parse_text` | |
| `python/utils/kv.py` `parse_kv` (to be renamed, see Plan 0) | `split(sep)` then `ast.literal_eval` | yes, and `sep` is configurable |
| ↳ callers | `python/flyc_compile.py` (`sep=' '` for `--signature` / `--hints`) | |
| `v3src/schemaless/schemaless.cc` `Schemaless` | `std::from_chars`, bounded | yes (C++ side); rename to `Pon` |

**Writers** — six, no two sharing code

| where | separator | strings |
|---|---|---|
| `tune/flash/module.py:45` `as_posix` | `,` | bare |
| `tune/flash/module.py:56` `as_text` | `;` | **quoted**, via a local `tr()` |
| `modules/flash/aot/flash_entry.py:40` | `;` | **quoted**, `tr()` duplicated verbatim |
| `ir/triton/ksignature.py:58` `perf_section` | `;` | n/a (see below) |
| `ir/triton/ksignature.py:66` `copt_section` | `;` | n/a |
| `ir/lib/naming.py:56` `render_schemaless` | `;` | bare |

### The two dialects, and why it matters

Measured:

```
quoted : name='foo';n=3;flag=True;t=(1,2)     parse_kv -> {'name': 'foo', ...}
bare   : v_lds_layout=transposed;block_m=256  parse_kv -> ValueError: malformed node
```

The `tr()` writers quote strings so their output round-trips through
`literal_eval`. `render_schemaless` and `copt_section` emit bare `str(v)`, so
their output does **not** parse. Two dialects of one grammar, distinguishable
only by knowing which function produced the string.

**Unify on the quoted (round-trippable) dialect.** A bare `transposed` is
ambiguous with an identifier, which is precisely why the safe parser rejects it.

### Blast radius: smaller than it looks

Quoting only changes *string* values, so the question is which producers ever
emit one. Measured across every Triton psel/copt in the reference shim build,
the complete set of distinct values is:

```
0 1 2 3 4 8 16 32 64 128 256 False True
```

**All int or bool. No strings anywhere.** So Triton's `perf_section` /
`copt_section` output is already valid quoted-dialect, and unifying changes
their bytes not at all: no hsaco entry-name change, no blake2b change, no
rebuild of the 47,766 Triton images.

The only output that moves is **flyc's `#P`**, whose four `str`-typed knobs
(`v_lds_layout`, `sched_strategy`, `fp_mode`, `path_tag`) become
`v_lds_layout='transposed'`. Consequences, both already accepted for Task 2:
flyc entry names change (ZIP names do not — they are the functional layer), and
`Schemaless::get_str` must strip the surrounding quotes.

### Plan

0. **Name the language: PON.**

   By analogy with JSON — JavaScript Object Notation — this is **PON**, which
   reads two ways, both true:

   * **Plain** Object Notation — the primary gloss. It exists for the cases
     where JSON is already too much ceremony: a filename component, a CLI
     argument, one column of a database row, something a C++ parser must read
     without linking a JSON library, something a human must read at a glance in
     a log.
   * **Python** Object Notation — the secondary gloss, and technically exact:
     the values are Python literals.

   `python/utils/pon.py`, with `parse_pon()` / `render_pon()`. In prose: "a PON
   string", "PON-encoded", "the `#P` section is PON".

   **`FON` (Flat Object Notation) was the runner-up** and is accurate about the
   grammar being nesting-free, but it is harder to say, and "flat" describes the
   Chomsky class while "plain" describes the purpose. The purpose is what a
   reader needs.

   Rejected earlier, recorded so they are not revisited: `kv.py` (KV means
   keys/values in an attention codebase), anything `nv` (NVIDIA), `lite`
   (sqlite3), `assignments`, `flatspec`, `slimspec`.

   **One caution the name must not carry.** The "Python Object Notation" gloss
   invites the thought *"so I can just `eval()` it"* — which is exactly the hole
   Plan step 4 exists to close. `parse_python`'s `eval()` is a
   remote-code-execution shape, because the strings reach the tuner from the
   database. PON is parsed with `ast.literal_eval` and **never** `eval`. The
   Python in the name refers to the *literal grammar*, not to the evaluator; say
   so in the module docstring, where someone reaching for the shortcut will read
   it.

1. **That module becomes the one home**, gaining the writer beside the parser:

   ```python
   def render_pon(d: dict, sep: str = ';') -> str: ...
   ```

   `repr()` per value, not `str()` — that is what makes
   `parse_pon(render_pon(d)) == d` hold, and it is the property
   the split dialects lack. Add that round-trip as a test.

2. **Delete `render_schemaless`** from `ir/lib/naming.py` (added in `54e644f4`;
   it was the right instinct in the wrong place) and call `render_pon`. Keep
   `entry_name` where it is — it owns the `;;#F;…;;#P;…` frame, not the `k=v`
   grammar inside a section.

3. **Fold the three `tr()` writers** — `module.py:45`, `module.py:56`,
   `flash_entry.py:40` — onto `render_pon`. The comma one passes `sep=','`.
   `flash_entry.py:40` and `module.py:56` are currently byte-identical
   duplicates.

4. **Retire `parse_python`.** Its two callers become `parse_kv`. This is a
   security fix as much as a cleanup: `eval()` on a line that reaches the tuner
   from the database is a remote-code-execution shape.

5. **`Pon::get_str` (renamed from `Schemaless`) strips one pair of single quotes** — and the writer
   guarantees that is sufficient. `get_int`/`get_bool` are unaffected.

   **The concern, stated properly.** A C++ parser that faithfully accepted
   `repr()` output would need a real string unescaper, because `repr` is not the
   simple "wrap in quotes" it looks like:

   ```
   repr("it's")  ->  "it's"     switches to DOUBLE quotes
   repr('a\nb')  ->  'a\nb'     backslash escape
   repr('a\\b')  ->  'a\\b'     backslash escape
   ```

   Both quote styles plus escape decoding is exactly the kind of parser that is
   subtly wrong for years. **Do not build it.**

   **Constrain the writer instead.** `render_pon` asserts, for every
   `str` value, that `repr(v) == "'" + v + "'"` — single-quoted, nothing
   escaped. Every value the codebase actually carries satisfies this
   (`transposed`, `auto`, `noninf`, and Triton's psel/copt have no strings at
   all). If a value ever stops satisfying it, the **build** fails loudly naming
   the key, instead of the C++ misparsing silently at runtime.

   That makes the C++ side provably a two-character strip, and moves an
   unbounded parsing problem to a one-line precondition on the producer. Test
   the precondition on the Python side and the strip on the C++ side.

   On the Python-vs-C++ boolean spelling: the generator emits Python's
   `True`/`False`, not C++'s `true`/`false`. That is already handled and needs
   no change — `Pon::get_bool` (`v3src/schemaless/schemaless.cc:82-95`, to be renamed)
   exact-matches `"True"`/`"False"` deliberately, case-sensitively, treating
   `"true"` as a miss rather than silently accepting it. It also survives this
   unification untouched, because `repr(True) == str(True) == 'True'`: booleans
   are byte-identical in both dialects, so only `str`-valued knobs move.

   Keep the C++ side as the place that knows about the Python spelling. The
   alternative — teaching the *writer* to emit `true`/`false` — would break
   the round-trip property the whole unification is for.

6. **Rename the C++ side to match the language.** `class Schemaless` →
   `class Pon` (`v3src/pon/pon.cc`, `include/aotriton/_internal/pon.h`).
   `Schemaless` names how it stores; `Pon` names what it reads, which is what a
   caller needs to know. Cheap now — the class has no callers outside the flyc
   shim yet — and awkward later.

**Ordering.** 1 → 5 → 2 (C++ before the producer changes, so no build sees an
unparsable `#P`), then 3 and 4 independently.

**Gate.** `parse_pon(render_pon(d)) == d` over the knob dict and
a `FlashEntry`; a string needing escapes raises at render time;
Triton hsaco entry names byte-identical before and after; flyc ZIP names
unchanged while flyc `#P` gains quotes; suite green.

---

## Part 2 — `builder_fn(choices, hints)` should take `Functional.choices`

Today the generator passes a plain `dict` built from `compact_choices`, and the
description subscripts it (`choices['BLOCK_DMODEL']`). It should pass the real
`Functional.choices` (`ChoiceView`).

### What the description has to change to

Measured against a live functional:

| description reads today | under `ChoiceView` |
|---|---|
| `choices['BLOCK_DMODEL']` | `choices.BLOCK_DMODEL` → `16` |
| `choices['CAUSAL_TYPE']` | `choices.CAUSAL_TYPE` → `0` |
| `choices['BIAS_TYPE']`, `['ENABLE_DROPOUT']`, `['PADDED_HEAD']` | same attribute form |
| `choices['Q']` | **`choices.arg('Q')`** → `'*fp16:16'` |

`choices.Q` raises `ChoiceVarAbsent`: `Q` is an *argument*, not a choice
variable — the dtype variable is `T_io`. Use `choices.arg('Q')` rather than
`choices.T_io`: it names the operand the description already declares in
`@ati.tensor('Q', 'T_io', …)`, so it does not depend on knowing the dtype
variable's name.

### The obstacle, and the resolution: `ChoiceView` becomes an interface

**`ChoiceView` is constructed from a `Functional`, and the build-time driver has
none.** `ChoiceView.__init__` reads `functional.choice` and `functional.resolved`
(TypedChoice-valued); `flyc_compile` has only the `--signature` text it parsed
and no linked IR to rebuild a Functional from — the constraint recorded as
"Correction 2" in `jit2aot.md`, and the original reason for the plain dict.

So both call sites must present one interface while only one of them can build
the real object. **Make that literal: `ChoiceView` becomes an ABC with two
implementations.**

```
ir/choices.py                     ChoiceView(ABC)      the interface
  .tc(var) .arg(aname) .arg_tc(aname) .__getattr__(var)

ir/functional.py                  FunctionalChoiceView  backed by a Functional
python/flyc_compile.py (or ir/)   MappingChoiceView     backed by the parsed dict
```

**Move the interface out of `functional.py`.** It is reused by something that is
not a Functional — that is the definition of belonging elsewhere — and leaving
it there forces the driver to import the Functional module for a type it cannot
construct. `ir/choices.py` is the natural home; `functional.py` keeps only the
concrete implementation and the `Functional.choices` property that returns it.

What each implementation can honestly answer differs, and the ABC should say so
rather than pretend:

| method | Functional-backed | Mapping-backed |
|---|---|---|
| `__getattr__(var)` | choice variable → signature | key → parsed literal |
| `arg(aname)` | resolved argument → signature | same, when the name is a key |
| `tc(var)` / `arg_tc(aname)` | the raw `TypedChoice` | **cannot** — no TypedChoice exists |

`tc`/`arg_tc` should therefore raise a clear `NotImplementedError` naming the
backing, not return `None`. A description that reaches for a raw TypedChoice is
asking for something the build-time path genuinely does not have, and it should
find that out immediately rather than at the point the value is used.

This is strictly better than the three options an earlier draft of this document
weighed. It is not the deleted `_FunctionalStandIn`: that was an untyped
duck-typed stand-in that drifted silently as descriptions read more attributes;
this is a declared interface where the gap is part of the type, and adding a
method to the ABC forces both implementations to answer for it.

