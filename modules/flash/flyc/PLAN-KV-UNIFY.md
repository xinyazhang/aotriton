# Unifying the `k=v` wire format, and the `builder_fn(choices, …)` interface

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
| `python/utils/kv.py` `parse_kv` | `split(sep)` then `ast.literal_eval` | yes, and `sep` is configurable |
| ↳ callers | `python/flyc_compile.py` (`sep=' '` for `--signature` / `--hints`) | |
| `v3src/schemaless/schemaless.cc` | `std::from_chars`, bounded | yes (C++ side) |

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

1. **`python/utils/kv.py` becomes the one home**, gaining the writer to sit
   beside `parse_kv`:

   ```python
   def render_kv(d: dict, sep: str = ';') -> str: ...   # quoted dialect, round-trips
   ```

   `repr()` per value, not `str()` — that is exactly what makes
   `parse_kv(render_kv(d)) == d` hold, and it is the property the split dialects
   lack. Add that round-trip as a test.

2. **Delete `render_schemaless`** from `ir/lib/naming.py` (added in `54e644f4`;
   it was the right instinct in the wrong place) and call `render_kv`. Keep
   `entry_name` where it is — it owns the `;;#F;…;;#P;…` frame, not the `k=v`
   grammar inside a section.

3. **Fold the three `tr()` writers** — `module.py:45`, `module.py:56`,
   `flash_entry.py:40` — onto `render_kv`. The comma one passes `sep=','`.
   `flash_entry.py:40` and `module.py:56` are currently byte-identical
   duplicates.

4. **Retire `parse_python`.** Its two callers become `parse_kv`. This is a
   security fix as much as a cleanup: `eval()` on a line that reaches the tuner
   from the database is a remote-code-execution shape.

5. **`Schemaless::get_str` strips quotes**; `get_int`/`get_bool` are unaffected.
   Extend its unit tests with a quoted value and an unterminated quote.

   On the Python-vs-C++ boolean spelling: the generator emits Python's
   `True`/`False`, not C++'s `true`/`false`. That is already handled and needs
   no change — `Schemaless::get_bool` (`v3src/schemaless/schemaless.cc:82-95`)
   exact-matches `"True"`/`"False"` deliberately, case-sensitively, treating
   `"true"` as a miss rather than silently accepting it. It also survives this
   unification untouched, because `repr(True) == str(True) == 'True'`: booleans
   are byte-identical in both dialects, so only `str`-valued knobs move.

   Keep the C++ side as the place that knows about the Python spelling. The
   alternative — teaching the *writer* to emit `true`/`false` — would break
   `parse_kv(render_kv(d)) == d`, which is the property the whole unification
   is for.

**Ordering.** 1 → 5 → 2 (C++ before the producer changes, so no build sees an
unparsable `#P`), then 3 and 4 independently.

**Gate.** `parse_kv(render_kv(d)) == d` over the knob dict and a `FlashEntry`;
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

### The obstacle, and it is real

**`ChoiceView` is constructed from a `Functional`, and the build-time driver has
none.** `ChoiceView.__init__` reads `functional.choice` and `functional.resolved`
(TypedChoice-valued); `flyc_compile` has only the `--signature` text it parsed,
and no linked IR to rebuild a Functional from — that is the same constraint
recorded as "Correction 2" in `jit2aot.md`, and the reason the plain dict was
chosen originally.

So both call sites must present one interface, and only one of them can
construct the real thing. Options:

- **(a) A dict-backed `ChoiceView`.** A classmethod
  `ChoiceView.from_values({name: literal})` used by the driver, storing raw
  values instead of TypedChoices. One class, one interface, both sides satisfy
  it. Needs `__getattr__` and `arg()` to tolerate both backings — small, but it
  widens a core IR class for a build-tool caller.
- **(b) A duck-typed shim in the driver.** `SimpleNamespace(**parsed)` plus an
  `arg()` method. Cheapest, but it is exactly the `_FunctionalStandIn` that was
  deleted in the jit2aot work for drifting the moment a description reads a
  third attribute.
- **(c) Restrict the contract** to what both can honestly provide — i.e. keep a
  mapping, but make it `compact_choices` keyed and documented as the contract
  rather than an ad-hoc dict.

**Recommendation: (a).** It is the only one that gives a single interface
without reintroducing a stand-in, and the widening is confined to one
constructor. (b) is a known-bad shape here. (c) is the status quo with better
documentation, which is worth doing only if (a) is judged too invasive for the
IR layer.

This needs a decision before implementation, because it changes what
`flyc_compile` passes as well as what the description reads.
