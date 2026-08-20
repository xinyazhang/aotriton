# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ChoiceView — the interface for reading one instantiation's pinned choices,
and the two things that can back it.

`ChoiceView` used to be one concrete class (`ir/functional.py`), constructed
from a `Functional`. That does not fit `python/flyc_compile.py`: the build-time
driver has only the plain `{name: literal}` dict parsed from `--signature`
text, not a linked `Functional` it could build one from (see
`jit2aot.md`, "Correction 2"). Rather than let the driver keep subscripting a
bare dict while the generator subscripts a real object -- two spellings for
"read this kernel's pinned choices" -- `ChoiceView` becomes an ABC with two
implementations, one per side:

  `FunctionalChoiceView` (`ir/functional.py`)      backed by a `Functional`
  `MappingChoiceView` (`python/flyc_compile.py`)   backed by a parsed dict

**The interface is `arg(aname)` + attribute access, and nothing else.**
`tc`/`arg_tc` -- which hand back the raw `TypedChoice` -- are deliberately NOT
on the ABC. A `TypedChoice` is a `Functional`-side object; a parsed dict never
carried one, because only the literal survives the wire format. Putting them on
the interface would have forced the mapping side to declare two methods whose
only possible body is a raise, which states "this view has these operations"
and then denies it. They live on `FunctionalChoiceView` alone, so a caller that
needs one is asking for a Functional and finds that out from the type.
"""

from abc import ABC, abstractmethod


class ChoiceVarAbsent(AttributeError):
    """Raised by a ChoiceView when a predicate reads a choice variable (or, for
    a mapping-backed view, a key) it does not have. Subclasses AttributeError
    so getattr/hasattr duck-typing still behaves, but
    KernelDescription.is_functional_disabled catches it specifically to emit a
    write-your-own-@ati.disable diagnostic (a cited disable predicate that reads
    a variable absent from the citing kernel)."""


class ChoiceView(ABC):
    """Ergonomic accessor over one instantiation's pinned choices.

    Attribute access is keyed by *choice-variable name*: `choices.T_io` returns
    the variable's signature. `.arg(aname)` reads a resolved argument by its
    real (kernel-signature) name -- the form to use for an operand whose axis
    variable is named differently, e.g. `choices.arg('Q')` rather than
    `choices.T_io` (see `ir/functional.py`'s `Axis.signature_name` for why the
    two can differ)."""

    @abstractmethod
    def arg(self, aname):
        """The resolved (post-override) signature for a real argument name."""

    @abstractmethod
    def __getattr__(self, var):
        """The signature for a choice variable, by attribute access."""

