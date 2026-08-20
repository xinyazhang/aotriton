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

  `FunctionalChoiceView` (`ir/functional.py`)   backed by a `Functional`
  `MappingChoiceView` (below)                   backed by a parsed dict

Both answer `.tc(var)` / `.arg(aname)` / `.arg_tc(aname)` / attribute access,
but not equally honestly: a `Functional` carries a real `TypedChoice` per
variable, while a parsed dict carries only the literal that survived the wire
format. `tc`/`arg_tc` therefore raise `NotImplementedError` on the
mapping-backed side, naming what is missing, rather than fabricating a
`TypedChoice` or returning `None` for something the caller quietly starts
depending on.
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
    the variable's signature. `.tc(var)` returns the raw `TypedChoice`, when the
    backing has one. `.arg(aname)` / `.arg_tc(aname)` read a resolved argument
    by its real (kernel-signature) name -- the form to use for an operand whose
    axis variable is named differently, e.g. `choices.arg('Q')` rather than
    `choices.T_io` (see `ir/functional.py`'s `Axis.signature_name` for why the
    two can differ)."""

    @abstractmethod
    def tc(self, var):
        """The raw TypedChoice for a choice variable, if the backing has one."""

    @abstractmethod
    def arg(self, aname):
        """The resolved (post-override) signature for a real argument name."""

    @abstractmethod
    def arg_tc(self, aname):
        """The resolved (post-override) raw TypedChoice for a real argument
        name, if the backing has one."""

    @abstractmethod
    def __getattr__(self, var):
        """The signature for a choice variable, by attribute access."""


class MappingChoiceView(ChoiceView):
    """`ChoiceView` backed by a plain `{name: literal}` dict -- what
    `python/flyc_compile.py` has: the `--signature` text, parsed by
    `parse_pon`, with no linked `Functional` to build a real one from.

    There is no distinction here between a "choice variable" and a "resolved
    argument": both are just keys of the one dict the wire format carries, so
    `arg(aname)` and attribute access answer identically when `aname` is a
    key. `tc`/`arg_tc` raise `NotImplementedError` naming this backing: a
    parsed dict never carried a `TypedChoice`, so returning one (or `None`)
    would manufacture a fact this side of the driver does not have."""

    __slots__ = ('_d',)

    def __init__(self, d: dict):
        self._d = dict(d)

    def tc(self, var):
        raise NotImplementedError(
            f'MappingChoiceView has no TypedChoice for {var!r}: it is backed by '
            f'a parsed dict (python/flyc_compile.py --signature text), not a '
            f'Functional, so there is no TypedChoice to return.')

    def arg(self, aname):
        if aname not in self._d:
            raise KeyError(
                f'{aname!r} is not a key of this choices mapping; '
                f'valid: {sorted(self._d)}')
        return self._d[aname]

    def arg_tc(self, aname):
        raise NotImplementedError(
            f'MappingChoiceView has no TypedChoice for {aname!r}: it is backed '
            f'by a parsed dict (python/flyc_compile.py --signature text), not a '
            f'Functional, so there is no TypedChoice to return.')

    def __getattr__(self, var):
        # __slots__ means only '_d' can ever be a real instance attribute, so
        # any other name that reaches here is a mapping key lookup.
        d = object.__getattribute__(self, '_d')
        if var not in d:
            raise ChoiceVarAbsent(
                f'{var!r} is not a key of this choices mapping; '
                f'valid: {sorted(d)}')
        return d[var]
