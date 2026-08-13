# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ContextHelper — a `wires_to=` value naming a host-side C++ member function instead
of an operand rename.

`wires_to` on `@ati.tensor` / `@ati.scalar` has historically been a plain operand
name (the real→apparel wiring: this kernel argument IS that operator operand, under
a different name). `ir/triton/kdesc.py`'s `apparel_of` docstring already anticipates
a second representation: "the apparel value is a plain operand name for now; the
representation is kept opaque so it can later carry a tuple of operator params or an
expression." `ContextHelper` is that second representation.

It names a member function on the generated CONTEXT class (e.g. `FlycAttnFwdContext`)
that the author hand-implements in `modules/flash/csrc/`, exactly as
`grid_calculator()` already is:

    wires_to=ati.context_helper('flyc_num_seqlens')
      -> declares  int32_t flyc_num_seqlens() const;  on the context class
      -> author implements it by hand in modules/flash/csrc/<kernel>.cc

Lives in `ir/`, not `decorators/`: it is not a decorator (never stacked on a def via
`@`) — it is a *value* passed to `wires_to=`, stored on the Tensor/ScalarSpec and
read back through the apparel-wiring machinery. It is also not language-specific
(not under `ir/triton/` or `ir/flyc/`): any backend needing host-side translation for
an argument that is not a plain rename can use it.

Phase 1: stored and otherwise unused. No codegen reads `ContextHelper` yet — that is
Phase 2's job (declaring the member on the context class, and any `wires_to`
consumption in the shim generator). It exists here so descriptions using it (like
`modules/flash/aot/flyc_attn_fwd.py`) parse whole.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ContextHelper:
    """A `wires_to=` value naming a host-side context member function, not an
    operand rename. `name` is the C++ member function name (e.g.
    `flyc_num_seqlens`), declared with no arguments and a return type taken from
    the `@ati.scalar`/`@ati.tensor` this value is attached to."""

    name: str

    def __post_init__(self):
        assert isinstance(self.name, str) and self.name, \
            f'ati.context_helper needs a non-empty member function name, got {self.name!r}'

    def __repr__(self):
        return f'ContextHelper({self.name!r})'


def context_helper(name: str) -> ContextHelper:
    """`ati.context_helper('flyc_num_seqlens')` — see module docstring."""
    return ContextHelper(name)
