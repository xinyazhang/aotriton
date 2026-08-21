# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The FlycDecl passive record + its collector (pipeline Stage 2).

The stacked-@ flyc finalizer (specs/finalize.py) partitions an @ati.flyc.* stack
into one FlycDecl, attached to the def as `fn.__ati_node__`. NO build, and — unlike
a triton kernel — NOT routed through describe(): describe() validates that specs
claim every parameter of a known (AST-parsed) signature exactly once, and a flyc
description has no such signature (the hsaco kernarg ABI is declared by the
@ati.tensor/@ati.scalar stack itself, not introspected from a def). Collected
passively instead, like AffineDecl: the disable predicate, the cite(s), the module
path, the placeholder function, the registered hints dataclass, and the
tensor/scalar specs as an inert list — nothing here is built or validated.
`aotriton.flyc_compile` (Phase 1's only consumer) reads `source_path` and `hints()`.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .node import BuildableDecl, derived_decl_fields
from .bundle import partition
from ..ast_params import find_one_function, has_decorator_attr, collect_params

if TYPE_CHECKING:
    from ..decorators.disable import DisableSpec
    from ..decorators.cite import CiteSpec


@dataclass(kw_only=True)
class FlycDecl(BuildableDecl):
    """Passive record of an @ati.flyc stack (a flyc kernel's "object file").

    Adds four fields to `BuildableDecl` (specs/node.py), and they are the four
    places flyc genuinely differs from Triton -- all of them build- or
    tune-time, none of them about how the kernel is DESCRIBED:

      desc_path       the description module's own file, for the Fly.compile
                      DESC column (build plumbing)
      hints_cls       the @ati.flyc.hints dataclass (tune)
      fn              the deferred builder the FlyDSL compiler calls (build)

    Everything else -- the kernarg ABI specs, cites, disable, params, and the
    vendored kernel file as `source_path` -- is the shared vocabulary. This
    record used to carry that path a SECOND time as `module_path`, one fact
    under two names with nothing keeping them equal.

    `cites` is plural (inherited): stacking several `@ati.cite` is a deliberate
    pattern, "citation mode (b)". `disable` stays singular, enforced by
    `partition_flyc`.
    """

    desc_path: Path = None             # the DESCRIPTION module's own file
    hints_cls: type | None = None      # the @ati.flyc.hints dataclass, or None
    fn: object = None                  # the placeholder def itself (the builder)

    def hints(self):
        """A default-constructed instance of the registered hints dataclass, or
        None if @ati.flyc.hints was never applied."""
        if self.hints_cls is None:
            return None
        return self.hints_cls()


def _flyc_kernel_stub(module_path, name):
    """AST-parse `module_path` (the vendored kernel file) for the unique
    `@flyc.kernel`-decorated function and wrap it as a `KernelStub` -- the same
    non-importing stand-in `@ati.source` builds for a Triton kernel
    (decorators/source.py). Unlike a Triton kernel (looked up by NAME,
    top-level only), the flyc kernel def is looked up by DECORATOR and may sit
    in a nested scope, so this walks every scope (`walk=True`) and requires the
    match to be unique."""
    from ..decorators.source import KernelStub

    path = Path(module_path)
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    what = f'flyc kernel {name!r}: {path}'
    fn = find_one_function(tree, lambda n: has_decorator_attr(n, 'kernel'),
                           walk=True, what=what)
    params = collect_params(fn, what=what)
    return KernelStub(fn.name, params, str(path))


def partition_flyc(specs):
    """Common vocabulary, restricted: a flyc kernel has the same kernarg-ABI
    concept as a Triton kernel (tensors/scalars/overrides/dtype-vars) and the
    same @ati.cite(s)/@ati.disable, but no tune-record concept of its own yet
    -- `tune` on FlycDecl is an inert placeholder nothing populates. The
    @ati.flyc.* markers (FlycKernelSpec/FlycHintsSpec) are flyc-specific and
    claimed out of b.unrecognized by collect_flyc_decl below.

    No `forbid('cites', ...)` here: unlike specs/affine.py, flyc accepts the
    common vocabulary's plural `cites` outright -- there used to be a
    `assert cite is None` here restricting a flyc stack to one @ati.cite, but
    that was an artifact of this collector being written narrowly before the
    common vocabulary existed, not a rule flyc actually needs; multi-cite
    ("citation mode (b)", see specs/bundle.py) is just as legitimate for a
    flyc kernel as for a Triton one."""
    b = partition(specs)
    b.forbid('tune_records', what='@ati.flyc')
    return b


def collect_flyc_decl(placeholder, specs):
    """Partition an @ati.flyc stack into a passive FlycDecl (no build, no
    describe() validation)."""
    from ..decorators.flyc import FlycKernelSpec, FlycHintsSpec

    b = partition_flyc(specs)
    marker = None
    hints_cls = None
    remaining = []
    for s in b.unrecognized:
        if isinstance(s, FlycKernelSpec):
            assert marker is None, 'multiple @ati.flyc.kernel markers in one stack'
            marker = s
        elif isinstance(s, FlycHintsSpec):
            assert hints_cls is None, 'duplicate @ati.flyc.hints on one kernel'
            hints_cls = s.hints_cls
        else:
            remaining.append(s)
    b.unrecognized = remaining
    b.reject_remaining('@ati.flyc')
    assert marker is not None, '@ati.start flyc path without an @ati.flyc.kernel marker'
    # The DESCRIPTION module's own file (e.g. modules/flash/aot/flyc_attn_fwd.py),
    # NOT marker.module_path (the vendored kernel file under modules/flash/flyc/).
    # This is what codegen/root.py's Fly.compile DESC column feeds to
    # `aotriton.flyc_compile <desc_path> --kernel_name ...` (Task 5c).
    desc_path = Path(inspect.getfile(placeholder)).resolve()
    from ..introspect import kernel_params
    stub = _flyc_kernel_stub(marker.module_path, placeholder.__name__)
    # `name` passed explicitly: the flyc DESCRIPTION's identity, not the
    # AST-located @flyc.kernel def's name in the vendored file.
    return FlycDecl(desc_path=desc_path,
                    hints_cls=hints_cls, fn=placeholder,
                    params=kernel_params(stub), tensors=b.tensors,
                    scalars=b.scalars, overrides=b.overrides,
                    dtype_vars=b.dtype_vars, cites=b.cites,
                    **derived_decl_fields(stub, b.disable,
                                          name=placeholder.__name__))
