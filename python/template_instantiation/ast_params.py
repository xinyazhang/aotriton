# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared AST function-parameter introspection.

Both `decorators/source.py` (a Triton kernel, located by NAME, top-level only)
and `ir/flyc/kdesc.py` (a flyc kernel, located by DECORATOR, anywhere in the
module) need the same two operations: find the one function definition a
predicate identifies, then read its plain parameter names -- no import, no
execution (agent-plans/ati_triton-free_exec0.md). They differ only in the
predicate and in whether nested scopes are searched; this module factors out
the AST mechanics both share so there is exactly one place that walks a
kernel file's syntax tree and one place that turns a `FunctionDef`'s `args`
into a signature-order name list.

This module has no dependency on `decorators/` or `ir/` (and neither of those
import it back into the other) precisely so it can sit underneath both
without inverting the existing `ir` <- `decorators` layering (`decorators/
derive.py` imports `..ir`; nothing in `ir/` imports `decorators/`)."""

import ast


class AstParamError(Exception):
    """A source file has no function definition matching the predicate, or the
    match uses `*args`/`**kwargs`, which cannot be introspected into a fixed
    ARGUMENTS order."""


def find_functions(tree, predicate, *, walk=False):
    """`FunctionDef`/`AsyncFunctionDef` nodes in `tree` (an `ast.parse` result)
    for which `predicate(node)` is truthy, in source order.

    `walk=False` (the default) looks at top-level definitions only -- a
    Triton kernel file is expected to define its kernel at module scope.
    `walk=True` visits every nested scope too (`ast.walk`) -- needed when the
    match is identified by a decorator rather than a name and is not
    guaranteed to sit at the top level."""
    nodes = ast.walk(tree) if walk else tree.body
    return [n for n in nodes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and predicate(n)]


def collect_params(fn, *, what):
    """The plain parameter names of `fn` (`posonlyargs + args + kwonlyargs`,
    in signature order) -- no import, no execution. Raises `AstParamError` if
    `fn` uses `*args`/`**kwargs`: ATI cannot introspect those into a fixed
    ARGUMENTS order. `what` names the caller's context for the error
    message (e.g. `"@ati.source(...): kernel 'attn_fwd'"`)."""
    a = fn.args
    if a.vararg is not None or a.kwarg is not None:
        raise AstParamError(
            f"{what} uses *args/**kwargs, which ATI cannot introspect into a "
            f"fixed ARGUMENTS order")
    return [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]
