# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FlyDSL analogue of `python/compile.py`: cross-compile one
`@ati.flyc.kernel` description to a target-arch `.hsaco`, with no GPU present
and no kernel ever launched (see `flyc_bootstrap.setup`, `COMPILE_ONLY=1`).

Invoked as `python -m aotriton.flyc_compile`:

    python -m aotriton.flyc_compile modules/flash/aot/flyc_attn_fwd.py \\
        --kernel_name flyc_attn_fwd --target gfx1201 \\
        --signature "Q='*fp16:16' BLOCK_DMODEL=64 CAUSAL_TYPE=0 BIAS_TYPE=0 \\
                     ENABLE_DROPOUT=False PADDED_HEAD=False" \\
        --hints "seqlen_q=0 seqlen_k=0" \\
        --out_path build/flyc/attn_fwd_hd64_f16

**Kernel-agnostic by construction.** This module drives *any*
`@ati.flyc.kernel` description through `fn.__ati_node__` (`source_path`,
`hints()`) and the plain `fn(choices, hints) -> built` call — it does not
import a specific kernel family's tuning module, and nothing here names a
specific kernel. It enters at the `@flyc.jit` `JitFunction` the description's
builder produced (`jit_function_of`), synthesises typed dummy arguments from
the launcher's own signature (`synthesise_args`) and calls it directly — no
host marshalling, no fabricated tensors, no kernel-specific shapes.
"""

import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
import types
from argparse import ArgumentParser
from multiprocessing import Process, Queue
from pathlib import Path

from . import flyc_bootstrap
from .template_instantiation.ir.choices import ChoiceView, ChoiceVarAbsent
from .utils import parse_pon


class MappingChoiceView(ChoiceView):
    """`ChoiceView` backed by a plain `{name: literal}` dict -- what this
    driver has: the `--signature` text, parsed by `parse_pon`, with no linked
    `Functional` to build a real view from.

    Lives here rather than beside the ABC because this is its only user. The
    generator side never constructs one (it has real Functionals), and a class
    with one call site belongs at that call site.

    There is no distinction here between a "choice variable" and a "resolved
    argument": both are just keys of the one dict the wire format carries, so
    `arg(aname)` and attribute access answer identically when `aname` is a key.
    There is deliberately no `tc`/`arg_tc`: a parsed dict never carried a
    `TypedChoice`, and the ABC does not ask for one (see `ir/choices.py`)."""

    __slots__ = ('_d',)

    def __init__(self, d: dict):
        self._d = dict(d)

    def arg(self, aname):
        if aname not in self._d:
            raise KeyError(
                f'{aname!r} is not a key of this choices mapping; '
                f'valid: {sorted(self._d)}')
        return self._d[aname]

    def __getattr__(self, var):
        # __slots__ means only '_d' can ever be a real instance attribute, so
        # any other name that reaches here is a mapping key lookup.
        d = object.__getattribute__(self, '_d')
        if var not in d:
            raise ChoiceVarAbsent(
                f'{var!r} is not a key of this choices mapping; '
                f'valid: {sorted(d)}')
        return d[var]


desc = """
FlyDSL ahead-of-time compiler: cross-compiles one @ati.flyc.kernel description
to a target-arch .hsaco. No GPU is used or required.
"""


def parse():
    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "path",
        help="Path to the ATI description module containing the @ati.flyc.kernel function.",
    )
    parser.add_argument(
        "--kernel_name", type=str, required=True,
        help="Name of the @ati.flyc.kernel def, in `path`, to drive.",
    )
    parser.add_argument(
        "--target", type=str, required=True,
        help="Ahead-of-time compile architecture, e.g. gfx1201.",
    )
    parser.add_argument(
        "--signature", type=str, required=True,
        help="The functional, as space-separated key=value pairs, e.g. "
             "\"Q='*fp16:16' BLOCK_DMODEL=64 CAUSAL_TYPE=0\".",
    )
    parser.add_argument(
        "--hints", type=str, default='',
        help="Overrides for the @ati.flyc.hints dataclass fields, same "
             "encoding as --signature. Unset fields keep their declared default.",
    )
    parser.add_argument("--out_path", type=Path, required=True, help="Out filename (without extension).")
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Maximal time the compiler can run, in minutes. 0 for indefinite.",
    )
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")
    parser.add_argument(
        "--verify", dest='verify', action='store_true', default=True,
        help="Verify the produced ELF with llvm-readelf (default on).",
    )
    parser.add_argument(
        "--no_verify", dest='verify', action='store_false',
        help="Skip ELF verification.",
    )
    return parser.parse_args()


def _build_hints(node, hints_str: str):
    """The `@ati.flyc.hints` dataclass instance: `node`'s defaults, `--hints`
    overrides applied on top. Rejects unknown fields loudly -- a typo in
    `--hints` must not silently build the default schedule."""
    defaults = node.hints()
    if not hints_str:
        return defaults
    overrides = parse_pon(hints_str, sep=' ')
    valid = {f.name for f in dataclasses.fields(defaults)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(
            f"--hints has unknown field(s) {unknown} for {type(defaults).__name__}; "
            f"valid fields: {sorted(valid)}"
        )
    return dataclasses.replace(defaults, **overrides)


def _load_description_module(path: Path, kernel_name: str):
    """Load the ATI description module at `path` and return `kernel_name` from it.

    `importlib.util.spec_from_file_location`, per the plan -- but a description
    module may live inside a real package and use relative imports
    (`modules/flash/aot/flyc_attn_fwd.py` does: `from ._common import ...`), so
    every ancestor directory that has an `__init__.py` is registered as a bare
    namespace-package stub first. That resolves the relative import without
    running the enclosing package's own `__init__.py` -- which, for a kernel
    family's `aot` package, imports every OTHER backend and is none of this
    driver's business.
    """
    path = path.resolve()
    pkg_parts = []
    root = path.parent
    while (root / '__init__.py').is_file():
        pkg_parts.insert(0, root.name)
        root = root.parent

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    pkg_dir = root
    pkg_name = ''
    for part in pkg_parts:
        pkg_dir = pkg_dir / part
        pkg_name = part if not pkg_name else f'{pkg_name}.{part}'
        if pkg_name not in sys.modules:
            stub = types.ModuleType(pkg_name)
            stub.__path__ = [str(pkg_dir)]
            stub.__package__ = pkg_name
            sys.modules[pkg_name] = stub

    module_name = '.'.join(pkg_parts + [path.stem]) if pkg_parts else path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, kernel_name)


class FakeTensor:
    """A device-less operand descriptor, used to trace a FlyDSL launch.

    Not a torch tensor, and the name is load-bearing: FlyDSL's host ABI
    (`fmha_abi_gfx1201.ptr_arg`) special-cases the exact class name
    "FakeTensor" to a null device pointer. Only `.shape`, `.dim()`,
    `.stride(i)`, `.data_ptr()` and `.device` are read from it (`strides_of`,
    `ptr_arg`, `dropout_args`). The shape does not affect the compiled
    artifact -- measured across four shape/layout combinations, byte-identical
    -- so one fixed contiguous BHSD shape covers every functional.

    ---------------------------------------------------------------------
    Why this class survives, and what it is for next: `fx.Tensor` operands
    ---------------------------------------------------------------------

    Today every operand of the gfx1201 launcher is an `fx.Pointer`, for which
    a descriptor is barely needed -- `flyc.from_c_void_p(dtype, 0)` would do,
    and the pointer element type is provably inert (measured: Uint8 / Float16
    / Int32 all give a byte-identical hsaco, because it reaches only the host
    function signature, which is discarded along with the rest of the host
    module).

    That is NOT true of `fx.Tensor`. `MemRefJitArg.__init__` requires
    `element_bits`, `shape`, `strides` and `dtype`, none of which is derivable
    from a parameter annotation -- `fx.Tensor` states the kind, not the rank,
    extents or element type. So the moment a launcher takes a tensor, the
    compiler needs a real descriptor, and this is it. Keeping one descriptor
    type for both annotations is also what lets a kernel promote an operand
    from pointer to tensor without touching its ATI description: the choice
    becomes a compiler-side table lookup.

    **`fx.Tensor` is deliberately NOT supported yet** -- see the raise in
    `_operand_for`. What is actually known is one measurement,
    recorded at `modules/flash/flyc/fmha_common_gfx1201.py:138-146`: each
    `fx.Tensor` argument adds a **40-byte by-value memref descriptor
    interleaved immediately after its pointer**, growing that kernel's kernarg
    segment from 268 to 428 bytes and shifting the offset of every argument
    after the first. (This kernel's current `.kernarg_segment_size` is 292.)

    What is unknown, and must be settled before the tensor path is written:

    * What those 40 bytes contain -- rank, sizes, strides, offset? The answer
      decides whether the explicit `stride_*` arguments become redundant.
    * Whether the memref layout is static or dynamic, i.e. whether a
      descriptor's concrete extents reach the IR type. The pointer element
      type turned out inert; that is no evidence either way about shapes.
    * How the C++ shim fills it. It currently writes a flat kernarg buffer
      from the params struct; an interleaved by-value descriptor is a
      different and more fragile layout to generate.
    * Whether it is wanted here at all. The SDPA kernels use raw pointers plus
      explicit strides *because* the kernarg layout is an AOTriton ABI. That
      trade was made knowingly and should not be undone by a compiler feature.

    Position while FlyDSL's ABI settles: stay explicit. Raw pointers with
    named stride arguments give a layout we write down and can diff against
    the `.offset` / `.size` entries in the artifact's own AMDGPU metadata.
    """

    def __init__(self, shape):
        self._shape = tuple(shape)
        b, h, s, d = self._shape
        self._stride = (h * s * d, s * d, d, 1)
        self.device = None

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def stride(self, i):
        return self._stride[i]

    def data_ptr(self):
        return 0


def _operand_for(param, desc=None):
    """The AOT value for one launcher parameter, derived from its annotation
    alone -- the whole of this driver's kernel knowledge.

    | annotation | value |
    |---|---|
    | `Pointer` | `flyc.from_c_void_p(desc.dtype if desc else fx.Uint8, 0)` |
    | `Int32`, `Int64` | `0` |
    | `Float32` | `0.0` |
    | `Stream` | `fx.Stream(None)` |
    | `Tensor` | raise -- not supported yet, see below |
    | anything else | raise, naming the parameter and its annotation |

    `desc` is a `FakeTensor` descriptor, optional and unused in Phase 1 (every
    gfx1201 operand is `fx.Pointer`, so `flyc.from_c_void_p(fx.Uint8, 0)`
    suffices -- the pointer element type is provably inert, see `FakeTensor`'s
    docstring). It exists for the `fx.Tensor` row, which needs `desc.dtype`
    (and, once implemented, `desc`'s rank/shape/strides) rather than a bare
    default.

    This is also the loud-stop this driver owes a launcher it cannot honestly
    synthesise for -- folded in here rather than kept as a separate
    enumerate-and-check pass, so there is exactly one place that knows the
    supported annotation set.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    name = getattr(param.annotation, '__name__', str(param.annotation))
    if name == 'Pointer':
        dtype = desc.dtype if desc is not None else fx.Uint8
        return flyc.from_c_void_p(dtype, 0)
    if name in ('Int32', 'Int64'):
        return 0
    if name == 'Float32':
        return 0.0
    if name == 'Stream':
        return fx.Stream(None)

    tensor_note = ''
    if name == 'Tensor':
        tensor_note = (
            "\n\nfx.Tensor operands are not supported yet -- deliberately, not by "
            "oversight. Each one adds a 40-byte by-value memref descriptor "
            "interleaved after its pointer, and the contents of those bytes, the "
            "static/dynamic layout question, and how the C++ shim fills them are "
            "all unverified. See FakeTensor's docstring in this file."
        )
    raise NotImplementedError(
        f"unsupported operand annotation for parameter {param.name!r}: fx.{name}."
        f"{tensor_note}"
    )


def synthesise_args(jf):
    """The positional argument list for one traced call of `jf`, built purely
    from `jf`'s own signature -- no functional, no choices, no shapes. See
    `_operand_for` for the per-parameter rule; Phase 1 passes `desc=None`
    everywhere, since every gfx1201 launcher parameter is `fx.Pointer` or a
    plain scalar/stream.
    """
    return [_operand_for(p) for p in _launcher_signature(jf).parameters.values()]


class _AmbiguousObject(Exception):
    """Raised by `_find_unique` when a closure/container walk finds zero or
    several instances of the wanted class instead of exactly one."""

    def __init__(self, obj, cls, found):
        self.obj = obj
        self.cls = cls
        self.found = found
        super().__init__(
            f"expected exactly one {cls.__name__}, found {len(found)} while "
            f"searching {obj!r}"
        )


def _find_unique(obj, cls):
    """The one search both `jit_function_of` and `kernel_function_of` need:
    find the single instance of `cls` reachable from `obj`.

    `obj` is either a plain closure (the common shape for a FlyDSL builder's
    returned wrapper, or a `@flyc.jit` launcher's own function object) or a
    tuple / list / dict of candidates (`pa_decode_swa.compile_pa_decode_sw_reduce`
    returns `{'launch': jit, 'kernel': kernel}`; `moe_sorting_kernel` returns a
    9-tuple of launches). Collect matches by identity and return the single
    one; raise `_AmbiguousObject` naming how many were found otherwise -- see
    that exception's callers for what "several" means in each case.
    """
    if isinstance(obj, dict):
        candidates = list(obj.values())
    elif isinstance(obj, (tuple, list)):
        candidates = list(obj)
    elif getattr(obj, '__closure__', None):
        candidates = [c.cell_contents for c in obj.__closure__]
    else:
        candidates = []

    found = []
    seen_ids = set()
    for item in candidates:
        if isinstance(item, cls) and id(item) not in seen_ids:
            seen_ids.add(id(item))
            found.append(item)

    if len(found) != 1:
        raise _AmbiguousObject(obj, cls, found)
    return found[0]


def jit_function_of(built):
    """Recover the `@flyc.jit` `JitFunction` a description's builder produced.

    1. `built` already IS the `JitFunction` -- the common case, 47 of 69
       builders in the FlyDSL tree return their `@flyc.jit` directly.
    2. Otherwise it is a wrapper closing over exactly one `JitFunction` -- the
       gfx1201 `_launch` host wrapper is this shape, and the ABI alignment
       (Step 0) did not change that: the `JitFunction` still lives in the
       closure shared by `_launch` and its `.compile` attribute (both are
       nested functions of the same builder, so they close over the same
       cells).
    3. Zero or several is a multi-kernel builder (3 known in the FlyDSL tree:
       `custom_all_reduce_kernel.make_allreduce_kernels`,
       `moe_sorting_kernel._compile_moe_sorting_multiphase`,
       `flash_attn_gfx950.build_flash_attn_dualwave_swp_module`). One builder,
       several hsacos, is a plural contract (`jit_functions_of`) this driver
       does not have -- raise rather than guess which one to pick.
    """
    from flydsl.compiler.jit_function import JitFunction

    if isinstance(built, JitFunction):
        return built
    try:
        return _find_unique(built, JitFunction)
    except _AmbiguousObject as e:
        if not e.found:
            raise RuntimeError(
                f"jit_function_of: no JitFunction reachable from {built!r}. "
                "The closure walk is the fragile part of this driver (see "
                "jit2aot-exec.md); do not broaden the search to make this "
                "pass -- report it, the fix is upstream in FlyDSL."
            ) from e
        raise RuntimeError(
            f"jit_function_of: found {len(e.found)} JitFunctions reachable "
            f"from {built!r}. That is a multi-kernel builder (one builder, "
            "several hsacos) -- out of scope for this driver, which drives "
            "one JitFunction per description. Needs a plural contract "
            "(jit_functions_of), not a guess at which one to pick."
        ) from e


def _launcher_signature(jf):
    """The `JitFunction`'s bound signature, resolved the way flydsl itself
    does -- NOT bare `inspect.signature`.

    `resolve_signature` (`flydsl.compiler.jit_argument`) is
    `inspect.signature(func, eval_str=True)`, and it is what
    `JitFunction._ensure_sig` binds against. This matters: four `@flyc.jit`
    files in the FlyDSL tree use `from __future__ import annotations`, so
    their annotations are strings at class-definition time -- bare
    `inspect.signature` would yield the string `'fx.Pointer'`, not the type,
    and every downstream annotation check would silently misfire.
    """
    from flydsl.compiler.jit_argument import resolve_signature

    return resolve_signature(jf.func)


def kernel_function_of(jf):
    """The `@flyc.kernel` `KernelFunction` a `JitFunction`'s launcher closes
    over -- reachable from `jf.func`'s closure, NOT from the builder's
    returned wrapper's (`kernel_function_of` walks `jf.func`, unlike
    `jit_function_of` which walks `built`; the two closures are not the
    same one)."""
    from flydsl.compiler.kernel_function import KernelFunction

    return _find_unique(jf.func, KernelFunction)


def _extract_hsaco(jf) -> bytes:
    """Pull the compiled hsaco ELF out of a traced `JitFunction`.

    `jf._last_compiled[1]._ir_text` is the linked MLIR after `jf(*args)` ran
    under `COMPILE_ONLY=1`. Two `gpu.binary` objects appear -- one per target
    on the `gpu.module`: the bare `#rocdl.target<chip=...>` FlyDSL's ROCm
    backend sets when it creates the module, and the `no_wave64` variant
    `rocdl-attach-target` appends. The first is the one shipped.

    They are two independent LLVM codegen runs, so they are *not* guaranteed
    byte-identical, and requiring it (as this used to) breaks the build: the
    AMDGPU backend is not bitwise reproducible. About one gfx1201
    `flyc_attn_fwd` `BLOCK_DMODEL=48` compile in five came out differing from
    its sibling -- benignly, e.g. a commutative VOP3's two source operands
    swapped -- and every such difference vanished under
    `setarch --addr-no-randomize`, i.e. it tracks address-space layout, not
    anything about the kernel.

    A difference is still worth knowing about, hence the warning rather than
    silence: the two targets are only equivalent as long as nothing lands on
    one and not the other. `rocdl-attach-target` is the only one of the two
    carrying the backend's configured options (`wave64`, `abi`, `O`, and
    `fast`/`unsafe-math` off the compile hints), so a hint that starts
    reaching it would show up here first -- as a warning on every compile
    rather than the occasional one.
    """
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu as gpud

    last = jf._last_compiled
    if last is None:
        raise RuntimeError("_extract_hsaco: jf._last_compiled is None; jf(*args) did not run")

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(last[1]._ir_text)
        blobs = []
        for op in module.body.operations:
            if op.operation.name == 'gpu.binary':
                blobs.extend(gpud.ObjectAttr(op.objects[i]).object for i in range(len(op.objects)))

    if not blobs:
        raise RuntimeError("_extract_hsaco: no gpu.binary op found in the compiled IR")
    if len(set(blobs)) != 1:
        print(
            f"_extract_hsaco: warning: the {len(blobs)} gpu.binary objects are not "
            f"byte-identical ({len(set(blobs))} distinct); shipping the first.",
            file=sys.stderr,
        )
    return blobs[0]


_RE_MACHINE = re.compile(r'Machine:\s*(\S+)')
_RE_FLAGS = re.compile(r'Flags:\s*(0x[0-9a-fA-F]+),\s*(\S+)')
_RE_KERNEL_NAME = re.compile(r'^\s*\.name:\s*(\S+)', re.MULTILINE)
_RE_GROUP_SEGMENT = re.compile(r'^\s*\.group_segment_fixed_size:\s*(\d+)', re.MULTILINE)


def _readelf_report(readelf: Path, hsaco_path: Path) -> str:
    result = subprocess.run(
        [str(readelf), '-h', '--notes', str(hsaco_path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def _elf_metadata(report: str) -> dict:
    """Machine/flags/kernel-symbol/LDS facts read back from the ELF itself.

    `--notes`' AMDGPU Metadata (a note the compiler embeds, not something this
    driver invents) already carries the kernel's public name and its LDS
    (`group_segment_fixed_size`) in one place, so no hand-rolled ELF/symtab
    parsing is needed for either.
    """
    machine = _RE_MACHINE.search(report)
    flags = _RE_FLAGS.search(report)
    kernel_name = _RE_KERNEL_NAME.search(report)
    shared = _RE_GROUP_SEGMENT.search(report)
    return dict(
        machine=machine.group(1) if machine else None,
        flags_hex=flags.group(1) if flags else None,
        flags_arch=flags.group(2) if flags else None,
        kernel_name=kernel_name.group(1) if kernel_name else None,
        shared=int(shared.group(1)) if shared else None,
    )


def _expected_gpu_symbol(node) -> str:
    """The HIP symbol the shim will ask `hipModuleGetFunction` for.

    Must match `ir/flyc/kdesc.py`'s `gpu_symbol_name`, which the generated shim
    embeds. Duplicated deliberately rather than imported: the two are computed in
    different processes at different times (the generator writes the shim at
    configure time; this driver compiles the kernel later), so they cannot share
    a value -- only a rule. `_verify_elf` is what stops the two copies of the
    rule from drifting apart silently.
    """
    return f'{node.kernel.__name__}_0'


def _verify_elf(meta: dict, target: str, report: str, node=None):
    if meta['machine'] != 'EM_AMDGPU':
        raise RuntimeError(f"--verify: expected Machine EM_AMDGPU, got {meta['machine']!r}. Full report:\n{report}")
    if meta['flags_arch'] != target:
        raise RuntimeError(
            f"--verify: ELF Flags name {meta['flags_arch']!r}, expected {target!r}. Full report:\n{report}"
        )
    if node is not None:
        want = _expected_gpu_symbol(node)
        if meta['kernel_name'] != want:
            raise RuntimeError(
                f"--verify: this hsaco exports {meta['kernel_name']!r}, but the "
                f"generated shim looks up {want!r}, so every launch would fail "
                f"in hipModuleGetFunction.\n\n"
                f"The shim's name comes from ir/flyc/kdesc.py's gpu_symbol_name: "
                f"the @flyc.kernel def's name plus FlyDSL's kernel id. A mismatch "
                f"means one of that rule's two assumptions no longer holds -- "
                f"either the kernel gained an explicit name= (FlyDSL's "
                f"_emit_kernel then uses it verbatim, with no id suffix), or a "
                f"single compilation now emits more than one kernel so the id is "
                f"no longer 0.\n\nFull report:\n{report}"
            )


def do_compile(args):
    stubbed = flyc_bootstrap.setup(args.target)
    if args.verbose and stubbed:
        print('flyc_bootstrap: installed the torch stub', file=sys.stderr)

    fn = _load_description_module(Path(args.path), args.kernel_name)
    node = fn.__ati_node__
    kernel_dir = str(Path(node.source_path).parent)
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    # `choices` is a MappingChoiceView over the plain dict parsed from
    # `--signature`: `{name: literal}`, nothing else, no fabricated `Functional`.
    # The driver runs in a separate process from the generator and has no linked
    # IR to hand the body a real one. Unlike the untyped stand-in this replaced,
    # `MappingChoiceView` (above) is a declared implementation of the
    # `ChoiceView` ABC (ir/choices.py), so a description written against that
    # interface reads the same here as it does on the generator side, and any
    # method it grows must be answered by both backings.
    choices = MappingChoiceView(parse_pon(args.signature, sep=' '))
    hints = _build_hints(node, args.hints)
    # The description body returns (built, sidecar): `built` is the FlyDSL
    # builder's result (driven to a code object below); `sidecar` is a
    # JSON-serialisable dict of whatever it wants recorded alongside the hsaco
    # (for flyc_attn_fwd, asdict(knobs) -- including block_m). The driver stays
    # kernel-agnostic: it serialises the dict without knowing what is in it.
    # The description returns a DEFERRED builder, not a built module: the code
    # generator calls `fn` for its knobs and never calls `build`, so it never
    # imports flydsl. Only this driver -- run by ninja -- calls `build()`.
    build, sidecar = fn(choices, hints)
    built = build()

    jf = jit_function_of(built)
    launch_args = synthesise_args(jf)
    jf(*launch_args)  # COMPILE_ONLY=1 -> traces and compiles, returns None, launches nothing
    hsaco = _extract_hsaco(jf)
    # BLOCK_SIZE is a *declared* value (`@flyc.kernel(known_block_size=...)`),
    # not something recovered from the artifact: the ELF's
    # `.max_flat_workgroup_size` is a bound, not the exact value, and
    # `.reqd_workgroup_size` is emitted empty, so the exact launch geometry
    # never makes it into the hsaco. flydsl itself validates the real launch
    # against this same value in `KernelLauncher._check_block_vs_known`, so
    # `_known_block_size` is the authoritative source, not a guess.
    block_size = kernel_function_of(jf)._known_block_size[0]

    out_path = args.out_path
    with open(out_path.with_suffix('.hsaco'), 'wb') as f:
        f.write(hsaco)

    import os
    readelf = Path(os.environ['ROCM_PATH']) / 'llvm' / 'bin' / 'llvm-readelf'
    report = _readelf_report(readelf, out_path.with_suffix('.hsaco'))
    meta = _elf_metadata(report)
    if args.verify:
        _verify_elf(meta, args.target, report, node)

    # aotriton.aks2's loader computes the AKS2 directory entry's block_threads
    # as `j['num_warps'] * j['warp_size']` -- the same key shape python/compile.py
    # (the Triton driver) writes. flyc has no num_warps of its own (FlyDSL's grid
    # model has no warp count knob), but block_size (the @flyc.kernel's declared
    # known_block_size, i.e. total threads per block) and warp_size==32 (fixed,
    # RDNA/CDNA wavefront size) determine it exactly: num_warps * warp_size ==
    # block_size by construction, so this is a derivation, not a guess.
    warp_size = 32
    assert block_size % warp_size == 0, (
        f'block_size={block_size} is not a multiple of warp_size={warp_size}; '
        f'cannot derive num_warps for the aks2 sidecar (aotriton.aks2 needs '
        f"j['num_warps'] * j['warp_size'] == block_threads)."
    )
    num_warps = block_size // warp_size

    di = {
        'compile_status': 'Complete',
        'kernel_name': meta['kernel_name'],
        'arch': args.target,
        'num_warps': num_warps,
        'warp_size': warp_size,
        'shared': meta['shared'],
        'signature': args.signature,
        'hints': args.hints,
        'sidecar': sidecar,
        # block_m rides in the sidecar dict (it is resolved and used by the
        # builder already; it just needed forwarding -- see flyc_attn_fwd.py).
        # block_size is NOT in the sidecar/knobs; it is the `@flyc.kernel`'s
        # declared known_block_size, read off the KernelFunction above.
        'block_m': sidecar.get('block_m') if isinstance(sidecar, dict) else None,
        'block_size': block_size,
    }
    with open(out_path.with_suffix('.json'), 'w') as f:
        json.dump(di, f, indent=2)
    return out_path


def ipc_compile(ipc_in, ipc_out):
    args = ipc_in.get()
    try:
        do_compile(args)
        ipc_out.put('Complete')
    except Exception as e:
        if args.verbose:
            print(e, file=sys.stderr)
        ipc_out.put('Exception')


def main():
    args = parse()
    if args.timeout <= 0:
        do_compile(args)
        return
    ipc_to_worker = Queue()
    ipc_worker_out = Queue()
    ipc_to_worker.cancel_join_thread()
    ipc_worker_out.cancel_join_thread()
    worker = Process(target=ipc_compile, args=(ipc_to_worker, ipc_worker_out))
    worker.start()
    ipc_to_worker.put(args)
    worker.join(args.timeout * 60.0)
    if worker.exitcode == 0:
        status = ipc_worker_out.get()
    elif worker.exitcode is None:
        worker.kill()
        status = 'Timeout'
    else:
        status = 'ExitWithError'
    if status == 'Timeout':
        print(
            f'Compiling {args.path=} {args.kernel_name} to {args.out_path=} '
            f'timed out with {args.timeout} minutes', file=sys.stderr,
        )
    ipc_to_worker.close()
    ipc_worker_out.close()
    if args.verbose and status == 'ExitWithError':
        print(
            f'Compiling {args.path=} {args.kernel_name} to {args.out_path=} '
            f'result with status {status} exitcode {worker.exitcode}',
        )
    if status != 'Complete':
        with open(args.out_path.with_suffix('.hsaco'), 'bw'):
            pass
        with open(args.out_path.with_suffix('.json'), 'w') as f:
            json.dump({'compile_status': status}, f, indent=2)


if __name__ == "__main__":
    main()
