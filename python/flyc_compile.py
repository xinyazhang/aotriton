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
`@ati.flyc.kernel` description through `fn.__ati_node__` (`module_path`,
`hints()`) and the plain `fn(functional, hints) -> built` call — it does not
import a specific kernel family's tuning module, and nothing here names a
specific kernel. The one deliberate exception is `_trace_fmha_launch`, which
knows the FlyDSL-attention ABI (`fmha_abi_gfx1201.run_compiled`, the launcher's
positional argument shape); see its docstring for why that one function is not
yet generic and what would make it so.
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
from .utils import parse_kv

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


class _Choices:
    """Attribute view over the parsed `--signature` dict.

    Mirrors `ir.Functional.choices` well enough for a description body to read
    `f.choices.NAME` without caring whether `f` is this stand-in or the real
    linked IR object.
    """

    def __init__(self, values: dict):
        self.__dict__.update(values)


class _FunctionalStandIn:
    """Phase 1 stand-in for `ir.Functional`: `.arch` and `.choices.<NAME>` only.

    The driver runs in a separate process from the generator and has no
    linked IR to hand a description body, so this exposes the same attribute
    surface a description already reads from the real `ir.Functional` --
    Phase 2 can pass the genuine object with no change to any description.
    """

    def __init__(self, arch: str, choices: dict):
        self.arch = arch
        self.choices = _Choices(choices)


def _build_hints(node, hints_str: str):
    """The `@ati.flyc.hints` dataclass instance: `node`'s defaults, `--hints`
    overrides applied on top. Rejects unknown fields loudly -- a typo in
    `--hints` must not silently build the default schedule."""
    defaults = node.hints()
    if not hints_str:
        return defaults
    overrides = parse_kv(hints_str, sep=' ')
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

    **`fx.Tensor` is deliberately NOT supported yet** -- see the assertion in
    `_assert_supported_operands`. What is actually known is one measurement,
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


# One fixed BHSD shape for every functional: `FakeTensor`'s docstring above
# explains why the actual numbers do not matter to the compiled artifact.
_FAKE_SHAPE = (1, 1, 128, 64)


_SUPPORTED_OPERAND_ANNOTATIONS = frozenset(
    {'Pointer', 'Int32', 'Int64', 'Float32', 'Stream'}
)


def _assert_supported_operands(jf):
    """Refuse a launcher whose signature this driver cannot honestly synthesise.

    A loud stop rather than a guess. `fx.Tensor` is the case that matters and the
    one to expect: it needs a real operand descriptor (rank, extents, dtype), the
    tensor kernarg ABI is not pinned down, and silently marshalling something
    plausible would produce an artifact whose kernarg layout we cannot verify.
    See `FakeTensor`'s docstring for what is known and what is not.
    """
    import inspect

    bad = []
    for p in inspect.signature(jf.func).parameters.values():
        name = getattr(p.annotation, '__name__', str(p.annotation))
        if name not in _SUPPORTED_OPERAND_ANNOTATIONS:
            bad.append((p.name, name))
    if not bad:
        return
    detail = ', '.join(f'{n}: fx.{a}' for n, a in bad)
    tensor_note = ''
    if any(a == 'Tensor' for _, a in bad):
        tensor_note = (
            "\n\nfx.Tensor operands are not supported yet -- deliberately, not by "
            "oversight. Each one adds a 40-byte by-value memref descriptor "
            "interleaved after its pointer, and the contents of those bytes, the "
            "static/dynamic layout question, and how the C++ shim fills them are "
            "all unverified. See FakeTensor's docstring in this file."
        )
    raise NotImplementedError(
        f"{jf.func.__name__}: unsupported operand annotation(s): {detail}."
        f"{tensor_note}"
    )


def _trace_fmha_launch(built, functional):
    """Trace `built` (the FlyDSL-attention launcher) and return `(jf, args)`.

    **Not kernel-agnostic** -- the one place in this driver that is not.
    `built` is a plain Python closure (`_launch` in
    `flash_attn_func_gfx1201_aiw.py`); nothing about the `@ati.flyc.kernel`
    contract exposes the compiled `JitFunction` it closes over or the launch
    argument shape it expects, so this recovers both the FlyDSL-attention way:
    monkeypatch `fmha_abi_gfx1201.run_compiled` (the function every such
    launcher calls to dispatch) to record its arguments instead of compiling,
    then call `built` with `FakeTensor` operands.

    The real fix is an upstream AOT entry point that returns `(jf, args)`
    directly, not a cleverer driver (`PLAN.md` open question 2). Until then,
    isolating the FlyDSL-attention specifics here is what keeps everything
    else in this file usable by a future non-attention `@ati.flyc.kernel`.
    """
    import fmha_abi_gfx1201 as abi

    cap = {}
    abi.run_compiled = lambda cache, exe, *a: cap.update(exe=exe, args=a)

    q = FakeTensor(_FAKE_SHAPE)
    k = FakeTensor(_FAKE_SHAPE)
    v = FakeTensor(_FAKE_SHAPE)
    o = FakeTensor(_FAKE_SHAPE)
    batch_size, seqlen_q = _FAKE_SHAPE[0], _FAKE_SHAPE[2]

    # CAUSAL_TYPE=3 (generalized sliding-window) has no sentinel window
    # (`fmha_abi_gfx1201.CAUSAL_SENTINEL` only covers 1/2) and
    # `abi.resolve_window` raises unless the caller supplies an explicit
    # `window=(left, right)`. window_left/window_right are runtime kernel
    # arguments (fx.Int32), not baked into the compiled artifact, so any
    # valid bound traces the same ELF; `(seqlen_q, 0)` -- top-left causal --
    # is the arbitrary-but-valid choice, matching what the ValueError itself
    # suggests.
    causal_type = getattr(functional.choices, 'CAUSAL_TYPE', 0)
    window = (seqlen_q, 0) if causal_type == 3 else None

    # philox_seed=None: `u64_scalar` short-circuits on None rather than
    # allocating a torch tensor, which the build venv must never do
    # (CMakeLists.txt:142) and which this driver has no device for anyway.
    built(q, k, v, o, batch_size, seqlen_q, window=window, philox_seed=None)
    if 'exe' in cap:
        _assert_supported_operands(cap['exe'])
    if 'exe' not in cap:
        raise RuntimeError(
            "_trace_fmha_launch: built(...) never reached fmha_abi_gfx1201.run_compiled; "
            "is this description's builder still the FlyDSL-attention launcher shape?"
        )
    return cap['exe'], cap['args']


def _extract_block_size(source_ir: str):
    """Recover BLOCK_SIZE from the PRE-LOWERING IR's `gpu.func` `known_block_size`
    attribute (`array<i32: N, 1, 1>`; N is BLOCK_SIZE).

    Not in the knobs: `resolve_knobs` leaves `flat_work_group_size = None` and the
    builder derives `BLOCK_SIZE = FLAT_WORK_GROUP_SIZE or NUM_WAVES * WARP_SIZE`
    internally. Not in `_ir_text` either -- `gpu-module-to-binary` has replaced the
    module body by then. Not in the ELF -- block size is a host launch decision
    never baked into the binary. `CompiledArtifact._source_ir` is the one place it
    still exists as MLIR text.
    """
    from flydsl._mlir import ir

    def _find(op):
        if op.operation.name == 'gpu.func':
            attr = op.operation.attributes.get('known_block_size')
            if attr is not None:
                return int(attr[0])
        for region in op.operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    found = _find(inner)
                    if found is not None:
                        return found
        return None

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source_ir)
        for op in module.body.operations:
            found = _find(op)
            if found is not None:
                return found
    return None


def _extract_hsaco(jf) -> bytes:
    """Pull the compiled hsaco ELF out of a traced `JitFunction`.

    `jf._last_compiled[1]._ir_text` is the linked MLIR after `jf(*args)` ran
    under `COMPILE_ONLY=1`. Exactly two `gpu.binary` objects are expected --
    the `#rocdl.target<chip=...>` one and the `no_wave64` variant the
    `rocdl-attach-target` pass adds -- and they are byte-identical; asserting
    that is cheap insurance that nothing downstream silently started emitting
    two different code objects.
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
        raise RuntimeError(
            f"_extract_hsaco: expected every gpu.binary object to be byte-identical, "
            f"got {len(set(blobs))} distinct blob(s) among {len(blobs)}"
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


def _verify_elf(meta: dict, target: str, report: str):
    if meta['machine'] != 'EM_AMDGPU':
        raise RuntimeError(f"--verify: expected Machine EM_AMDGPU, got {meta['machine']!r}. Full report:\n{report}")
    if meta['flags_arch'] != target:
        raise RuntimeError(
            f"--verify: ELF Flags name {meta['flags_arch']!r}, expected {target!r}. Full report:\n{report}"
        )


def do_compile(args):
    stubbed = flyc_bootstrap.setup(args.target)
    if args.verbose and stubbed:
        print('flyc_bootstrap: installed the torch stub', file=sys.stderr)

    fn = _load_description_module(Path(args.path), args.kernel_name)
    node = fn.__ati_node__
    kernel_dir = str(Path(node.module_path).parent)
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)

    functional = _FunctionalStandIn(args.target, parse_kv(args.signature, sep=' '))
    hints = _build_hints(node, args.hints)
    # The description body returns (built, sidecar): `built` is the FlyDSL
    # builder's result (driven to a code object below); `sidecar` is a
    # JSON-serialisable dict of whatever it wants recorded alongside the hsaco
    # (for flyc_attn_fwd, asdict(knobs) -- including block_m). The driver stays
    # kernel-agnostic: it serialises the dict without knowing what is in it.
    built, sidecar = fn(functional, hints)

    jf, launch_args = _trace_fmha_launch(built, functional)
    jf(*launch_args)  # COMPILE_ONLY=1 -> traces and compiles, returns None, launches nothing
    hsaco = _extract_hsaco(jf)
    block_size = _extract_block_size(jf._last_compiled[1]._source_ir)

    out_path = args.out_path
    with open(out_path.with_suffix('.hsaco'), 'wb') as f:
        f.write(hsaco)

    import os
    readelf = Path(os.environ['ROCM_PATH']) / 'llvm' / 'bin' / 'llvm-readelf'
    report = _readelf_report(readelf, out_path.with_suffix('.hsaco'))
    meta = _elf_metadata(report)
    if args.verify:
        _verify_elf(meta, args.target, report)

    di = {
        'compile_status': 'Complete',
        'kernel_name': meta['kernel_name'],
        'arch': args.target,
        'warp_size': 32,
        'shared': meta['shared'],
        'signature': args.signature,
        'hints': args.hints,
        'sidecar': sidecar,
        # block_m rides in the sidecar dict (it is resolved and used by the
        # builder already; it just needed forwarding -- see flyc_attn_fwd.py).
        # block_size is NOT in the sidecar/knobs; it is recovered from the
        # pre-lowering IR above (see _extract_block_size).
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
