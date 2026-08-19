# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/flytune.<kernel_name>/<functional>.cc
#
# The flyc analogue of AutotuneCodeGenerator (autotune.py), but degenerate: flyc
# has no LUT, no binning, no PerfFields struct, and (Phase 2) exactly ONE hsaco
# candidate per surviving functional -- there is no autotune choice among several
# compiled images the way Triton's shim has.
#
# The generator must NEVER invoke the FlyDSL builder / compiler -- only
# `python/flyc_compile.py`, run by ninja at build time, may do that. A flyc
# description's builder function returns `(build, knobs)`
# (`modules/flash/aot/flyc_attn_fwd.py`): `build` is a deferred callable that
# actually constructs the FlyDSL module (and, for the standalone
# `flyc_compile.py` driver, transitively imports flydsl), while `knobs` is a
# plain dict already known by the time `fn(choices, hints)` returns. This
# generator calls `fn(choices, hints)` for `knobs` alone and discards `build`
# without ever calling it -- so the FlyDSL compiler is never invoked 288 times
# per configure. `knobs` is what feeds `ir/flyc/ksignature.py`'s
# `perf_section` (Task 2); the true on-disk `<hsaco>.json` sidecar is a
# separate, later, build-time artifact this generator never reads.

import sys
from pathlib import Path

from ..template_instantiation.ir import Functional
from ..template_instantiation.ir.flyc import KernelSignature
from .template import get_template
from ..utils import LazyFile, log
from .basetune import BaseTuneCodeGenerator


class FlycTuneCodeGenerator(BaseTuneCodeGenerator):
    FLYTUNE_TEMPLATE = get_template('flytune_table_entry.cc')

    def __init__(self,
                 args,
                 f : Functional,
                 dataframe_for_tuning : 'pandas.DataFrame | None',
                 sql : tuple,
                 parent_repo):
        super().__init__(args, f, dataframe_for_tuning, parent_repo)
        self._sql = sql
        kdesc = self._f.meta_object
        # Replicate flyc_compile.py's do_compile call, up to the point where it
        # would build: choices is a plain dict of the functional's resolved
        # axis values (keyed by the SAME semantic axis names -- Q,
        # BLOCK_DMODEL, CAUSAL_TYPE, ... -- the builder body reads), hints is
        # the description's declared @ati.flyc.hints dataclass (no CLI
        # override exists at generate time).
        choices = {name: tc.triton_compile_signature for name, tc in f.resolved.items()}
        hints = kdesc.hints()
        assert kdesc.builder_fn is not None, (
            f'flyc kernel {kdesc.NAME!r} has no builder_fn '
            f'(linker.py:_build_flycs must thread builder_fn=decl.fn)')
        # The builder body (e.g. modules/flash/aot/flyc_attn_fwd.py) imports its
        # vendored kernel/tuning modules by bare name (e.g. `import
        # fmha_tuning_gfx1201`), resolved relative to the directory containing
        # kdesc.MODULE_PATH -- not relative to this generator's own package.
        # This is a path, not flydsl, so it is fine for the generator to set up.
        kernel_dir = str(Path(kdesc.MODULE_PATH).parent)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)
        # fn(choices, hints) returns (build, knobs): `build` is a deferred
        # callable that actually constructs the FlyDSL module (and, for the
        # standalone flyc_compile.py driver, imports flydsl transitively);
        # `knobs` is a plain, already-resolved dict. Discard `build` WITHOUT
        # EVER CALLING IT -- that is the one architectural rule this generator
        # must never break.
        _build, sidecar = kdesc.builder_fn(choices, hints)
        self._sig = KernelSignature(f, sidecar=sidecar)

    def generate(self):
        log(lambda : f'Writing to {self._cc_file}')
        with LazyFile(self._cc_file) as fout:
            self.write_flytune_src(fout)
        hsaco_registry = self._parent_repo.get_hsaco_registry('hsaco')
        hsaco_registry.register(self._f, self.all_signatures)

    def write_flytune_src(self, fout):
        f = self._f
        kdesc = f.meta_object
        flatzip_path = f.full_flatzip_path.as_posix()
        assert f.filepack_inzip_name == f.unified_signature
        meta_hsacos = self.codegen_compact_kernels(self._sig, flatzip_path)
        d = {
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'godel_number'          : f.godel_number,
            'flatzip_path'          : flatzip_path,
            'func_name'             : f.unified_signature,
            'arch_name'             : f.arch,
            'meta_hsacos'           : meta_hsacos,
            'context_class_name'    : kdesc.context_class_name,
            'deduplicated_pp_args_function_index' : self.codegen_deduplicated_pp_args_function_index(),
            'arch_number'           : f.arch_number,
            'human_readable_signature' : f.human_readable_signature,
            'sql'                   : self._sql,
        }
        print(self.FLYTUNE_TEMPLATE.format_map(d), file=fout)

    def codegen_compact_kernels(self, ksig, flatzip_path):
        string_registry = self._parent_repo.get_string_registry('per_kernel_packed_string')
        b2sum_u64, raw = ksig.blake2b_hash(flatzip_path)
        u8raw = raw.decode('utf-8')
        assert len(b2sum_u64) == 16
        b2sum_u64_hi = b2sum_u64[:8]
        b2sum_u64_lo = b2sum_u64[8:]
        psel_offset = string_registry.register(ksig.perf_section)
        copt_offset = string_registry.register(ksig.copt_section)
        return (f'{{ 0x{b2sum_u64_hi}u, 0x{b2sum_u64_lo}u, {psel_offset}, {copt_offset} }}, '
                f'// {b2sum_u64} = b2sum -l 64 <<< {u8raw}')

    def codegen_deduplicated_pp_args_function_index(self):
        """Unlike autotune.py's equivalent, flyc collapses every functional onto
        ONE shared pp_args function (PLAN-PHASE2.md Task 5): there is no
        constexpr baking (no assign_skips[i] ever True), so assign_skips is
        always the all-False tuple of the same length, and the
        SignaturedFunctionRegistry naturally deduplicates it to a single
        registration across all 288 functionals. Context helpers (if any) are
        populated by a preamble right before the return statement -- their
        return value has no other stable home for pp_args's `const context&`
        signature to take the address of (see ir/flyc/kdesc.py's
        iter_context_helpers / iter_launch_arguments)."""
        kdesc = self._f.meta_object
        pp_registry = self._parent_repo.get_signatured_function_registry('pp_function')
        largs = list(kdesc.iter_launch_arguments())
        assign_skips = (False,) * len(largs)
        hit, findex = pp_registry.contains(assign_skips)
        if hit:
            return findex
        stmt = []
        for name, ctype in kdesc.iter_context_helpers():
            # The helper is a member function of the context (hand-implemented
            # in modules/<family>/csrc/<kernel>.cc, PLAN-PHASE2.md Task 6, out of
            # scope for Tasks 1-5): the generator only declares
            # `<ctype> <name>() const;` on the context (flyc.h's
            # context_helper_declares slot). Calling it through `context.` here
            # keeps pp_args a free function while still reading the context's
            # own params/tensors.
            stmt.append(f'context._{name}_scratch = context.{name}();')
        ret_lines = [larg.expr + f', // {larg.aname}' for larg in largs]
        pfx = '  return { '
        join = '\n' + ' ' * len(pfx)
        sfx = '         };'
        # Do NOT join the return-vector lines with ','. There is comment text
        # after each parameter.
        stmt.append(pfx + join.join(ret_lines) + '\n' + sfx)
        src = '\n  '.join(stmt)
        return pp_registry.register(assign_skips, src)

    @property
    def all_signatures(self):
        return [self._sig]
