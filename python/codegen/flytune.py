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
# plain dict already known by the time `fn(arch, choices, hints)` returns.
# This generator calls `fn(arch, choices, hints)` for `knobs` (plus two plain
# string attributes off `build` -- `flyc_source`/`flyc_kernel_name`, item D)
# and discards `build` without ever CALLING it -- so the FlyDSL compiler is
# never invoked 288 times per configure. `knobs` is what feeds
# `ir/flyc/ksignature.py`'s `perf_section` (Task 2); the true on-disk
# `<hsaco>.json` knobs is a separate, later, build-time artifact this
# generator never reads.

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
        self._sigs = list(self._gen_signatures())

    def _gen_signatures(self):
        """Yield one KernelSignature per compiled image of this functional.

        A generator, not a single value, because that is the shape flyc autotune
        needs: once FlyDSL's schedule becomes `hints`-dependent, one
        (choices, hints) resolves to SEVERAL (knobs, build) pairs and each one is
        an image with its own `#P`. Today the count is one; nothing here assumes
        it, so growing it is adding a loop, not reshaping the class.
        """
        f = self._f
        kdesc = f.meta_object
        # `f.choices` (a FunctionalChoiceView) is the real thing, not a dict
        # rebuilt from it: this generator has a linked Functional, unlike
        # python/flyc_compile.py's build-time driver, which parses `--signature`
        # text into a MappingChoiceView instead (defined there; the interface
        # they share is ir/choices.py's ChoiceView). The
        # description reads scalar axes by attribute (`choices.BLOCK_DMODEL`)
        # and real arguments via `.arg(aname)` (`choices.arg('Q')`) -- the same
        # split root.py:write_flyc_hsaco's `--signature` string preserves
        # (rendered from `compact_choices`, unaffected by this).
        choices = f.choices
        hints = kdesc.hints()
        assert kdesc.builder_fn is not None, (
            f'flyc kernel {kdesc.NAME!r} has no builder_fn '
            f'(linker.py:_build_flycs must thread builder_fn=decl.fn)')
        # The description imports its vendored kernel/tuning modules by bare name
        # (e.g. `import fmha_tuning_gfx1201`), resolved relative to
        # kdesc.source_path -- the vendored flyc DIRECTORY itself (item D; it
        # used to be a specific kernel FILE's parent). A path, not flydsl --
        # fine for the generator.
        kernel_dir = str(kdesc.source_path)
        if kernel_dir not in sys.path:
            sys.path.insert(0, kernel_dir)
        # (build, knobs): `build` is a deferred callable that would construct the
        # FlyDSL module; `knobs` is already resolved. Discard `build` WITHOUT EVER
        # CALLING IT -- the one architectural rule this generator must not break.
        # `f.arch` is now the builder's first argument (item F): every flyc
        # description takes (arch, choices, hints), arch arriving first, not
        # smuggled into choices.
        build, knobs = kdesc.builder_fn(f.arch, choices, hints)
        # `build.flyc_source`/`build.flyc_kernel_name` (item D) are the only
        # two attributes read off `build` -- never call it (see above).
        kdesc.ensure_stub_resolved(f.arch, build)
        yield KernelSignature(f, psels=knobs)

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
        meta_hsacos = self.codegen_compact_kernels(self._sigs, flatzip_path)
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

    def codegen_compact_kernels(self, ksigs, flatzip_path):
        """One TritonKernelCompactMeta row per image, same shape as
        autotune.py's. Plural for the same reason `_gen_signatures` is a
        generator: N is 1 today and nothing here says so."""
        string_registry = self._parent_repo.get_string_registry('per_kernel_packed_string')
        rows = []
        for ksig in ksigs:
            b2sum_u64, raw = ksig.blake2b_hash(flatzip_path)
            u8raw = raw.decode('utf-8')
            assert len(b2sum_u64) == 16
            b2sum_u64_hi = b2sum_u64[:8]
            b2sum_u64_lo = b2sum_u64[8:]
            psel_offset = string_registry.register(ksig.perf_section)
            copt_offset = string_registry.register(ksig.copt_section)
            rows.append(f'{{ 0x{b2sum_u64_hi}u, 0x{b2sum_u64_lo}u, {psel_offset}, {copt_offset} }}, '
                        f'// {b2sum_u64} = b2sum -l 64 <<< {u8raw}')
        ALIGN = '\n' + 4 * ' '
        return ALIGN.join(rows)

    def codegen_deduplicated_pp_args_function_index(self):
        """Unlike autotune.py's equivalent, flyc collapses every functional onto
        ONE shared pp_args function (PLAN-PHASE2.md Task 5): there is no
        constexpr baking (no assign_skips[i] ever True), so assign_skips is
        always the all-False tuple of the same length, and the
        SignaturedFunctionRegistry naturally deduplicates it to a single
        registration across all 288 functionals.

        item C (mirroring autotune.py's constexpr fold via
        `Functional.pp_arg_doc`) was implemented and build-verified, then
        REVERTED: autotune.py's fold is only sound for Triton because
        `tl.constexpr` parameters are genuinely elided from the JIT-compiled
        kernel's ABI, so a shorter pp_args vector matches that functional's
        differently-compiled binary. flyc's vendored kernel functions are
        fixed, static `@flyc.kernel`-decorated Python signatures (grep-
        confirmed for `flash_attn_func_aiw_kernel` in
        flash_attn_func_gfx1201_aiw.py and `bwd_dq_kernel` in
        fmha_bwd_dq_gfx1201_kernel.py): `window_left`, `window_right`,
        `philox_offset2`, `hdim_qk`, `hdim_vo` are ALWAYS formal `fx.Int32`/
        `fx.Int64` parameters of the compiled kernel, resolved at RUNTIME
        inside the kernel body (e.g. via `fmha_common_gfx1201.resolve_window`)
        -- never elided from the kernarg ABI based on the operator
        description's `@ati.derives`/VarRef-driven constexpr-ness. flyc's
        `real_param_order` is a static, arch-wide (not per-functional) AST
        parse of that fixed signature, and `flyc_compile.py`'s
        `synthesise_args` builds trace placeholders from parameter
        ANNOTATION TYPE alone, never from the functional's resolved value --
        there is no per-functional ABI specialization to match a folded
        vector against. Commenting an entry out of the return vector would
        therefore desync the `std::vector<void*>` positionally against the
        compiled hsaco's actual, unchanging `hipModuleLaunchKernel`
        kernelParams layout for every OTHER functional sharing that pattern.
        This directly contradicts the plan's own claim that
        `pp_arg_doc`/`Functional.resolved` make the fold "no new mechanism"
        for flyc; the plan's stated EXPECTED outcome ("gfx1201's all-False
        pattern") is the one this reverted, no-fold implementation actually
        produces. See the Phase A execution report for the full writeup.

        Context helpers (if any) are populated by a preamble right before the
        return statement -- their return value has no other stable home for
        pp_args's `const context&` signature to take the address of (see
        ir/flyc/kdesc.py's iter_context_helpers / iter_launch_arguments)."""
        kdesc = self._f.meta_object
        pp_registry = self._parent_repo.get_signatured_function_registry('pp_function')
        largs = list(kdesc.iter_launch_arguments(self._f.arch))
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
            stmt.append(f'context.scratch_params.{name} = context.{name}();')
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
        return self._sigs
