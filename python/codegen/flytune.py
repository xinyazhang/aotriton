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
        very few shared pp_args functions (PLAN-PHASE2.md Task 5), rather than
        one per compiled variant: the SignaturedFunctionRegistry deduplicates
        on `assign_skips`, and on gfx1201 that tuple is always all-False, so
        every one of gfx1201's functionals still shares a single registration.

        item C (mirroring autotune.py's constexpr fold via a
        `pp_arg_doc`-shaped `(is_constexpr, comment_value)` lookup) was
        implemented once for the GENERAL, per-functional case, build-
        verified, then REVERTED: autotune.py's fold is only sound for Triton
        because `tl.constexpr` parameters are genuinely elided from the
        JIT-compiled kernel's ABI on a PER-FUNCTIONAL basis, so a shorter
        pp_args vector matches that functional's differently-compiled
        binary. flyc's vendored kernel functions are fixed, static
        `@flyc.kernel`-decorated Python signatures (grep-confirmed for
        `flash_attn_func_aiw_kernel` in flash_attn_func_gfx1201_aiw.py and
        `bwd_dq_kernel` in fmha_bwd_dq_gfx1201_kernel.py): `window_left`,
        `window_right`, `philox_offset2`, `hdim_qk`, `hdim_vo` are ALWAYS
        formal `fx.Int32`/`fx.Int64` parameters of the compiled kernel,
        resolved at RUNTIME inside the kernel body (e.g. via
        `fmha_common_gfx1201.resolve_window`) -- never elided from the
        kernarg ABI based on the operator description's `@ati.derives`/
        VarRef-driven constexpr-ness, which is what the reverted, general
        `Functional.pp_arg_doc`-based fold would have kept baking per
        functional. flyc's `real_param_order` is a static, arch-wide (not
        per-functional) AST parse of that fixed signature, and
        `flyc_compile.py`'s `synthesise_args` builds trace placeholders from
        parameter ANNOTATION TYPE alone, never from the functional's
        resolved value -- there is no per-functional ABI specialization to
        match a PER-FUNCTIONAL-VARYING folded vector against for those axes.

        What is now (re-)implemented is a NARROWER fold that does not have
        that problem: `kdesc.pp_arg_doc(aname)` (`ir/flyc/kdesc.py`) answers
        is_constexpr from THIS kernel's own `@ati.scalar([...], options=...)`
        declaration for a real kernel argument, not from the functional's
        resolved axis choice. For an argument like gfx950's `Workspace`/
        `BlockTable`/`block_table_stride` (see `flyc_attn_fwd.py`'s
        `@ati.scalar(['Workspace', 'BlockTable', 'block_table_stride'],
        options=[0])`), that declared constexpr-ness is a build-config-wide
        fact -- true uniformly for every functional, because every gfx950
        build here pins `paged=False, num_kv_splits=1` -- so folding it out
        of the shared pp_args vector desyncs nothing: every functional this
        pp_args registration is shared across agrees the parameter is
        skipped. The general, per-functional-varying case above remains
        unimplemented and reverted; only this uniform, per-description case
        is folded. `flyc_compile.py`'s `_operand_for`/`_expected_kernarg_size`
        independently confirm (from the real, `eval_str=True`-resolved
        `Constexpr` annotation) that FlyDSL itself elides the same
        parameters from the compiled kernarg ABI -- this fold and that
        confirmation are two separate mechanisms that must agree, not one
        deriving the other; item H's kernarg-size check is what catches it
        if they ever stop agreeing.

        Context helpers (if any) are populated once, in `lookup_optimal()`,
        BEFORE pp_args ever runs (item I, PLAN-PHASE2.md Task 5 option (b);
        see codegen/flyc.py's codegen_context_helper_evaluate). pp_args here
        only READS `context.scratch_params.<name>` -- via
        iter_launch_arguments's 'context_helper' LaunchArg kind -- it does not
        populate it. This used to be pp_args's job (a preamble right before
        the return statement, evaluated once per deduplicated pp_args
        registration rather than once per description), which double-counted
        nothing today only because there is exactly one pp_args registration
        per description in Phase 2 -- but was the wrong home regardless:
        `iter_context_helpers` is per-description, and pp_args is
        per-functional and deduplicated, not per-description."""
        kdesc = self._f.meta_object
        pp_registry = self._parent_repo.get_signatured_function_registry('pp_function')
        largs = list(kdesc.iter_launch_arguments(self._f.arch))
        # IR-neutral, flyc-shaped: kdesc.pp_arg_doc(aname) -> (is_constexpr,
        # comment_value), sourced from THIS kernel's own @ati.scalar
        # declarations (see ir/flyc/kdesc.py's pp_arg_doc for why it cannot
        # delegate to Functional.resolved the way autotune.py's does).
        doc = {larg.aname: kdesc.pp_arg_doc(larg.aname) for larg in largs}
        assign_skips = tuple(doc[larg.aname][0] for larg in largs)
        hit, findex = pp_registry.contains(assign_skips)
        if hit:
            return findex
        ret_lines = []
        for larg in largs:
            is_constexpr, comment_value = doc[larg.aname]
            line = larg.expr + f', // {larg.aname}'
            # Comment out constexpr values -- see the docstring above for why
            # this is safe here (a uniform, per-description fact) but is not
            # the same thing as the general, per-functional fold that stays
            # reverted.
            if is_constexpr:
                line = '// ' + line + f' as constexpr {comment_value}'
            ret_lines.append(line)
        pfx = '  return { '
        join = '\n' + ' ' * len(pfx)
        sfx = '         };'
        # Do NOT join the return-vector lines with ','. There is comment text
        # after each parameter.
        src = pfx + join.join(ret_lines) + '\n' + sfx
        return pp_registry.register(assign_skips, src)

    @property
    def all_signatures(self):
        return self._sigs
