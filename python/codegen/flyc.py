# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/flyc.<kernel_name>.{h,cc}
#
# The flyc analogue of kernel.py's KernelShimGenerator. Mirrors it closely, with
# two deltas mandated by PLAN-PHASE2.md:
#   (1) pp_args functions take `(const {context_class_name}& context)`, not
#       `(const {param_class_name}& params, const TritonAuxiliaryArguments& aux)`
#       -- flyc kernels take exactly their real parameters, none of them the two
#       trailing scratch pointers Triton's own kernels take (Task 4, delta 1).
#       A `const auto& params = *context.params;` alias line is prepended so the
#       registered pp_args source (which still refers to `params.*` for
#       tensor/scalar operands) keeps working unmodified.
#   (2) flyc.h declares per-kernel "context helpers" (host-side computations
#       hand-implemented in csrc/, Task 6, out of scope here) as const member
#       functions on the context struct, plus mutable scratch members that
#       cache their return values for pp_args to take the address of (Task 5).

from ..template_instantiation.ir import Functional
from .interface import InterfaceGenerator
from .template import get_template
from ..utils import LazyFile, log
from .common import codegen_struct_cfields, codegen_includes
from .flytune import FlycTuneCodeGenerator


# TODO(de-duplication): this generator and codegen/kernel.py's
# KernelShimGenerator are substantially parallel, as are the flyc.{h,cc}
# templates against shim.{h,cc} (measured: ~34 of ~200 lines differ in the .cc,
# ~31 of ~110 in the .h). The duplication is deliberate for now -- flyc's shape
# was still moving -- but it is real debt: a fix or feature applied to one side
# silently misses the other. The `kctl.control_bits` handling is the concrete
# example, appearing three times in each template.
#
# Unify once flyc stops moving. The divergences are all parameterisable: the
# header name, the PP_FUNC signature (flyc has no TritonAuxiliaryArguments),
# one Pon line in lookup_optimal, and the tune namespace.

class FlycShimGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('flyc.h')
    SOURCE_TEMPLATE = get_template('flyc.cc')
    PFX = 'flyc'

    def create_sub_generator(self, functional: Functional, df: 'pandas.DataFrame', sql: tuple):
        if functional.meta_object.is_functional_disabled(functional):
            log(lambda: f'Functional {functional.godel_number=} disabled')
            return None, False
        return FlycTuneCodeGenerator(self._args, functional, df, sql, self._this_repo), True

    def write_shim_header(self, functionals, fout):
        kdesc = self._iface
        shared_iface = kdesc.SHARED_IFACE is not None
        if shared_iface:
            self._add_iface_for_source(kdesc.SHARED_IFACE)
        shared_iface_family = kdesc.SHARED_IFACE.FAMILY if shared_iface else kdesc.FAMILY
        d = {
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'flyc_gpu_symbol'       : kdesc.gpu_symbol_name,
            'param_class_name'      : kdesc.param_class_name,
            'shared_iface_family'   : shared_iface_family,
            'shared_iface'          : 1 if shared_iface else 0,
            'call_options_struct'   : kdesc.SHARED_IFACE.CALL_OPTIONS_NAME if shared_iface else 'void',
            'context_class_name'    : kdesc.context_class_name,
            'metadata_class_name'   : kdesc.metadata_class_name,
            'func_fields'           : codegen_struct_cfields(kdesc.func_cfields, nalign=4),
            'context_helper_declares'        : self.codegen_context_helper_declares(),
            'context_helper_scratch_members' : self.codegen_context_helper_scratch_members(),
            'declare_compiled_in_features'  : self.codegen_declare_compiled_in_features(),
            'kernel_table_entry_declares'   : self.codegen_tune_table_entry_declares(functionals),
            'number_of_functionals' : kdesc.godel_number,
            'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._hdr_include_repo.get_data())
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        kdesc = self._iface
        shared_iface = kdesc.SHARED_IFACE is not None
        shared_iface_family = kdesc.SHARED_IFACE.FAMILY if shared_iface else kdesc.FAMILY
        list_of_pp_args_function_defs, list_of_pp_args_function_decls, pp_func_num = self.codegen_kernel_arguments()
        d = {
            'shared_iface'        : 1 if shared_iface else 0,
            'shared_iface_family' : shared_iface_family,
            'call_options_struct' : kdesc.SHARED_IFACE.CALL_OPTIONS_NAME if shared_iface else 'void',
            'kernel_family_name'  : kdesc.FAMILY,
            'shim_kernel_name'    : kdesc.NAME,
            'flyc_gpu_symbol'     : kdesc.gpu_symbol_name,
            'param_class_name'    : kdesc.param_class_name,
            'context_class_name'  : kdesc.context_class_name,
            'godel_number_body'   : self.codegen_godel_number_body(),
            'pp_func_num'         : pp_func_num,
            'list_of_pp_args_function_defs'  : list_of_pp_args_function_defs,
            'list_of_pp_args_function_decls' : list_of_pp_args_function_decls,
            'get_archmod_number_body' : self.codegen_archmod_number_body(),
            'number_of_functionals'  : kdesc.godel_number,
            'define_compiled_in_features' : self.codegen_define_compiled_in_features(),
            'per_kernel_packed_string'  : self.codegen_per_kernel_packed_string(),
            'kernel_table_entries' : self.codegen_tune_table_entries(functionals),
            'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_per_kernel_packed_string(self):
        return self._this_repo.get_data('per_kernel_packed_string')

    def codegen_declare_compiled_in_features(self):
        kdesc = self._iface
        decl_list = []
        for tp in kdesc.list_functional_params():  # tp: TemplateParam
            if not tp.emit_feature_table:
                continue
            infotype = tp.repr_typed_choice.infotype
            decl_code = f'static const std::vector<{infotype}>& get_{tp.repr_name}_choices();'
            decl_list.append(decl_code)
        return '\n    '.join(decl_list)

    def codegen_define_compiled_in_features(self):
        def_list = []
        kdesc = self._iface
        meta_class = kdesc.metadata_class_name
        for tp in kdesc.list_functional_params():  # tp: TemplateParam
            if not tp.emit_feature_table:
                continue
            infotype = tp.repr_typed_choice.infotype
            choices = ', '.join([tc.infotext for tc in tp.choices])
            def_code = f'''
const std::vector<{infotype}>& {meta_class}::get_{tp.repr_name}_choices()
{{
    static const std::vector<{infotype}> choices = {{ {choices} }};
    return choices;
}}'''
            def_list.append(def_code)
        return '\n'.join(def_list)

    def codegen_context_helper_declares(self):
        kdesc = self._iface
        lines = [f'{ctype} {name}() const;' for name, ctype in kdesc.iter_context_helpers()]
        return '\n    '.join(lines)

    def codegen_context_helper_scratch_members(self):
        """Storage for the ati.context_helper() results.

        Grouped into one struct rather than loose members so the kernarg
        vector's `CAST(&context.scratch_params.<name>)` reads as one namespace,
        and so adding a helper does not add another top-level context field.
        `mutable` because pp_args fills them from a const context, and the
        kernarg vector holds their addresses -- a local would dangle.
        """
        kdesc = self._iface
        helpers = list(kdesc.iter_context_helpers())
        if not helpers:
            return '// no context helpers'
        body = '\n'.join(f'        {ctype} {name};' for name, ctype in helpers)
        return 'mutable struct {\n' + body + '\n    } scratch_params;'
        return '\n    '.join(lines)

    def codegen_kernel_arguments(self):
        context_class_name = self._iface.context_class_name
        pp_registry = self._this_repo.get_data('pp_function')
        stmt = []
        array = []
        for assign_skips, (findex, src) in pp_registry.items():
            pp_function_name = f'{self._iface.NAME}_pp_args_{findex}'
            stmt.append(f'static std::vector<void*>')
            stmt.append(f'{pp_function_name}(const {context_class_name}& context) {{')
            stmt.append(f'  const auto& params = *context.params;')
            stmt.append(src)
            stmt.append(f'}}')
            array.append(pp_function_name)
        pp_func_num = len(pp_registry.keys())
        return '\n'.join(stmt), ',\n  '.join(array), pp_func_num
