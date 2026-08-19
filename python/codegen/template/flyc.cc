// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "flyc.[[shim_kernel_name]].h"
#include <aotriton/util.h>
#include <aotriton/_internal/log.h>
#include <tuple>
#if AOTRITON_BUILD_FOR_TUNING
#include <aotriton/cpp_tune.h>
#endif
[[includes]]

namespace AOTRITON_NS::v3::[[kernel_family_name]] {

#if [[shared_iface]]
using AOTRITON_NS::v3::[[shared_iface_family]]::[[param_class_name]];
#define KERNEL_SLOT_INDEX ([[call_options_struct]]::KernelSlot::[[shim_kernel_name]])
#endif

#define CAST(x) const_cast<void*>(static_cast<const void*>(x))
// No TritonAuxiliaryArguments here: the flyc kernel takes exactly 44 real
// parameters and none of them is the two trailing scratch pointers Triton's
// own kernels take (PLAN-PHASE2.md Task 4, delta 1).
typedef std::vector<void*>(*PP_FUNC)(const [[context_class_name]]& context);

namespace {
extern PP_FUNC prepare_arguments[ [[pp_func_num]] ];
}

int64_t [[context_class_name]]::godel_number() const
{
    int64_t sum = 0;
    const auto& args = *params;
[[godel_number_body]]
    return sum;
}

hipError_t
[[context_class_name]]::lookup_optimal(Gpu gpu) {
    // Note:
    // context object must be called in this order
    //  ctor -> lookup_optimal -> launch
    // Here launch_condition is re-used as initial value from ctor, which is
    // the condition set by metro kernel to completely disable a specific
    // kernel (e.g. debug_simulate_encoded_softmax).
    // Hence, launch_condition must be checked at the beginning of
    // lookup_optimal() as well.
    if (!launch_condition)
      return hipSuccess;
#if AOTRITON_BUILD_FOR_TUNING && [[shared_iface]]
    if (call_options) {
        auto& kctl = *call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;

        // Check Ignore flag - skip lookup/execution if set
        if (ctrl & KernelControl::Ignore) {
            launch_condition = false;
            return hipSuccess;
        }

        // Check Skip flag - lookup but skip execution
        if (ctrl & KernelControl::Skip) {
            launch_condition = false;
        }

        // Check Manual flag - use hsaco_index if set
        if (ctrl & KernelControl::Manual) {
            _has_preferred_kernel = kctl.hsaco_index;
            peek_kernel_image = (ctrl & KernelControl::ExtractImage);
        }
    }
#endif

    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    kernel_on_device = nullptr;
    auto number = godel_number();
    if (number < 0)
        return hipErrorNotSupported;
    auto tune_func = autotune_table[arch_number][number];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    auto kernel_index = tune_func(*this, mod_number);
    // aks2_entry/func_name/arch_name are only set when a valid kernel is selected.
    if (kernel_index < 0) {
        AOTRITON_LOG(LOG_DEBUG,
                     "[[shim_kernel_name]] lookup_optimal: no valid kernel selected "
                     "(kernel_index = %d)",
                     kernel_index);
    } else {
        AOTRITON_LOG(LOG_DEBUG,
                     "[[shim_kernel_name]] lookup_optimal: kernel_index = %d "
                     "aks2_entry = %.*s func_name = %.*s arch_name = %.*s",
                     kernel_index,
                     int(aks2_entry.size()), aks2_entry.data(),
                     int(func_name.size()), func_name.data(),
                     int(arch_name.size()), arch_name.data());
    }
    if (!kernel_on_device)
        return hipErrorSharedObjectSymbolNotFound;
    // Design B (PLAN-PHASE2.md Task 2): the perf/copt knob set is a schemaless
    // string, not a struct. Parse once here -- lookup_optimal already runs
    // exactly once before launch and already touches the selected kernel --
    // so grid_calculator() never parses on the launch path.
    perf_ = Schemaless(kernel_on_device->psel());

#if AOTRITON_BUILD_FOR_TUNING && [[shared_iface]]
    if (call_options) {
        auto& kctl = *call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;

        // Write total_hsacos if Query is set
        if (ctrl & KernelControl::Query) {
            kctl.total_hsacos = _total_number_of_kernels;
            kctl.kernel_psels = _preferred_kernel_psels;
            kctl.kernel_copts = _preferred_kernel_copts;
        }
    }
#endif
    return hipSuccess;
}

hipError_t
[[context_class_name]]::launch(hipStream_t stream) const {
    if (!launch_condition)
      return hipSuccess;
    // Renamed from Triton's `triton_kernel_name` (cosmetic, PLAN-PHASE2.md
    // Task 4, delta 3): passed to invoke() purely for logging.
    constexpr std::string_view kFlycKernelName { "[[shim_kernel_name]]" };
    auto args = prepare_arguments[pp_args_index](*this);
    dim3 grid;
    if (custom_grid_calculator) {
        grid = custom_grid_calculator(*this);
    } else {
        grid = grid_calculator();
    }
#if AOTRITON_BUILD_FOR_TUNING && [[shared_iface]]
    // invoke() is reused AS-IS (PLAN-PHASE2.md Task 4, delta 2): nothing in its
    // signature is Triton-specific, and the block size comes from the aks2
    // directory entry (num_warps * warp_size), not from the caller.
    auto ret = kernel_on_device->invoke(kFlycKernelName,
                                        flatzip_path,
                                        aks2_entry,
                                        func_name,
                                        arch_name,
                                        grid,
                                        args,
                                        peek_kernel_image,
                                        stream);
    if (ret != hipSuccess)
         return ret;
    if (call_options) {
        auto& kctl = *call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;
        if (ctrl & KernelControl::Manual && ctrl & KernelControl::ExtractImage) {
            auto essentials = kernel_on_device->get_image_info_iff_decompressed();
            kctl.kernel_image = essentials.image;
            kctl.image_size = essentials.size;
        }
    }
    return ret;
#else
    return kernel_on_device->invoke(kFlycKernelName,
                                    flatzip_path,
                                    aks2_entry,
                                    func_name,
                                    arch_name,
                                    grid,
                                    args,
                                    stream);
#endif
}

std::tuple<int, int>
[[context_class_name]]::get_archmod_number(Gpu gpu) {
    [[get_archmod_number_body]];
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}


[[list_of_pp_args_function_defs]]

namespace {
PP_FUNC prepare_arguments[ [[pp_func_num]] ] = {
  [[list_of_pp_args_function_decls]]
};
}

[[define_compiled_in_features]]

namespace flytune {

const char [[shim_kernel_name]]_packed_string[] =
[[per_kernel_packed_string]];

[[list_of_deduplicated_lut_functions]]

} // namespace flytune

[[context_class_name]]::AutoTuneTableEntry
[[context_class_name]]::autotune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

}

// vim: set fileencoding=utf-8
