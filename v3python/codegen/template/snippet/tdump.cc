    if (AOTRITON_NS::debug::is_tensordump_enabled()) {
        using AOTRITON_NS::libtensordump::tensordump;
        using AOTRITON_NS::libtensordump::scalardump;
        auto [gpu_index, call_index] = AOTRITON_NS::debug::tdump_trackcall(stream, triton_kernel_name);
        std::string kernel_name_string(triton_kernel_name);
        [[tdump_statements]];
    }
