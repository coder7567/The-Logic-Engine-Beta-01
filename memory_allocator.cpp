#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>

// Engineering Blueprint: Memory Allocation Strategy
// The Logic Engine enforces strict tensor constraints to fit the 8B model LoRA tuning
// into the 16GB VRAM of the RTX 5070.
// We utilize PyTorch's CUDACachingAllocator for fast block reuse but inject a 
// ceiling check.

size_t check_vram_limit(size_t requested_bytes) {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    
    // Hard threshold: 14.5 GB (leaves room for OS display server and basic OS processes)
    const size_t HARD_LIMIT = 14.5 * 1024 * 1024 * 1024ULL;
    
    // We query PyTorch's current allocator statistics
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    size_t currently_allocated = stats.allocated_bytes[0].current;
    
    if (currently_allocated + requested_bytes > HARD_LIMIT) {
        throw std::runtime_error("Logic Engine OOM: VRAM Hard Limit (14.5GB) Exceeded. Aborting to prevent OS freeze.");
    }
    return currently_allocated;
}

PYBIND11_MODULE(logic_engine_memory, m) {
    m.def("check_vram_limit", &check_vram_limit, "Check strict RTX 5070 VRAM bounds");
}
