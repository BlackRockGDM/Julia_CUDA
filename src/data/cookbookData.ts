import { CodeRecipe } from "../types";

export const COOKBOOK_RECIPES: CodeRecipe[] = [
  {
    id: "device-query",
    title: "GPU Device Verification & Introspection",
    category: "Quickstart",
    difficulty: "Beginner",
    summary: "Check CUDA availability, query GPU device properties, driver version, and memory status in Julia.",
    code: `using CUDA

# 1. Verify functional GPU driver and toolkit
if !CUDA.functional()
    error("CUDA is not functional. Ensure NVIDIA drivers are installed.")
end

println("CUDA is functional! Version: ", CUDA.runtime_version())

# 2. Inspect active device
dev = CUDA.device()
println("Device Name:       ", CUDA.name(dev))
println("Compute Capability: ", CUDA.capability(dev))
println("Total Device Memory:", Base.format_bytes(CUDA.totalmem(dev)))

# 3. Print real-time memory usage (Free vs Used)
println("\nCurrent Memory Status:")
CUDA.memory_status()

# 4. Check device limits (e.g. max threads per block)
props = CUDA.properties(dev)
println("\nMax Threads per Block: ", props.maxThreadsPerBlock)
println("Total SM Count:        ", props.multiProcessorCount)`,
    explanation: [
      "CUDA.jl automatically downloads and configures pre-built CUDA artifacts via Julia's artifact system without needing manual CUDA toolkit installation.",
      "CUDA.functional() returns a boolean indicating if both hardware and driver are ready.",
      "CUDA.memory_status() is invaluable during development to detect GPU memory leaks or OOM issues."
    ],
    tags: ["init", "setup", "memory", "device", "properties"]
  },
  {
    id: "fused-broadcasting",
    title: "High-Level Array Vectorization & Kernel Fusion",
    category: "CuArray Basics",
    difficulty: "Beginner",
    summary: "Leverage Julia's dot syntax (.) to fuse multiple operations into a single optimized GPU kernel automatically.",
    code: `using CUDA

# Generate random CuArrays directly on GPU (no host-to-device copy needed)
N = 10_000_000
x_d = CUDA.rand(Float32, N)
y_d = CUDA.rand(Float32, N)

# Fused mathematical expression:
# In C++ or Python, evaluating (3.0 * x^2 + 2.0 * y - 1.0) would allocate
# multiple intermediate temporary GPU buffers and launch 3 separate kernels.
# In Julia, the '.' syntax fuses this entire pipeline into a SINGLE PTX kernel!
CUDA.@sync begin
    y_d .= 3.0f0 .* (x_d .^ 2) .+ 2.0f0 .* y_d .- 1.0f0
end

# Verification: mapreduce runs on the GPU with logarithmic tree reduction
total = sum(y_d)
max_val = maximum(y_d)

println("Evaluation complete. Sum: ", total, " | Max: ", max_val)`,
    explanation: [
      "Julia's broadcast fusion (the dot operator '.') compiles arbitrary mathematical expressions into a single GPU kernel without allocating intermediate arrays.",
      "Notice the '3.0f0' literal: always use single-precision Float32 literals (e.g. 1.0f0 instead of 1.0) on consumer/GeForce GPUs for max speed.",
      "CUDA.@sync forces the host CPU to wait for GPU asynchronous operations to finish before moving to the next line."
    ],
    tags: ["broadcasting", "cuarray", "fusion", "vectorization", "performance"]
  },
  {
    id: "avoid-scalar-indexing",
    title: "The #1 Trap: Avoiding Scalar Indexing",
    category: "Profiling & Traps",
    difficulty: "Intermediate",
    summary: "Why accessing individual GPU elements on the CPU will kill performance, and how allowscalar(false) saves your code.",
    code: `using CUDA

# ALWAYS enforce this at the start of production Julia scripts!
CUDA.allowscalar(false)

# Create GPU array
A_d = CUDA.zeros(Float32, 1000)

# ============================================================
# ⚠️ FATAL ANTI-PATTERN (Scalar Indexing):
# If allowscalar were true, the following loop would make 1,000
# round-trip PCIe transfers between CPU and GPU:
#
# for i in 1:1000
#     A_d[i] = Float32(i) # EXTREMELY SLOW: 1000x PCIe overhead!
# end
# ============================================================

# ✅ SOLUTION 1: Vectorized broadcast initialization
# Let the GPU initialize itself in parallel:
indices_d = CuArray(1:1000)
A_d .= Float32.(indices_d)

# ✅ SOLUTION 2: Custom GPU Kernel
function set_elements_kernel!(A)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(A)
        @inbounds A[i] = Float32(i)
    end
    return nothing
end

threads = 256
blocks = cld(length(A_d), threads)
@cuda threads=threads blocks=blocks set_elements_kernel!(A_d)
CUDA.synchronize()

println("Successfully updated GPU array without scalar transfers!")`,
    explanation: [
      "Scalar indexing occurs whenever CPU code accesses A[i] where A is a CuArray.",
      "Each scalar read or write requires a PCIe memory round-trip (~microseconds), turning a milliseconds task into minutes.",
      "Setting CUDA.allowscalar(false) turns silent performance degradation into an immediate error with stack trace so you can pinpoint the offending line."
    ],
    tags: ["allowscalar", "anti-pattern", "pcie", "performance", "pitfall"]
  },
  {
    id: "custom-1d-kernel",
    title: "Writing Custom 1D Kernels with @cuda",
    category: "Custom Kernels",
    difficulty: "Intermediate",
    summary: "Write and launch low-level GPU kernels with 1-based indexing, boundary checks, and dynamic grid sizing.",
    code: `using CUDA

# Custom element-wise kernel
# CRITICAL: GPU kernels must return \`nothing\`
function saxpy_kernel!(y, a, x)
    # 1-based indexing idiom in Julia CUDA:
    # Notice the 'Int32(1)' to prevent promotion to 64-bit integer registers
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Boundary guard: threads * blocks is typically larger than array length
    if i <= length(y)
        @inbounds y[i] = a * x[i] + y[i]
    end

    return nothing
end

function main()
    N = 2^22 # ~4.19 million elements
    a = 2.5f0
    x_d = CUDA.fill(1.0f0, N)
    y_d = CUDA.fill(2.0f0, N)

    # Grid & Block configuration
    threads = 256
    blocks = cld(N, threads) # Julia's ceil-div function: ceil(Int, N / threads)

    println("Launching: \$blocks blocks with \$threads threads per block...")
    @cuda threads=threads blocks=blocks saxpy_kernel!(y_d, a, x_d)
    CUDA.synchronize()

    # Verify result: 2.5 * 1.0 + 2.0 = 4.5
    sample = Array(y_d[1:5]) # Copy only 5 elements back to CPU
    println("First 5 elements: ", sample)
end

main()`,
    explanation: [
      "Julia uses 1-based indexing! The formula is `i = (blockIdx().x - 1) * blockDim().x + threadIdx().x`.",
      "Julia's cld(N, threads) computes the ceiling of N / threads, guaranteeing enough blocks to cover the array.",
      "Inside the boundary check, use `@inbounds` to disable bounds checking and save precious cycles.",
      "CUDA kernels in Julia must always terminate with `return nothing`."
    ],
    tags: ["@cuda", "saxpy", "indexing", "launch", "threads", "blocks"]
  },
  {
    id: "launch-configuration",
    title: "Automatic Optimal Launch Configuration",
    category: "Custom Kernels",
    difficulty: "Intermediate",
    summary: "Query the CUDA driver for the mathematically optimal block size and grid size for your specific kernel.",
    code: `using CUDA

function vector_calc_kernel!(out, in_a, in_b)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(out)
        @inbounds out[i] = sqrt(in_a[i]^2 + in_b[i]^2)
    end
    return nothing
end

function launch_optimal()
    N = 5_000_000
    a_d = CUDA.rand(Float32, N)
    b_d = CUDA.rand(Float32, N)
    out_d = CUDA.zeros(Float32, N)

    # Compile the kernel without launching it:
    kernel = @cuda launch=false vector_calc_kernel!(out_d, a_d, b_d)

    # Let CUDA.jl calculate the optimal occupancy configuration:
    config = launch_configuration(kernel.fun)
    threads = config.threads
    blocks = min(config.blocks, cld(N, threads))

    println("Hardware-Optimized Config:")
    println("  Optimal Threads per Block: ", threads)
    println("  Calculated Blocks:          ", blocks)

    # Launch kernel with the driver-calculated configuration:
    kernel(out_d, a_d, b_d; threads=threads, blocks=blocks)
    CUDA.synchronize()

    println("Optimal kernel executed successfully!")
end

launch_optimal()`,
    explanation: [
      "@cuda launch=false compiles the kernel to PTX/cubin and returns an executable HostKernel object.",
      "launch_configuration(kernel.fun) inspects the register pressure and shared memory of the compiled kernel to maximize active warps.",
      "This eliminates manual guessing of thread block dimensions."
    ],
    tags: ["launch_configuration", "occupancy", "optimization", "driver"]
  },
  {
    id: "static-shared-memory",
    title: "Shared Memory Tiling (@cuStaticSharedMem)",
    category: "Shared Memory",
    difficulty: "Advanced",
    summary: "Coalesce global memory loads into ultra-fast on-chip Shared Memory with thread synchronization barriers.",
    code: `using CUDA

# 1D Reverse kernel using Shared Memory to demo @cuStaticSharedMem and sync_threads()
function reverse_block_kernel!(output, input)
    # Allocate 256 Float32 slots in L1 on-chip Shared Memory per block
    s_data = @cuStaticSharedMem(Float32, 256)

    t_id = threadIdx().x
    b_id = blockIdx().x
    b_dim = blockDim().x

    global_id = (b_id - Int32(1)) * b_dim + t_id

    # 1. Collaborative load from global memory into on-chip shared memory
    if global_id <= length(input)
        @inbounds s_data[t_id] = input[global_id]
    end

    # 2. Block-wide barrier: wait until all 256 threads finished loading
    sync_threads()

    # 3. Read reversed element from shared memory and write back to global memory
    reversed_tid = b_dim - t_id + Int32(1)
    if global_id <= length(output)
        @inbounds output[global_id] = s_data[reversed_tid]
    end

    return nothing
end

function run_reverse()
    N = 256 * 10
    inp_d = CuArray(Float32.(1:N))
    out_d = CUDA.zeros(Float32, N)

    @cuda threads=256 blocks=10 reverse_block_kernel!(out_d, inp_d)
    CUDA.synchronize()

    # Verify first block is reversed: [256, 255, ..., 1]
    println("First block result: ", Array(out_d[1:6]), " ... ", Array(out_d[256]))
end

run_reverse()`,
    explanation: [
      "@cuStaticSharedMem allocates memory inside the SM's L1 cache, with latency < 5ns (compared to > 200ns for global memory).",
      "sync_threads() is essential: without it, threads will read from shared memory before neighboring threads have written their values (data race).",
      "Shared memory is private to each thread block and discarded when the block completes."
    ],
    tags: ["shared_memory", "sync_threads", "caching", "tiling", "barrier"]
  },
  {
    id: "dynamic-shared-memory",
    title: "Dynamic Shared Memory (@cuDynamicSharedMem)",
    category: "Shared Memory",
    difficulty: "Advanced",
    summary: "Allocate dynamic shared memory sizes at kernel launch time using the shmem argument in @cuda.",
    code: `using CUDA

function dynamic_shmem_kernel!(dest, src, block_len)
    # Dynamically sized shared memory buffer
    shmem = @cuDynamicSharedMem(Float32, block_len)

    tid = threadIdx().x
    gid = (blockIdx().x - Int32(1)) * blockDim().x + tid

    if gid <= length(src)
        @inbounds shmem[tid] = src[gid] * 2.0f0
    end

    sync_threads()

    if gid <= length(dest)
        @inbounds dest[gid] = shmem[tid]
    end

    return nothing
end

function run_dynamic()
    N = 1024
    threads = 512
    blocks = cld(N, threads)

    src_d = CUDA.rand(Float32, N)
    dest_d = CUDA.zeros(Float32, N)

    # Calculate shared memory required in bytes
    shmem_bytes = threads * sizeof(Float32)

    # Pass shmem bytes in @cuda
    @cuda threads=threads blocks=blocks shmem=shmem_bytes dynamic_shmem_kernel!(dest_d, src_d, threads)
    CUDA.synchronize()

    println("Dynamic shared memory kernel completed with ", shmem_bytes, " bytes/block.")
end

run_dynamic()`,
    explanation: [
      "@cuDynamicSharedMem allows the shared memory array size to be configured at runtime rather than hardcoded at compile time.",
      "The `shmem` keyword argument in `@cuda` specifies the total bytes to allocate per block.",
      "Check your GPU's maximum shared memory per block (typically 48 KB up to 99 KB or 227 KB on Hopper)."
    ],
    tags: ["dynamic_shmem", "shmem", "runtime", "memory"]
  },
  {
    id: "warp-shuffles",
    title: "Warp Shuffles & Atomics for Fast Reductions",
    category: "Atomics & Warps",
    difficulty: "Advanced",
    summary: "Exchange registers directly between threads in a warp using shfl_down_sync without touching shared memory.",
    code: `using CUDA

# Warp-level parallel reduction: 32 threads compute sum in 5 clock cycles
@inline function warp_reduce_sum(val::Float32)
    offset = Int32(16)
    while offset > Int32(0)
        # 0xffffffff means all 32 threads in the warp participate
        val += shfl_down_sync(0xffffffff, val, offset)
        offset >>= 1
    end
    return val
end

function block_reduce_kernel!(out_sum, input)
    tid = threadIdx().x
    gid = (blockIdx().x - Int32(1)) * blockDim().x + tid

    # Load from global memory
    val = (gid <= length(input)) ? input[gid] : 0.0f0

    # Reduce across warp
    val = warp_reduce_sum(val)

    # Thread 1 of the warp has the partial sum
    # Atomically add partial sum to global accumulator
    if (tid - Int32(1)) % 32 == 0
        CUDA.@atomic out_sum[1] += val
    end

    return nothing
end

function run_warp_sum()
    N = 1_000_000
    x_d = CUDA.fill(1.0f0, N)
    sum_d = CUDA.zeros(Float32, 1)

    threads = 256
    blocks = cld(N, threads)

    @cuda threads=threads blocks=blocks block_reduce_kernel!(sum_d, x_d)
    CUDA.synchronize()

    gpu_sum = Array(sum_d)[1]
    println("Computed GPU sum: ", gpu_sum, " (Expected: ", Float32(N), ")")
end

run_warp_sum()`,
    explanation: [
      "Warp shuffles (shfl_down_sync) allow threads within the same 32-thread warp to read each other's registers directly with 0 memory latency.",
      "CUDA.@atomic prevents race conditions when multiple warps update the same memory location simultaneously.",
      "Combining warp shuffles with atomic operations reduces atomic contention by 32x."
    ],
    tags: ["warp_shuffle", "atomics", "reduction", "shfl_down_sync", "warps"]
  },
  {
    id: "cublas-gemm",
    title: "NVIDIA CUBLAS & CUSOLVER Integration",
    category: "Libraries",
    difficulty: "Intermediate",
    summary: "Call hardware-accelerated Tensor Core BLAS routines directly in Julia for maximum matrix multiplication performance.",
    code: `using CUDA
using LinearAlgebra

# Dimensions
M, K, N = 2048, 2048, 2048

# Allocate on GPU
A_d = CUDA.rand(Float32, M, K)
B_d = CUDA.rand(Float32, K, N)
C_d = CUDA.zeros(Float32, M, N)

# Standard high-level matrix multiplication automatically routes to CUBLAS:
# C = A * B
CUDA.@sync begin
    mul!(C_d, A_d, B_d)
end

println("CUBLAS gemm executed successfully!")

# Direct low-level CUBLAS call with alpha and beta scaling:
# C = 1.0f0 * A * B + 0.0f0 * C
CUDA.CUBLAS.gemm!('N', 'N', 1.0f0, A_d, B_d, 0.0f0, C_d)

# CUSOLVER for linear system solving (A \ b) on GPU:
A_sq = CUDA.rand(Float32, 1000, 1000)
A_sym = A_sq * A_sq' # make symmetric positive-definite
b = CUDA.rand(Float32, 1000)

x = A_sym \\ b # Uses CUSOLVER Cholesky / LU decomposition
println("Solved 1000x1000 linear system on GPU. Norm: ", norm(x))`,
    explanation: [
      "CUDA.jl includes comprehensive wrappers around NVIDIA's vendor libraries: CUBLAS, CUSOLVER, CURAND, CUFFT, and CUTENSOR.",
      "Julia's standard LinearAlgebra functions (like mul!, \\, qr, svd) automatically dispatch to CUBLAS / CUSOLVER when given CuArrays.",
      "Vendor libraries take full advantage of specialized Tensor Cores for peak TFLOPs."
    ],
    tags: ["cublas", "cusolver", "linear_algebra", "tensor_cores", "gemm"]
  },
  {
    id: "profiling-ptx",
    title: "Kernel Inspection: PTX, LLVM & NVTX Profiling",
    category: "Profiling & Traps",
    difficulty: "Advanced",
    summary: "Inspect compiled PTX/LLVM assembly code and insert NVTX ranges for NVIDIA Nsight Systems timeline profiling.",
    code: `using CUDA

function my_calc!(y, x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] = sin(x[i]) * 2.0f0
    end
    return nothing
end

# 1. View compiled PTX assembly (NVIDIA Virtual Machine code)
println("=== Generated PTX Assembly ===")
@device_code_ptx @cuda threads=256 blocks=1 my_calc!(CUDA.zeros(Float32, 256), CUDA.rand(Float32, 256))

# 2. View generated LLVM IR to ensure type stability
# (look for dynamic dispatch or boxing, which should be absent)
# @device_code_llvm @cuda threads=256 blocks=1 my_calc!(...)

# 3. NVTX Timeline markers for NVIDIA Nsight Systems:
# using NVTX
# NVTX.range("Matrix Operations") do
#     CUDA.@sync mul!(C_d, A_d, B_d)
# end

# 4. Measure execution time with CUDA.@time (includes GPU sync)
x_d = CUDA.rand(Float32, 1_000_000)
y_d = CUDA.zeros(Float32, 1_000_000)

println("\n=== Accurate GPU Timing ===")
CUDA.@time begin
    @cuda threads=256 blocks=cld(1_000_000, 256) my_calc!(y_d, x_d)
end`,
    explanation: [
      "@device_code_ptx lets you verify register usage and instructions generated by LLVM.",
      "@device_code_warntype / @device_code_llvm helps ensure there are no type instabilities or garbage collection allocations inside the GPU kernel.",
      "Always use CUDA.@time or BenchmarkTools with CUDA.@sync, because ordinary @time only measures asynchronous kernel launch time, not execution time."
    ],
    tags: ["ptx", "llvm", "nsight", "nvtx", "profiling", "timing"]
  }
];
