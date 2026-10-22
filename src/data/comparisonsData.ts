export interface CodeComparison {
  id: string;
  name: string;
  description: string;
  juliaCode: string;
  cppCode: string;
  pythonCode: string;
  advantages: string[];
}

export const CODE_COMPARISONS: CodeComparison[] = [
  {
    id: "vector-add",
    name: "Element-wise Vector Addition Kernel",
    description: "Standard 1D element-wise kernel across C++ CUDA, Python Numba CUDA, and Julia CUDA.jl.",
    juliaCode: `using CUDA

function vadd!(c, a, b)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

# Launch
N = 1_000_000
a = CUDA.ones(Float32, N)
b = CUDA.ones(Float32, N)
c = CUDA.zeros(Float32, N)

threads = 256
blocks = cld(N, threads)
@cuda threads=threads blocks=blocks vadd!(c, a, b)
CUDA.synchronize()`,
    cppCode: `// C++ CUDA (requires nvcc compilation)
#include <cuda_runtime.h>

__global__ void vadd(float* c, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host launch
int main() {
    int n = 1000000;
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vadd<<<blocks, threads>>>(d_c, d_a, d_b, n);
    cudaDeviceSynchronize();
    return 0;
}`,
    pythonCode: `# Python Numba CUDA
from numba import cuda
import numpy as np

@cuda.jit
def vadd(c, a, b):
    i = cuda.grid(1)
    if i < c.size:
        c[i] = a[i] + b[i]

# Launch
n = 1_000_000
a_d = cuda.to_device(np.ones(n, dtype=np.float32))
b_d = cuda.to_device(np.ones(n, dtype=np.float32))
c_d = cuda.device_array(n, dtype=np.float32)

threads = 256
blocks = (n + threads - 1) // threads
vadd[blocks, threads](c_d, a_d, b_d)
cuda.synchronize()`,
    advantages: [
      "No separate nvcc build step: Julia JIT-compiles Julia AST straight to PTX via LLVM.",
      "Generic types: Julia kernels work on Float32, Float64, Complex, Dual numbers, or custom user structs with 0 changes.",
      "Integrated REPL: Test, inspect, and benchmark GPU kernels interactively in seconds."
    ]
  },
  {
    id: "fused-math",
    name: "Multi-Operation Kernel Fusion",
    description: "Evaluating complex formulas y = 2*x^2 + 3*sin(x) without temporary array allocations.",
    juliaCode: `using CUDA

# In Julia, this produces a SINGLE optimized GPU kernel
# with 0 temporary array allocations!
x = CUDA.rand(Float32, 10_000_000)
y = CUDA.zeros(Float32, 10_000_000)

CUDA.@sync begin
    y .= 2.0f0 .* (x .^ 2) .+ 3.0f0 .* sin.(x)
end`,
    cppCode: `// In C++ CUDA, you must manually write a custom __global__
// kernel with thread indexing or link Thrust / custom templates.
__global__ void fused_op(float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = 2.0f * (x[i] * x[i]) + 3.0f * sinf(x[i]);
    }
}
// Plus boilerplate host allocations and launches...`,
    pythonCode: `# In PyTorch / NumPy / CuPy without torch.compile:
import cupy as cp

x = cp.random.rand(10_000_000, dtype=cp.float32)
# ⚠️ This allocates 4 temporary memory buffers on GPU:
# 1 for x^2, 1 for 2*..., 1 for sin(x), 1 for 3*...
y = 2.0 * (x ** 2) + 3.0 * cp.sin(x)
cp.cuda.Stream.null.synchronize()`,
    advantages: [
      "Julia's dot operator '.' is first-class syntactic loop fusion, not a library macro.",
      "Zero intermediate memory allocations saves gigabytes of VRAM.",
      "Achieves hand-written C++ CUDA kernel performance with one readable line of high-level math."
    ]
  },
  {
    id: "shared-memory-tiling",
    name: "Shared Memory Stencil / Tiling",
    description: "Caching neighborhood memory in high-speed on-chip shared memory.",
    juliaCode: `using CUDA

function stencil_1d!(out, in, radius)
    # L1 on-chip cache per block
    s_data = @cuStaticSharedMem(Float32, 256 + 2 * 2) # Block + Halos

    t_id = threadIdx().x
    g_id = (blockIdx().x - Int32(1)) * blockDim().x + t_id
    # Collaboratively load into shared memory...
    sync_threads()
    # Compute stencil...
    return nothing
end`,
    cppCode: `// C++ CUDA
__global__ void stencil_1d(float* out, const float* in, int radius) {
    __shared__ float s_data[256 + 2 * 2];

    int t_id = threadIdx.x;
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Collaboratively load...
    __syncthreads();
    // Compute stencil...
}`,
    pythonCode: `# Python Numba CUDA
from numba import cuda

@cuda.jit
def stencil_1d(out, in_arr, radius):
    s_data = cuda.shared.array(shape=(260,), dtype=numba.float32)
    t_id = cuda.threadIdx.x
    g_id = cuda.grid(1)
    # Load...
    cuda.syncthreads()
    # Compute...`,
    advantages: [
      "Explicit @cuStaticSharedMem and @cuDynamicSharedMem mirror CUDA C++ semantics.",
      "Compile-time bounds checking can be selectively disabled via @inbounds.",
      "Clean syntax with full Julia introspection (@device_code_ptx)."
    ]
  }
];
