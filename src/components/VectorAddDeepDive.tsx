import React, { useState } from "react";
import { Copy, Check, Terminal, Play, ArrowDown, Cpu, HardDrive, Info, Layers, CheckCircle2, AlertTriangle, Sliders } from "lucide-react";
import { GpuSpec } from "../types";

interface VectorAddDeepDiveProps {
  selectedGpu: GpuSpec;
}

export const VectorAddDeepDive: React.FC<VectorAddDeepDiveProps> = ({ selectedGpu }) => {
  const [activeStep, setActiveStep] = useState<number>(1);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);
  const [blockSize, setBlockSize] = useState<number>(256);
  const [elementsN, setElementsN] = useState<number>(1_000_000);

  // Computed block counts and warps
  const totalBlocks = Math.ceil(elementsN / blockSize);
  const warpsPerBlock = blockSize / 32;
  const isMultipleOf32 = blockSize % 32 === 0;

  // Occupancy estimation for vector add (simple kernel with ~16 registers per thread and 0 shared memory)
  const maxWarpsSm = selectedGpu.maxWarpsPerSm;
  const activeBlocksByWarps = Math.floor(maxWarpsSm / Math.max(1, warpsPerBlock));
  const activeBlocksSm = Math.min(selectedGpu.maxBlocksPerSm, activeBlocksByWarps);
  const activeWarpsSm = activeBlocksSm * warpsPerBlock;
  const occupancyPercent = Math.min(100, Math.round((activeWarpsSm / maxWarpsSm) * 100));

  const fullJuliaScript = `# ==============================================================================
# Complete Julia CUDA.jl Element-wise Vector Addition & Memory Transfer Guide
# ==============================================================================
using CUDA
using Test

# 1. Best practice: Prevent accidental, slow PCIe scalar transfers
CUDA.allowscalar(false)

# 2. Kernel Definition
# Custom GPU kernel that executes in parallel across thousands of threads
function vadd_kernel!(c, a, b)
    # 1-Based Global Thread Indexing:
    # blockIdx().x is 1-based (1, 2, 3...)
    # threadIdx().x is 1-based (1, 2, ..., blockDim().x)
    # Subtract 1 from blockIdx().x before multiplying by blockDim().x:
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    # Boundary Guard:
    # The grid size (blocks * threads) might be larger than array length N.
    # Threads where i > length(c) must do nothing.
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end

    return nothing  # Critical: CUDA.jl kernels must return nothing!
end

function main()
    println("CUDA functional: ", CUDA.functional())
    println("Running on:      ", CUDA.name(CUDA.device()))

    # Problem size: 1,000,000 single-precision floats (4 MB each)
    N = 1_000_000

    # --------------------------------------------------------------------------
    # Step 1: Allocate CPU Arrays (Host Memory)
    # --------------------------------------------------------------------------
    println("\\n[1] Allocating host CPU arrays...")
    a_cpu = fill(1.5f0, N)   # Float32 array on CPU
    b_cpu = fill(2.5f0, N)   # Float32 array on CPU

    # --------------------------------------------------------------------------
    # Step 2: Allocate & Copy to GPU (Host-to-Device Memory Transfer)
    # --------------------------------------------------------------------------
    println("[2] Transferring arrays from CPU to GPU (H2D)...")
    # CuArray(cpu_array) allocates device VRAM and copies the bytes via PCIe
    a_gpu = CuArray(a_cpu)
    b_gpu = CuArray(b_cpu)

    # Allocate uninitialized or zero-filled output array directly in GPU VRAM:
    c_gpu = CUDA.zeros(Float32, N)

    # --------------------------------------------------------------------------
    # Step 3: Configure Thread Block Size & Launch Kernel
    # --------------------------------------------------------------------------
    # Thread Block Size (must be multiple of 32 for warp scheduling; 256 is standard)
    threads = ${blockSize}
    # Grid Size: ceil-division cld(N, threads) ensures all N elements are covered
    blocks = cld(N, threads)
    println("[3] Launching kernel with \$blocks blocks and \$threads threads/block...")

    # Launch kernel asynchronously on GPU
    @cuda threads=threads blocks=blocks vadd_kernel!(c_gpu, a_gpu, b_gpu)

    # Synchronize: block CPU execution until the GPU has finished executing
    CUDA.synchronize()

    # --------------------------------------------------------------------------
    # Step 4: Copy Result Back to CPU (Device-to-Host Memory Transfer)
    # --------------------------------------------------------------------------
    println("[4] Transferring result back from GPU to CPU (D2H)...")
    # Array(cu_array) downloads the device buffer over PCIe to the host CPU
    c_cpu = Array(c_gpu)

    # --------------------------------------------------------------------------
    # Step 5: Verification
    # --------------------------------------------------------------------------
    # Expected: 1.5 + 2.5 = 4.0 for all elements
    println("[5] Verifying output...")
    @test all(c_cpu .== 4.0f0)
    println("✓ Vector addition verified successfully!")
    println("Sample results (first 5 elements): ", c_cpu[1:5])
end

main()`;

  const copyScript = () => {
    navigator.clipboard.writeText(fullJuliaScript);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Terminal className="w-5 h-5 text-emerald-400" />
              Vector Addition, Memory Management &amp; Thread Sizing in CUDA.jl
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              End-to-end breakdown: CPU-to-GPU allocations (<code className="text-emerald-400 font-mono">CuArray</code>),
              1-based kernel indexing (<code className="text-purple-400 font-mono">@cuda</code>), and thread block performance tuning.
            </p>
          </div>

          <button
            onClick={copyScript}
            className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold transition-all shadow-md cursor-pointer self-start md:self-auto"
          >
            {copiedCode ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            <span>{copiedCode ? "Copied Script!" : "Copy Full Julia Script"}</span>
          </button>
        </div>
      </div>

      {/* 3 Main Pillars Navigation */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Pillar 1 */}
        <button
          onClick={() => setActiveStep(1)}
          className={`text-left p-4 rounded-2xl border transition-all ${
            activeStep === 1
              ? "bg-slate-900 border-emerald-500/50 shadow-md ring-1 ring-emerald-500/30"
              : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
          }`}
        >
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-emerald-400">
            <HardDrive className="w-4 h-4" />
            <span>1. GPU Memory Transfers</span>
          </div>
          <h3 className="text-sm font-bold text-white mt-1">Allocation &amp; PCIe Copies</h3>
          <p className="text-xs text-slate-400 mt-1">
            <code className="text-slate-300 font-mono">CuArray(cpu_arr)</code> (H2D) and <code className="text-slate-300 font-mono">Array(gpu_arr)</code> (D2H).
          </p>
        </button>

        {/* Pillar 2 */}
        <button
          onClick={() => setActiveStep(2)}
          className={`text-left p-4 rounded-2xl border transition-all ${
            activeStep === 2
              ? "bg-slate-900 border-purple-500/50 shadow-md ring-1 ring-purple-500/30"
              : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
          }`}
        >
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-purple-400">
            <Cpu className="w-4 h-4" />
            <span>2. The @cuda Kernel</span>
          </div>
          <h3 className="text-sm font-bold text-white mt-1">Thread Indexing &amp; Guards</h3>
          <p className="text-xs text-slate-400 mt-1">
            Julia's 1-based index formula, <code className="text-slate-300 font-mono">@inbounds</code>, and boundary conditions.
          </p>
        </button>

        {/* Pillar 3 */}
        <button
          onClick={() => setActiveStep(3)}
          className={`text-left p-4 rounded-2xl border transition-all ${
            activeStep === 3
              ? "bg-slate-900 border-cyan-500/50 shadow-md ring-1 ring-cyan-500/30"
              : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
          }`}
        >
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-cyan-400">
            <Sliders className="w-4 h-4" />
            <span>3. Block Size &amp; Performance</span>
          </div>
          <h3 className="text-sm font-bold text-white mt-1">Occupancy &amp; Tuning</h3>
          <p className="text-xs text-slate-400 mt-1">
            Warp multiples, register limits, and <code className="text-slate-300 font-mono">launch_configuration</code>.
          </p>
        </button>
      </div>

      {/* Deep-Dive Step 1: Memory Management */}
      {activeStep === 1 && (
        <div className="space-y-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-emerald-400" />
              Memory Allocation &amp; Copying: CPU (Host) ⇄ GPU (Device)
            </h3>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Host-to-Device (H2D) &amp; Device-to-Host (D2H)
            </span>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            In CUDA programming, the CPU (Host) and GPU (Device) maintain distinct physical memory spaces connected across the PCIe bus.
            In Julia, standard arrays (<code className="text-slate-200 font-mono">Array&#123;Float32&#125;</code>) reside in CPU RAM, while{" "}
            <code className="text-emerald-400 font-mono">CuArray&#123;Float32&#125;</code> resides in GPU High-Bandwidth VRAM.
          </p>

          {/* Interactive Memory Transfer Diagram */}
          <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 items-center gap-4 text-center">
              {/* Host RAM */}
              <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 space-y-2">
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                  Host (CPU Memory)
                </span>
                <div className="font-mono text-sm font-bold text-white">a_cpu, b_cpu</div>
                <div className="text-[11px] text-slate-500 font-mono">Array&#123;Float32&#125;(N)</div>
                <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                  Allocated in System DDR RAM
                </div>
              </div>

              {/* PCIe Bridge */}
              <div className="space-y-2">
                <div className="text-xs font-mono font-bold text-emerald-400 flex items-center justify-center gap-1.5">
                  <span>H2D: CuArray(a_cpu)</span>
                  <span className="text-slate-500">→</span>
                </div>
                <div className="h-1 bg-gradient-to-r from-slate-700 via-emerald-500 to-purple-500 rounded-full"></div>
                <div className="text-xs font-mono font-bold text-purple-400 flex items-center justify-center gap-1.5">
                  <span className="text-slate-500">←</span>
                  <span>D2H: Array(c_gpu)</span>
                </div>
                <div className="text-[10px] text-slate-500">PCIe 4.0/5.0 Bus (~25-64 GB/s)</div>
              </div>

              {/* Device VRAM */}
              <div className="bg-slate-900 p-4 rounded-xl border border-emerald-500/30 space-y-2 glow-nvidia">
                <span className="text-[11px] font-bold text-emerald-400 uppercase tracking-wider block">
                  Device (GPU VRAM)
                </span>
                <div className="font-mono text-sm font-bold text-emerald-300">a_gpu, b_gpu, c_gpu</div>
                <div className="text-[11px] text-emerald-400/80 font-mono">CuArray&#123;Float32&#125;(N)</div>
                <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800">
                  High-Bandwidth Memory ({selectedGpu.memoryBandwidthGBs} GB/s)
                </div>
              </div>
            </div>
          </div>

          {/* VRAM Budget & Allocation Calculator */}
          <div className="bg-slate-950 p-5 rounded-xl border border-purple-500/30 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-purple-300 uppercase tracking-wider flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-purple-400" />
                VRAM Capacity &amp; Allocation Limits on {selectedGpu.name}
              </span>
              <span className="text-xs font-mono font-bold text-purple-400 bg-purple-950/60 px-2.5 py-0.5 rounded border border-purple-800">
                {selectedGpu.vramMiB} MByte Total VRAM
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-1">
              <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Total Device VRAM:</span>
                <span className="font-mono font-bold text-white text-sm">
                  {selectedGpu.vramMiB} MByte ({(selectedGpu.vramMiB / 1024).toFixed(2)} GB)
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Reported by CUDA.totalmem()
                </span>
              </div>

              <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Vector Add Footprint (3 × 1M):</span>
                <span className="font-mono font-bold text-emerald-400 text-sm">
                  11.44 MB (0.28%)
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  a_gpu + b_gpu + c_gpu
                </span>
              </div>

              <div className="bg-slate-900/90 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Max Vector Elements (3 vectors):</span>
                <span className="font-mono font-bold text-cyan-400 text-sm">
                  ~315 Million Float32
                </span>
                <span className="text-[10px] text-slate-500 block mt-0.5">
                  Fits comfortably in {selectedGpu.vramMiB} MByte
                </span>
              </div>
            </div>

            <div className="text-[11px] text-slate-400 pt-1 font-mono bg-slate-900 p-2.5 rounded border border-slate-800">
              <span className="text-slate-500"># Query real-time free/used memory in Julia:</span><br/>
              <span className="text-emerald-300">CUDA.memory_status()</span><br/>
              <span className="text-slate-400"># Prints: Memory: 11.44 MB / {selectedGpu.vramMiB}.00 MB (Free: {(selectedGpu.vramMiB - 350).toFixed(0)} MB)</span>
            </div>
          </div>

          {/* Code Breakdown */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
              Essential Memory Operations
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1.5">
                <span className="font-bold text-emerald-400 font-mono">1. Copy CPU to GPU (H2D)</span>
                <pre className="text-emerald-300 font-mono text-[11px] bg-slate-900 p-2 rounded">
                  <code>a_gpu = CuArray(a_cpu)</code>
                </pre>
                <p className="text-slate-400 text-[11px]">
                  Allocates device memory and copies the CPU array contents via PCIe.
                </p>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1.5">
                <span className="font-bold text-purple-400 font-mono">2. Direct Device Allocation</span>
                <pre className="text-purple-300 font-mono text-[11px] bg-slate-900 p-2 rounded">
                  <code>c_gpu = CUDA.zeros(Float32, N)</code>
                </pre>
                <p className="text-slate-400 text-[11px]">
                  Allocates memory directly on the GPU without copying from CPU, saving PCIe bandwidth.
                </p>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1.5">
                <span className="font-bold text-cyan-400 font-mono">3. Copy GPU to CPU (D2H)</span>
                <pre className="text-cyan-300 font-mono text-[11px] bg-slate-900 p-2 rounded">
                  <code>c_cpu = Array(c_gpu)</code>
                </pre>
                <p className="text-slate-400 text-[11px]">
                  Transfers the output array back from device VRAM to host RAM for plotting or saving.
                </p>
              </div>

              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1.5">
                <span className="font-bold text-amber-400 font-mono">4. Guard Against Scalar Indexing</span>
                <pre className="text-amber-300 font-mono text-[11px] bg-slate-900 p-2 rounded">
                  <code>CUDA.allowscalar(false)</code>
                </pre>
                <p className="text-slate-400 text-[11px]">
                  Disallows accessing <code className="text-slate-300">c_gpu[1]</code> from the CPU, preventing accidental single-byte PCIe stalls.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Deep-Dive Step 2: The Kernel */}
      {activeStep === 2 && (
        <div className="space-y-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Cpu className="w-5 h-5 text-purple-400" />
              CUDA.jl Kernel Definition &amp; 1-Based Thread Indexing
            </h3>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">
              Parallel Execution Model
            </span>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            In CUDA.jl, a GPU kernel is a standard Julia function decorated and executed across thousands of threads via{" "}
            <code className="text-purple-400 font-mono">@cuda threads=... blocks=... my_kernel!(args...)</code>.
          </p>

          {/* Annotated Code Block */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3 font-mono text-xs">
            <div className="text-slate-500">// 1. Kernel Definition</div>
            <div className="text-purple-300 font-bold">function vadd_kernel!(c, a, b)</div>
            
            <div className="pl-4 space-y-2 border-l-2 border-purple-500/40">
              <div className="text-slate-500">
                # 2. 1-Based Global Index calculation:
              </div>
              <div className="text-emerald-300 font-bold bg-slate-900/80 p-2 rounded border border-slate-800">
                i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
              </div>

              <div className="text-slate-500">
                # 3. Boundary guard (threads * blocks may exceed array length N):
              </div>
              <div className="text-cyan-300">
                if i &lt;= length(c)
              </div>
              <div className="pl-4 text-amber-300">
                @inbounds c[i] = a[i] + b[i]
              </div>
              <div className="text-cyan-300">end</div>

              <div className="text-slate-500 pt-1">
                # 4. Mandatory return type for GPU kernels:
              </div>
              <div className="text-rose-300 font-bold">return nothing</div>
            </div>

            <div className="text-purple-300 font-bold">end</div>
          </div>

          {/* Key Explanations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="font-bold text-emerald-400 flex items-center gap-1.5">
                <Info className="w-4 h-4" />
                <span>Why subtract Int32(1) from blockIdx().x?</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                In C++ CUDA, <code className="text-slate-200">blockIdx.x</code> starts at 0.
                In Julia, arrays are <strong>1-based</strong>, and CUDA.jl reflects this: the first block has <code className="text-purple-300">blockIdx().x == 1</code>, and the first thread has <code className="text-emerald-300">threadIdx().x == 1</code>.
                Subtracting 1 from the block index shifts it to a 0-based offset so that Block 1 starts at index 1:
                <br />
                <code className="text-slate-300 font-mono text-[11px] block mt-1">
                  Block 1: (1 - 1) * 256 + 1 = 1 to 256
                  <br />
                  Block 2: (2 - 1) * 256 + 1 = 257 to 512
                </code>
              </p>
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <div className="font-bold text-amber-400 flex items-center gap-1.5">
                <AlertTriangle className="w-4 h-4" />
                <span>Why is the Boundary Check Essential?</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                When array length N = 1,000,000 and block size is 256, we launch ceil(1,000,000 / 256) = 3,907 blocks.
                This launches 3,907 &times; 256 = 1,000,192 threads.
                The last 192 threads have i &gt; 1,000,000. Without <code className="text-cyan-300 font-mono">if i &lt;= length(c)</code>, those 192 threads would write out of bounds, triggering a fatal GPU illegal memory access exception!
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Deep-Dive Step 3: Block Size & Performance Tuning */}
      {activeStep === 3 && (
        <div className="space-y-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-cyan-400" />
              Thread Block Size Selection &amp; Hardware Occupancy Impact
            </h3>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Hardware SM Scheduling
            </span>
          </div>

          <p className="text-xs text-slate-300 leading-relaxed">
            NVIDIA GPUs group threads into <strong>Warps</strong> of 32 threads.
            Setting the block size too low (e.g., 16 or 32) underutilizes the warp scheduler, while setting it too high (e.g., 1024) can exhaust shared memory or register files, reducing the number of concurrent blocks an SM can hold.
          </p>

          {/* Interactive Block Size Slider */}
          <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
              <div>
                <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
                  Select Thread Block Size (threads)
                </span>
                <span className="text-xs text-slate-400">
                  Drag slider to see real-time impact on warps, blocks, and occupancy on {selectedGpu.name}.
                </span>
              </div>
              <span className="font-mono text-sm font-bold text-cyan-400 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">
                {blockSize} threads ({warpsPerBlock} warps)
              </span>
            </div>

            <div className="grid grid-cols-6 gap-2">
              {[32, 64, 128, 256, 512, 1024].map((sz) => (
                <button
                  key={sz}
                  onClick={() => setBlockSize(sz)}
                  className={`py-2 rounded-xl text-xs font-mono font-bold border transition-all ${
                    blockSize === sz
                      ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/50 shadow-sm"
                      : "bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {sz}
                </button>
              ))}
            </div>

            {/* Impact Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-3 border-t border-slate-800 text-xs">
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Warps Per Block:</span>
                <span className="font-mono font-bold text-emerald-400 text-sm">
                  {warpsPerBlock} warps
                </span>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Grid Blocks (cld):</span>
                <span className="font-mono font-bold text-white text-sm">
                  {totalBlocks.toLocaleString()} blocks
                </span>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Theoretical Occupancy:</span>
                <span className="font-mono font-bold text-cyan-400 text-sm">
                  {occupancyPercent}%
                </span>
              </div>
              <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                <span className="text-slate-400 block text-[11px]">Active Warps / SM:</span>
                <span className="font-mono font-bold text-purple-400 text-sm">
                  {activeWarpsSm} / {maxWarpsSm}
                </span>
              </div>
            </div>
          </div>

          {/* Performance Guidelines Table */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
              Thread Block Size Rules of Thumb
            </h4>
            <div className="space-y-2 text-xs">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white">Always a multiple of 32:</strong> Because hardware warps are 32 threads, selecting a non-multiple (e.g. 50 threads) leaves 14 threads in the second warp inactive, wasting 28% of execution bandwidth.
                </div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white">Sweet Spot (128 - 256 threads):</strong> 256 threads per block is the golden default for 1D element-wise and reduction kernels. It allows up to 6 blocks per SM, providing enough active warps to hide memory latency without exhausting registers.
                </div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white">Automatic Tuning via launch_configuration:</strong> Instead of guessing, let CUDA.jl compute the hardware-optimal block size automatically:
                  <pre className="mt-2 text-emerald-300 font-mono text-[11px] bg-slate-900 p-2.5 rounded border border-slate-800">
                    <code>kernel = @cuda launch=false vadd_kernel!(c, a, b){"\n"}config = launch_configuration(kernel.fun){"\n"}threads = config.threads # Optimal threads/block{"\n"}blocks = min(config.blocks, cld(N, threads))</code>
                  </pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
