import React, { useState, useMemo } from "react";
import { GpuSpec, OccupancyResult } from "../types";
import { Activity, AlertTriangle, CheckCircle, Info, Copy, Check } from "lucide-react";

interface OccupancyCalculatorProps {
  selectedGpu: GpuSpec;
}

export const OccupancyCalculator: React.FC<OccupancyCalculatorProps> = ({ selectedGpu }) => {
  const [threadsPerBlock, setThreadsPerBlock] = useState<number>(256);
  const [registersPerThread, setRegistersPerThread] = useState<number>(32);
  const [sharedMemBytes, setSharedMemBytes] = useState<number>(0);
  const [arrayElements, setArrayElements] = useState<number>(1_000_000);
  const [copied, setCopied] = useState<boolean>(false);

  // Compute CUDA Occupancy metrics
  const occupancyData: OccupancyResult = useMemo(() => {
    const warpSize = 32;
    const warpsPerBlock = Math.ceil(threadsPerBlock / warpSize);

    // Register allocation granularity (usually 256 registers on modern architectures)
    const regGranularity = 256;
    const regPerWarp = Math.ceil((registersPerThread * warpSize) / regGranularity) * regGranularity;
    const regPerBlock = regPerWarp * warpsPerBlock;

    // Shared memory allocation granularity (typically 256 or 512 bytes)
    const shmemGranularity = 256;
    const shmemPerBlockAlloc = Math.ceil(sharedMemBytes / shmemGranularity) * shmemGranularity;

    // 1. Block limit by hardware max blocks per SM
    const limitByHardwareBlocks = selectedGpu.maxBlocksPerSm;

    // 2. Block limit by threads/warps per SM
    const limitByWarps = Math.floor(selectedGpu.maxWarpsPerSm / warpsPerBlock);

    // 3. Block limit by registers
    const limitByRegisters = regPerBlock > 0
      ? Math.floor(selectedGpu.maxRegistersPerSm / regPerBlock)
      : selectedGpu.maxBlocksPerSm;

    // 4. Block limit by shared memory
    const maxShmemBytes = selectedGpu.maxSharedMemoryPerSmKb * 1024;
    const limitByShmem = shmemPerBlockAlloc > 0
      ? Math.floor(maxShmemBytes / shmemPerBlockAlloc)
      : selectedGpu.maxBlocksPerSm;

    // Active blocks is minimum of all limits
    const activeBlocksPerSm = Math.min(
      limitByHardwareBlocks,
      limitByWarps,
      limitByRegisters,
      limitByShmem
    );

    const activeWarpsPerSm = activeBlocksPerSm * warpsPerBlock;
    const theoreticalOccupancy = Math.min(
      100,
      Math.round((activeWarpsPerSm / selectedGpu.maxWarpsPerSm) * 100)
    );

    // Determine limiting factor
    let limitingFactor: OccupancyResult["limitingFactor"] = "Optimal";
    if (activeBlocksPerSm === 0) {
      if (shmemPerBlockAlloc > maxShmemBytes) limitingFactor = "Shared Memory";
      else if (regPerBlock > selectedGpu.maxRegistersPerSm) limitingFactor = "Registers";
      else limitingFactor = "Warps";
    } else if (activeBlocksPerSm === limitByRegisters && limitByRegisters < limitByWarps) {
      limitingFactor = "Registers";
    } else if (activeBlocksPerSm === limitByShmem && limitByShmem < limitByWarps) {
      limitingFactor = "Shared Memory";
    } else if (activeBlocksPerSm === limitByHardwareBlocks && limitByHardwareBlocks < limitByWarps) {
      limitingFactor = "Blocks";
    } else if (activeBlocksPerSm === limitByWarps) {
      limitingFactor = theoreticalOccupancy >= 95 ? "Optimal" : "Warps";
    }

    const totalBlocks = Math.ceil(arrayElements / threadsPerBlock);

    const warnings: string[] = [];
    if (threadsPerBlock % 32 !== 0) {
      warnings.push("Thread block size is not a multiple of 32 (Warp Size). Unused threads in the final warp will cause warp divergence!");
    }
    if (theoreticalOccupancy < 50) {
      warnings.push(`Low GPU Occupancy (${theoreticalOccupancy}%). The GPU execution pipelines may stall waiting for memory latency.`);
    }
    if (sharedMemBytes > selectedGpu.maxSharedMemoryPerBlockKb * 1024) {
      warnings.push(`Shared memory per block exceeds device limit (${selectedGpu.maxSharedMemoryPerBlockKb} KB)! Kernel will fail to launch.`);
    }
    if (registersPerThread > 64) {
      warnings.push("High register usage per thread (>64) significantly limits block concurrency per SM.");
    }

    return {
      threadsPerBlock,
      totalBlocks,
      activeWarpsPerSm,
      theoreticalOccupancy,
      limitingFactor,
      activeBlocksPerSm,
      sharedMemPerBlock: shmemPerBlockAlloc,
      registersPerBlock: regPerBlock,
      warpAllocation: warpsPerBlock,
      warnings,
    };
  }, [threadsPerBlock, registersPerThread, sharedMemBytes, arrayElements, selectedGpu]);

  // Generated Julia Code snippet
  const juliaLaunchCode = `# Julia CUDA.jl Launch Configuration for ${selectedGpu.name}
using CUDA

N = ${arrayElements.toLocaleString("en-US").replace(/,/g, "_")}
threads = ${threadsPerBlock}
blocks = cld(N, threads) # = ${occupancyData.totalBlocks} blocks

# Kernel invocation:
# @cuda threads=threads blocks=blocks ${sharedMemBytes > 0 ? `shmem=${sharedMemBytes} ` : ""}your_kernel!(output, input)
CUDA.synchronize()

# Dynamic driver-optimal configuration:
# kernel = @cuda launch=false your_kernel!(output, input)
# opt_config = launch_configuration(kernel.fun)
# threads = opt_config.threads
# blocks = min(opt_config.blocks, cld(N, threads))`;

  const copyCode = () => {
    navigator.clipboard.writeText(juliaLaunchCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Intro card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              NVIDIA GPU Occupancy &amp; Grid Sizing Calculator
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Analyze Streaming Multiprocessor (SM) warp scheduling, register pressure, and shared memory limits on{" "}
              <span className="text-emerald-400 font-semibold">{selectedGpu.name}</span>.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono bg-slate-950 px-3 py-2 rounded-xl border border-slate-800">
            <span className="text-slate-400">SMs:</span>
            <span className="text-emerald-400 font-bold">CC {selectedGpu.computeCapability}</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-400">Max Warps/SM:</span>
            <span className="text-slate-200">{selectedGpu.maxWarpsPerSm}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive Controls */}
        <div className="lg:col-span-5 space-y-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <span>Kernel Launch Parameters</span>
          </h3>

          {/* Threads per Block */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="text-slate-300 font-medium">Threads per Block</label>
              <span className="font-mono text-emerald-400 font-bold bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                {threadsPerBlock} threads ({threadsPerBlock / 32} warps)
              </span>
            </div>
            <input
              type="range"
              min={32}
              max={1024}
              step={32}
              value={threadsPerBlock}
              onChange={(e) => setThreadsPerBlock(parseInt(e.target.value))}
              className="w-full accent-emerald-500 bg-slate-800 cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>32 (1 warp)</span>
              <span>256 (Common)</span>
              <span>512</span>
              <span>1024 (Max)</span>
            </div>
          </div>

          {/* Registers per Thread */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="text-slate-300 font-medium">Registers per Thread</label>
              <span className="font-mono text-cyan-400 font-bold bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                {registersPerThread} regs
              </span>
            </div>
            <input
              type="range"
              min={16}
              max={128}
              step={4}
              value={registersPerThread}
              onChange={(e) => setRegistersPerThread(parseInt(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-800 cursor-pointer"
            />
            <p className="text-[11px] text-slate-400">
              Queried via <code className="text-slate-300 font-mono">CUDA.@device_code_ptx</code>. Lower registers allow higher occupancy.
            </p>
          </div>

          {/* Shared Memory per Block */}
          <div className="space-y-2">
            <div className="flex justify-between items-center text-xs">
              <label className="text-slate-300 font-medium">Shared Memory per Block</label>
              <span className="font-mono text-purple-400 font-bold bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                {(sharedMemBytes / 1024).toFixed(1)} KB ({sharedMemBytes} B)
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={selectedGpu.maxSharedMemoryPerBlockKb * 1024}
              step={1024}
              value={sharedMemBytes}
              onChange={(e) => setSharedMemBytes(parseInt(e.target.value))}
              className="w-full accent-purple-500 bg-slate-800 cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-500 font-mono">
              <span>0 KB</span>
              <span>16 KB</span>
              <span>32 KB</span>
              <span>{selectedGpu.maxSharedMemoryPerBlockKb} KB (Max)</span>
            </div>
          </div>

          {/* Total Array Elements N */}
          <div className="space-y-2 pt-2 border-t border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <label className="text-slate-300 font-medium">Array Elements (N)</label>
              <span className="font-mono text-amber-400 font-bold bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                {arrayElements.toLocaleString()}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {[65536, 1000000, 10000000, 50000000].map((preset) => (
                <button
                  key={preset}
                  onClick={() => setArrayElements(preset)}
                  className={`text-[11px] py-1 rounded font-mono border transition-all ${
                    arrayElements === preset
                      ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                      : "bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200"
                  }`}
                >
                  {preset >= 1000000 ? `${preset / 1000000}M` : `${preset / 1000}k`}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Computed Metrics & Occupancy Gauge */}
        <div className="lg:col-span-7 space-y-5">
          {/* Occupancy Gauge & Primary Stats */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-6">
              {/* Circular Occupancy Progress */}
              <div className="flex items-center gap-5">
                <div className="relative w-28 h-28 flex items-center justify-center">
                  <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      className="text-slate-800"
                      strokeWidth="10"
                      stroke="currentColor"
                      fill="transparent"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      className={`transition-all duration-500 ${
                        occupancyData.theoreticalOccupancy >= 75
                          ? "text-emerald-500"
                          : occupancyData.theoreticalOccupancy >= 50
                          ? "text-amber-500"
                          : "text-rose-500"
                      }`}
                      strokeWidth="10"
                      strokeDasharray={251.2}
                      strokeDashoffset={251.2 - (251.2 * occupancyData.theoreticalOccupancy) / 100}
                      strokeLinecap="round"
                      stroke="currentColor"
                      fill="transparent"
                    />
                  </svg>
                  <div className="absolute flex flex-col items-center">
                    <span className="text-2xl font-black font-mono text-white">
                      {occupancyData.theoreticalOccupancy}%
                    </span>
                    <span className="text-[10px] uppercase font-semibold text-slate-400 tracking-wider">
                      Occupancy
                    </span>
                  </div>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase text-slate-400 tracking-wider">
                    Limiting Constraint
                  </div>
                  <div className="text-lg font-bold text-white flex items-center gap-2 mt-0.5">
                    {occupancyData.limitingFactor === "Optimal" ? (
                      <span className="text-emerald-400 flex items-center gap-1.5">
                        <CheckCircle className="w-4 h-4" /> Balanced &amp; Optimal
                      </span>
                    ) : (
                      <span className="text-amber-400 flex items-center gap-1.5">
                        <AlertTriangle className="w-4 h-4" /> Limited by {occupancyData.limitingFactor}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-1 max-w-xs">
                    {occupancyData.limitingFactor === "Registers" &&
                      "SM register file capacity limits concurrent thread blocks. Reduce per-thread local variables."}
                    {occupancyData.limitingFactor === "Shared Memory" &&
                      "Shared memory allocated per block restricts SM concurrency. Consider smaller tiles."}
                    {occupancyData.limitingFactor === "Warps" &&
                      "Block dimensions constrain the maximum active warps per Streaming Multiprocessor."}
                    {occupancyData.limitingFactor === "Blocks" &&
                      "Hardware SM block limit reached (max " + selectedGpu.maxBlocksPerSm + " blocks/SM)."}
                    {occupancyData.limitingFactor === "Optimal" &&
                      "High SM saturation. Memory latency can be hidden by active warp concurrency."}
                  </p>
                </div>
              </div>

              {/* Grid Statistics Badge */}
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2 w-full sm:w-auto">
                <div className="flex justify-between items-center text-xs gap-4">
                  <span className="text-slate-400">Grid Blocks (cld):</span>
                  <span className="font-mono font-bold text-emerald-400">{occupancyData.totalBlocks.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center text-xs gap-4">
                  <span className="text-slate-400">Active Warps / SM:</span>
                  <span className="font-mono font-bold text-slate-200">
                    {occupancyData.activeWarpsPerSm} / {selectedGpu.maxWarpsPerSm}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs gap-4">
                  <span className="text-slate-400">Active Blocks / SM:</span>
                  <span className="font-mono font-bold text-slate-200">
                    {occupancyData.activeBlocksPerSm} / {selectedGpu.maxBlocksPerSm}
                  </span>
                </div>
              </div>
            </div>

            {/* Warnings list */}
            {occupancyData.warnings.length > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-800 space-y-2">
                {occupancyData.warnings.map((w, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 px-3 py-2 rounded-lg">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Julia Launch Code Generator */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                Generated Julia Launch Configuration
              </h4>
              <button
                onClick={copyCode}
                className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded-md transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? "Copied!" : "Copy Code"}</span>
              </button>
            </div>
            <pre className="bg-slate-950 p-4 rounded-xl text-xs font-mono text-emerald-300/90 overflow-x-auto border border-slate-800/80 leading-relaxed">
              <code>{juliaLaunchCode}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
