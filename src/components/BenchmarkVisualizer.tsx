import React, { useState, useMemo } from "react";
import { GpuSpec } from "../types";
import { Zap, TrendingUp, Cpu, HardDrive, Info, ArrowRight } from "lucide-react";

interface BenchmarkVisualizerProps {
  selectedGpu: GpuSpec;
}

export const BenchmarkVisualizer: React.FC<BenchmarkVisualizerProps> = ({ selectedGpu }) => {
  const [arraySize, setArraySize] = useState<number>(10_000_000);
  const [precision, setPrecision] = useState<"Float32" | "Float64">("Float32");
  const [computeIntensity, setComputeIntensity] = useState<"low" | "medium" | "high">("medium");

  const benchmarks = useMemo(() => {
    const bytesPerElement = precision === "Float32" ? 4 : 8;
    const totalBytes = arraySize * bytesPerElement;
    const totalGB = totalBytes / 1e9;

    // Flops per element multiplier
    const flopsPerElem = computeIntensity === "low" ? 4 : computeIntensity === "medium" ? 24 : 128;
    const totalGFlops = (arraySize * flopsPerElem) / 1e9;

    // CPU Assumptions (Modern 16-core CPU @ 4.5GHz)
    const cpuSingleCoreGFlops = precision === "Float32" ? 80 : 40;
    const cpuMultiCoreGFlops = cpuSingleCoreGFlops * 14; // ~14x scaling
    const cpuMemoryBandwidth = 75; // 75 GB/s DDR5

    // 1. Single CPU execution time (ms)
    const cpuComputeMs = (totalGFlops / cpuSingleCoreGFlops) * 1000;
    const cpuMemoryMs = (totalGB / cpuMemoryBandwidth) * 1000;
    const cpuSingleTimeMs = Math.max(cpuComputeMs, cpuMemoryMs);

    // 2. Multithreaded CPU time (ms)
    const cpuMultiComputeMs = (totalGFlops / cpuMultiCoreGFlops) * 1000;
    const cpuMultiTimeMs = Math.max(cpuMultiComputeMs, cpuMemoryMs * 0.9) + 0.05;

    // GPU PCIe 4.0 x16 transfer speed (~25 GB/s host-to-device)
    const pcieBandwidthGBs = 25;
    const pcieTransferTimeMs = (totalGB / pcieBandwidthGBs) * 2000; // round-trip copy-in + copy-out
    const kernelLaunchOverheadMs = 0.008; // ~8 microseconds CUDA launch overhead

    // Peak GPU GFlops adjusted for precision
    const gpuTFlops = precision === "Float32" ? selectedGpu.fp32TFlops : selectedGpu.fp64TFlops;
    const gpuGFlops = gpuTFlops * 1000;
    const gpuBandwidth = selectedGpu.memoryBandwidthGBs;

    // 3. Pure GPU Compute Time (CuArray already in VRAM)
    const gpuComputeTimeMs = Math.max(
      (totalGFlops / gpuGFlops) * 1000,
      (totalGB / gpuBandwidth) * 1000
    ) + kernelLaunchOverheadMs;

    // 4. End-to-End GPU Time (with Host-to-Device + Device-to-Host transfers)
    const gpuEndToEndTimeMs = gpuComputeTimeMs + pcieTransferTimeMs;

    // Speedups
    const speedupComputeOnly = (cpuSingleTimeMs / gpuComputeTimeMs).toFixed(1);
    const speedupEndToEnd = (cpuSingleTimeMs / gpuEndToEndTimeMs).toFixed(1);

    return {
      totalBytes,
      totalGFlops,
      cpuSingleTimeMs,
      cpuMultiTimeMs,
      gpuComputeTimeMs,
      gpuEndToEndTimeMs,
      speedupComputeOnly,
      speedupEndToEnd,
      pcieTransferTimeMs,
    };
  }, [arraySize, precision, computeIntensity, selectedGpu]);

  return (
    <div className="space-y-6">
      {/* Overview card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-emerald-400" />
              Julia GPU vs CPU Scaling &amp; PCIe Latency Analyzer
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Simulate performance gains of <span className="text-emerald-400 font-mono">CuArray</span> vs{" "}
              <span className="text-purple-400 font-mono">Threads.@threads</span> across data sizes and PCIe overhead.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-950 p-2 rounded-xl border border-slate-800 text-xs">
            <span className="text-slate-400">Peak GPU Throughput:</span>
            <span className="text-emerald-400 font-mono font-bold">
              {precision === "Float32" ? selectedGpu.fp32TFlops : selectedGpu.fp64TFlops} TFLOPS
            </span>
            <span className="text-slate-600">|</span>
            <span className="text-cyan-400 font-mono font-bold">{selectedGpu.memoryBandwidthGBs} GB/s</span>
          </div>
        </div>
      </div>

      {/* Control sliders */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Array Size */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2">
          <label className="text-xs font-semibold text-slate-300 block">Elements (N)</label>
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-emerald-400 font-bold">{arraySize.toLocaleString()}</span>
            <span className="text-slate-500">
              {((arraySize * (precision === "Float32" ? 4 : 8)) / 1024 / 1024).toFixed(1)} MB
            </span>
          </div>
          <div className="grid grid-cols-4 gap-1 pt-1">
            {[10_000, 100_000, 1_000_000, 10_000_000].map((size) => (
              <button
                key={size}
                onClick={() => setArraySize(size)}
                className={`text-[10px] py-1 rounded font-mono border transition-all ${
                  arraySize === size
                    ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                    : "bg-slate-950 text-slate-400 border-slate-800"
                }`}
              >
                {size >= 1_000_000 ? `${size / 1_000_000}M` : `${size / 1_000}k`}
              </button>
            ))}
          </div>
        </div>

        {/* Precision */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2">
          <label className="text-xs font-semibold text-slate-300 block">Number Precision</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setPrecision("Float32")}
              className={`p-2 rounded-xl text-xs font-mono font-bold border transition-all flex flex-col items-center ${
                precision === "Float32"
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/40"
                  : "bg-slate-950 text-slate-400 border-slate-800"
              }`}
            >
              <span>Float32</span>
              <span className="text-[10px] text-emerald-400 font-normal">Max Consumer Speed</span>
            </button>
            <button
              onClick={() => setPrecision("Float64")}
              className={`p-2 rounded-xl text-xs font-mono font-bold border transition-all flex flex-col items-center ${
                precision === "Float64"
                  ? "bg-purple-500/20 text-purple-300 border-purple-500/40"
                  : "bg-slate-950 text-slate-400 border-slate-800"
              }`}
            >
              <span>Float64</span>
              <span className="text-[10px] text-purple-400 font-normal">Double Precision</span>
            </button>
          </div>
        </div>

        {/* Computation Intensity */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 space-y-2">
          <label className="text-xs font-semibold text-slate-300 block">Arithmetic Intensity</label>
          <div className="grid grid-cols-3 gap-1.5">
            {[
              { id: "low", label: "Low", desc: "x + y" },
              { id: "medium", label: "Med", desc: "Fused Poly" },
              { id: "high", label: "High", desc: "Trig/Exp" },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setComputeIntensity(item.id as any)}
                className={`py-1.5 rounded-lg text-xs font-medium border text-center transition-all ${
                  computeIntensity === item.id
                    ? "bg-cyan-500/20 text-cyan-300 border-cyan-500/40"
                    : "bg-slate-950 text-slate-400 border-slate-800"
                }`}
              >
                <div className="font-bold">{item.label}</div>
                <div className="text-[9px] text-slate-500">{item.desc}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Results Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Single CPU */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Single CPU</span>
            <Cpu className="w-4 h-4 text-slate-500" />
          </div>
          <div className="text-2xl font-black font-mono text-white">
            {benchmarks.cpuSingleTimeMs.toFixed(2)} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="text-xs text-slate-400">Baseline 1.0x (1 CPU Core, standard Julia loop)</div>
        </div>

        {/* Multi-thread CPU */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">CPU Threads (16)</span>
            <Cpu className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-black font-mono text-purple-300">
            {benchmarks.cpuMultiTimeMs.toFixed(2)} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="text-xs text-slate-400">
            {(benchmarks.cpuSingleTimeMs / benchmarks.cpuMultiTimeMs).toFixed(1)}x faster via{" "}
            <code className="text-purple-300 font-mono">Threads.@threads</code>
          </div>
        </div>

        {/* GPU End-to-End (PCIe + Compute) */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-cyan-400 uppercase tracking-wider">GPU + PCIe Transfer</span>
            <HardDrive className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-black font-mono text-cyan-300">
            {benchmarks.gpuEndToEndTimeMs.toFixed(2)} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="text-xs text-slate-400">
            <span className="text-cyan-400 font-bold font-mono">{benchmarks.speedupEndToEnd}x</span> speedup (includes{" "}
            {benchmarks.pcieTransferTimeMs.toFixed(2)}ms PCIe transfer)
          </div>
        </div>

        {/* Pure GPU In-VRAM */}
        <div className="bg-gradient-to-br from-slate-900/90 to-emerald-950/40 border border-emerald-500/30 rounded-2xl p-5 space-y-3 glow-nvidia">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider">GPU Resident (VRAM)</span>
            <Zap className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-black font-mono text-emerald-300">
            {benchmarks.gpuComputeTimeMs.toFixed(3)} <span className="text-xs font-normal text-slate-400">ms</span>
          </div>
          <div className="text-xs text-emerald-400 font-semibold">
            <span className="text-emerald-300 font-bold font-mono">{benchmarks.speedupComputeOnly}x</span> faster than single-core CPU!
          </div>
        </div>
      </div>

      {/* Break-even & Architectural Insight */}
      <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-3">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Info className="w-4 h-4 text-emerald-400" />
          The Break-Even Principle in Julia CUDA Computing
        </h3>
        <p className="text-xs text-slate-400 leading-relaxed">
          NVIDIA GPUs excel when data remains in GPU High Bandwidth Memory (<span className="text-emerald-400 font-mono">CuArray</span>).
          Notice how the PCIe round-trip transfer ({benchmarks.pcieTransferTimeMs.toFixed(2)} ms) takes far longer than the actual GPU kernel calculation ({benchmarks.gpuComputeTimeMs.toFixed(3)} ms).
          To achieve the maximum {benchmarks.speedupComputeOnly}x speedup, structure your Julia pipelines so data is loaded onto the GPU once, operated on across multiple iterations or steps, and only downloaded when finished!
        </p>
      </div>
    </div>
  );
};
