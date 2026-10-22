import React from "react";
import { Cpu, Zap, BookOpen, Calculator, Play, GitCompare, MessageSquare, Award, Terminal } from "lucide-react";
import { GpuSpec } from "../types";

import { CudaToolkitSpec } from "../data/cudaToolkits";

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  selectedGpu: GpuSpec;
  setSelectedGpu: (gpu: GpuSpec) => void;
  gpuSpecs: GpuSpec[];
  selectedToolkit: CudaToolkitSpec;
  setSelectedToolkit: (toolkit: CudaToolkitSpec) => void;
  toolkits: CudaToolkitSpec[];
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  selectedGpu,
  setSelectedGpu,
  gpuSpecs,
  selectedToolkit,
  setSelectedToolkit,
  toolkits,
}) => {
  const tabs = [
    { id: "vector-add", label: "Vector Add & Memory", icon: Terminal },
    { id: "simulator", label: "Kernel Simulator", icon: Play },
    { id: "occupancy", label: "Occupancy Calculator", icon: Calculator },
    { id: "cookbook", label: "CUDA.jl Recipes", icon: BookOpen },
    { id: "benchmark", label: "GPU vs CPU Benchmark", icon: Zap },
    { id: "compare", label: "Julia vs C++ / Python", icon: GitCompare },
    { id: "ai", label: "CUDA.jl AI Copilot", icon: MessageSquare },
    { id: "quiz", label: "Mastery Check", icon: Award },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur-md sticky top-0 z-50">
      {/* Top Banner */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center space-x-3.5">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-purple-600/30 border border-emerald-500/30 shadow-inner">
            <span className="text-xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-purple-400 font-mono">
              Jl
            </span>
            <div className="absolute -bottom-1 -right-1 w-3.5 h-3.5 bg-emerald-500 rounded-full border-2 border-slate-900"></div>
          </div>

          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                Julia &amp; CUDA.jl
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  GPU Acceleration Studio
                </span>
              </h1>
            </div>
            <p className="text-xs text-slate-400 flex items-center gap-1.5">
              <span>HPC</span>
              <span className="text-slate-600">•</span>
              <span className="text-emerald-400 font-mono font-semibold">CUDA {selectedToolkit.version}</span>
              <span className="text-slate-600">•</span>
              <span className="text-purple-400 font-mono">PTX ISA {selectedToolkit.ptxIsaVersion}</span>
              <span className="text-slate-600">•</span>
              <span className="text-cyan-400 font-mono">Native JIT</span>
            </p>
          </div>
        </div>

        {/* Configuration Selectors */}
        <div className="flex items-center space-x-2.5 flex-wrap">
          {/* CUDA Toolkit Selector */}
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800">
            <Zap className="w-4 h-4 text-amber-400" />
            <div className="text-xs">
              <span className="text-slate-400 block text-[10px] uppercase font-semibold tracking-wider">CUDA Toolkit</span>
              <select
                value={selectedToolkit.version}
                onChange={(e) => {
                  const found = toolkits.find((t) => t.version === e.target.value);
                  if (found) setSelectedToolkit(found);
                }}
                className="bg-transparent text-amber-300 font-mono font-bold text-xs focus:outline-none cursor-pointer pr-1"
              >
                {toolkits.map((t) => (
                  <option key={t.version} value={t.version} className="bg-slate-900 text-slate-200">
                    CUDA v{t.version} {t.isLatest ? "(Latest)" : ""}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Hardware Preset Selector */}
          <div className="flex items-center space-x-2.5 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <div className="text-xs">
              <span className="text-slate-400 block text-[10px] uppercase font-semibold tracking-wider">Target GPU</span>
              <select
                value={selectedGpu.id}
                onChange={(e) => {
                  const found = gpuSpecs.find((g) => g.id === e.target.value);
                  if (found) setSelectedGpu(found);
                }}
                className="bg-transparent text-slate-200 font-medium text-xs focus:outline-none cursor-pointer pr-1"
              >
                {gpuSpecs.map((g) => (
                  <option key={g.id} value={g.id} className="bg-slate-900 text-slate-200">
                    {g.name} (CC {g.computeCapability})
                  </option>
                ))}
              </select>
            </div>
            <div className="pl-2 border-l border-slate-800 hidden sm:block">
              <div className="text-[10px] text-slate-400">VRAM</div>
              <div className="text-xs font-mono font-semibold text-purple-400">
                {selectedGpu.vramMiB} MByte
              </div>
            </div>
            <div className="pl-2 border-l border-slate-800 hidden lg:block">
              <div className="text-[10px] text-slate-400">Bandwidth</div>
              <div className="text-xs font-mono font-semibold text-emerald-400">
                {selectedGpu.memoryBandwidthGBs} GB/s
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 border-t border-slate-800/60 overflow-x-auto">
        <nav className="flex space-x-1 py-1.5" aria-label="Tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 text-xs font-medium rounded-lg transition-all duration-150 whitespace-nowrap ${
                  isActive
                    ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? "text-emerald-400" : "text-slate-400"}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
