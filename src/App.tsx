import React, { useState } from "react";
import { GPU_SPECS } from "./data/gpuSpecs";
import { CUDA_TOOLKITS, CudaToolkitSpec } from "./data/cudaToolkits";
import { GpuSpec } from "./types";
import { Header } from "./components/Header";
import { VectorAddDeepDive } from "./components/VectorAddDeepDive";
import { KernelSimulator } from "./components/KernelSimulator";
import { OccupancyCalculator } from "./components/OccupancyCalculator";
import { CookbookViewer } from "./components/CookbookViewer";
import { BenchmarkVisualizer } from "./components/BenchmarkVisualizer";
import { CodeComparator } from "./components/CodeComparator";
import { AiAssistant } from "./components/AiAssistant";
import { QuizDiagnostic } from "./components/QuizDiagnostic";
import { Sparkles, Terminal, ShieldCheck, Zap, HardDrive, Layers } from "lucide-react";

export default function App() {
  const [selectedGpu, setSelectedGpu] = useState<GpuSpec>(
    GPU_SPECS.find((g) => g.id === "ampere-rtx3050") || GPU_SPECS[0]
  );
  const [selectedToolkit, setSelectedToolkit] = useState<CudaToolkitSpec>(CUDA_TOOLKITS[0]); // 13.1.1
  const [activeTab, setActiveTab] = useState<string>("vector-add");

  // State passed to AI Copilot from recipe book or quick actions
  const [aiPrompt, setAiPrompt] = useState<string>("");
  const [aiCode, setAiCode] = useState<string>("");

  const handleSelectRecipeForAi = (prompt: string, code: string) => {
    setAiPrompt(prompt);
    setAiCode(code);
    setActiveTab("ai");
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-emerald-500/30 selection:text-emerald-200">
      {/* Top Navigation & GPU / Toolkit Selector */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        selectedGpu={selectedGpu}
        setSelectedGpu={setSelectedGpu}
        gpuSpecs={GPU_SPECS}
        selectedToolkit={selectedToolkit}
        setSelectedToolkit={setSelectedToolkit}
        toolkits={CUDA_TOOLKITS}
      />

      {/* Hardware & Toolkit Telemetry Bar */}
      <section className="bg-slate-900/40 border-b border-slate-800/80 py-2.5 px-4 sm:px-6 lg:px-8 text-xs">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3 text-slate-400">
          <div className="flex items-center gap-4 flex-wrap">
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>CUDA.functional() == true</span>
            </span>
            <span className="text-slate-700 hidden sm:inline">|</span>
            <span className="flex items-center gap-1 font-mono text-amber-300">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Toolkit:</span>
              <strong className="text-amber-200">v{selectedToolkit.version}</strong>
              <span className="text-slate-500">({selectedToolkit.juliaArtifactVersion})</span>
            </span>
            <span className="text-slate-700 hidden sm:inline">|</span>
            <span className="flex items-center gap-1 font-mono text-slate-300">
              <HardDrive className="w-3.5 h-3.5 text-slate-400" />
              <span>GPU:</span>
              <strong className="text-white">{selectedGpu.name}</strong>
            </span>
            <span className="text-slate-700 hidden md:inline">|</span>
            <span className="text-slate-400 hidden lg:inline text-[11px]">
              {selectedGpu.notes}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                setAiPrompt(`Can you help optimize a Julia CUDA kernel for ${selectedGpu.name} running on CUDA Toolkit ${selectedToolkit.version}?`);
                setAiCode("");
                setActiveTab("ai");
              }}
              className="flex items-center gap-1.5 text-[11px] text-purple-300 hover:text-white bg-purple-950/40 hover:bg-purple-900/50 border border-purple-800/50 px-2.5 py-1 rounded-md transition-all"
            >
              <Sparkles className="w-3 h-3 text-purple-400" />
              <span>Ask Copilot</span>
            </button>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "vector-add" && <VectorAddDeepDive selectedGpu={selectedGpu} />}
        {activeTab === "simulator" && <KernelSimulator />}
        {activeTab === "occupancy" && <OccupancyCalculator selectedGpu={selectedGpu} />}
        {activeTab === "cookbook" && <CookbookViewer onSelectForAi={handleSelectRecipeForAi} />}
        {activeTab === "benchmark" && <BenchmarkVisualizer selectedGpu={selectedGpu} />}
        {activeTab === "compare" && <CodeComparator />}
        {activeTab === "ai" && <AiAssistant initialPrompt={aiPrompt} initialCode={aiCode} />}
        {activeTab === "quiz" && <QuizDiagnostic />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/90 py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-slate-400">Julia &amp; CUDA.jl GPU Studio</span>
            <span>•</span>
            <span>LLVM PTX JIT Compilation</span>
            <span>•</span>
            <span className="font-mono text-emerald-400">CUDA.allowscalar(false)</span>
          </div>
          <div className="flex items-center space-x-4 text-slate-400">
            <span>Julia 1.10+ / 1.11+</span>
            <span>•</span>
            <span>CUDA.jl v5.x</span>
            <span>•</span>
            <button
              onClick={() => setActiveTab("quiz")}
              className="hover:text-emerald-400 transition-colors"
            >
              Mastery Quiz
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
