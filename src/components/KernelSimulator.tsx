import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, SkipForward, RotateCcw, Info, Check, Eye } from "lucide-react";

interface KernelPreset {
  id: string;
  name: string;
  description: string;
  gridDim: { x: number; y: number };
  blockDim: { x: number; y: number };
  code: string;
}

const PRESETS: KernelPreset[] = [
  {
    id: "saxpy-1d",
    name: "1D Vector Addition (SAXPY)",
    description: "Threads calculate 1-based global indices and evaluate y[i] = a * x[i] + y[i].",
    gridDim: { x: 3, y: 1 },
    blockDim: { x: 8, y: 1 }, // total 24 threads
    code: `function saxpy_kernel!(y, a, x)
    # 1-based indexing idiom:
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    
    if i <= length(y)
        @inbounds y[i] = a * x[i] + y[i]
    end
    return nothing
end`
  },
  {
    id: "matmul-2d",
    name: "2D Grid Indexing (Matrix Tile)",
    description: "2D Block and Grid coordinates mapping to row i and column j.",
    gridDim: { x: 2, y: 2 },
    blockDim: { x: 4, y: 4 }, // 16 threads per block, 4 blocks = 64 threads
    code: `function mat_kernel!(C, A, B)
    # Row and Column calculation (1-based)
    col = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    row = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    if row <= size(C, 1) && col <= size(C, 2)
        @inbounds C[row, col] = A[row, col] + B[row, col]
    end
    return nothing
end`
  },
  {
    id: "warp-reduction",
    name: "Warp Shuffle Reduction",
    description: "Threads in warp 1 use shfl_down_sync() to fold 32 elements into 1 in log2(32)=5 steps.",
    gridDim: { x: 1, y: 1 },
    blockDim: { x: 32, y: 1 }, // 1 warp (32 threads)
    code: `function warp_reduce_sum(val::Float32)
    offset = Int32(16)
    while offset > Int32(0)
        # Register-to-register exchange
        val += shfl_down_sync(0xffffffff, val, offset)
        offset >>= 1
    end
    return val
end`
  }
];

export const KernelSimulator: React.FC = () => {
  const [selectedPresetId, setSelectedPresetId] = useState<string>("saxpy-1d");
  const [activeStep, setActiveStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [selectedThread, setSelectedThread] = useState<{ block: number; thread: number } | null>(null);

  const preset = PRESETS.find((p) => p.id === selectedPresetId) || PRESETS[0];

  const totalBlocks = preset.gridDim.x * preset.gridDim.y;
  const threadsPerBlock = preset.blockDim.x * preset.blockDim.y;
  const totalThreads = totalBlocks * threadsPerBlock;

  // Max simulation steps
  const maxSteps = preset.id === "warp-reduction" ? 5 : totalThreads;

  // Playback timer
  useEffect(() => {
    let timer: any;
    if (isPlaying) {
      timer = setInterval(() => {
        setActiveStep((prev) => {
          if (prev >= maxSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 700);
    }
    return () => clearInterval(timer);
  }, [isPlaying, maxSteps]);

  const handleReset = () => {
    setIsPlaying(false);
    setActiveStep(0);
    setSelectedThread(null);
  };

  const handleStepForward = () => {
    if (activeStep < maxSteps - 1) {
      setActiveStep((prev) => prev + 1);
    }
  };

  // Inspect details of a selected thread or currently active stepping thread
  const inspectionThread = selectedThread || {
    block: Math.floor(activeStep / threadsPerBlock) + 1,
    thread: (activeStep % threadsPerBlock) + 1,
  };

  // Compute 1-based coordinates
  const bIdxX = ((inspectionThread.block - 1) % preset.gridDim.x) + 1;
  const bIdxY = Math.floor((inspectionThread.block - 1) / preset.gridDim.x) + 1;

  const tIdxX = ((inspectionThread.thread - 1) % preset.blockDim.x) + 1;
  const tIdxY = Math.floor((inspectionThread.thread - 1) / preset.blockDim.x) + 1;

  const globalIndex1D = (bIdxX - 1) * preset.blockDim.x + tIdxX;

  return (
    <div className="space-y-6">
      {/* Header card with presets */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Play className="w-5 h-5 text-emerald-400" />
              Interactive Julia CUDA Kernel Execution Simulator
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Step through GPU thread execution, inspect Julia 1-based indexing formulas, and observe memory mapping.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                onClick={() => {
                  setSelectedPresetId(p.id);
                  handleReset();
                }}
                className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                  selectedPresetId === p.id
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Visual Thread Grid */}
        <div className="lg:col-span-7 space-y-5 bg-slate-900/80 border border-slate-800 rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <span>GPU Thread Grid</span>
              <span className="text-xs font-mono font-normal text-slate-400">
                ({totalBlocks} Blocks × {threadsPerBlock} Threads = {totalThreads} Threads)
              </span>
            </h3>

            {/* Playback Controls */}
            <div className="flex items-center space-x-2">
              <button
                onClick={handleReset}
                className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors"
                title="Reset simulation"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold transition-colors"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isPlaying ? "Pause" : "Simulate"}</span>
              </button>
              <button
                onClick={handleStepForward}
                disabled={activeStep >= maxSteps - 1}
                className="p-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 rounded-lg transition-colors"
                title="Step forward"
              >
                <SkipForward className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Grid Blocks Layout */}
          <div className="space-y-4 max-h-[480px] overflow-y-auto pr-2">
            {Array.from({ length: totalBlocks }).map((_, blockIdx) => {
              const currentBlockNum = blockIdx + 1;
              return (
                <div
                  key={blockIdx}
                  className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3.5 space-y-2.5"
                >
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-mono font-bold text-purple-400 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                      Block {currentBlockNum} (blockIdx = {currentBlockNum})
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono">
                      {threadsPerBlock} threads
                    </span>
                  </div>

                  {/* Thread cells */}
                  <div
                    className="grid gap-1.5"
                    style={{
                      gridTemplateColumns: `repeat(${preset.blockDim.x}, minmax(0, 1fr))`,
                    }}
                  >
                    {Array.from({ length: threadsPerBlock }).map((_, threadIdx) => {
                      const currentThreadNum = threadIdx + 1;
                      const globalLinearThread = blockIdx * threadsPerBlock + threadIdx;
                      const isSimulated = globalLinearThread <= activeStep;
                      const isCurrentStep = globalLinearThread === activeStep;
                      const isSelected =
                        selectedThread?.block === currentBlockNum &&
                        selectedThread?.thread === currentThreadNum;

                      return (
                        <button
                          key={threadIdx}
                          onClick={() =>
                            setSelectedThread({
                              block: currentBlockNum,
                              thread: currentThreadNum,
                            })
                          }
                          className={`h-9 rounded-lg font-mono text-[11px] font-bold transition-all relative flex flex-col items-center justify-center border ${
                            isSelected
                              ? "bg-amber-500/30 text-amber-200 border-amber-400 ring-2 ring-amber-400/50"
                              : isCurrentStep
                              ? "bg-emerald-500 text-slate-950 border-emerald-300 scale-105 shadow-md shadow-emerald-500/30 z-10"
                              : isSimulated
                              ? "bg-emerald-950/60 text-emerald-400 border-emerald-800/60"
                              : "bg-slate-900 text-slate-500 border-slate-800 hover:border-slate-700"
                          }`}
                        >
                          <span>T{currentThreadNum}</span>
                          <span className="text-[9px] opacity-75 font-normal">
                            g={globalLinearThread + 1}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-4 text-xs text-slate-400 pt-2 border-t border-slate-800/80">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-emerald-500"></span> Active Executing
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-emerald-950/60 border border-emerald-800"></span> Executed
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded bg-amber-500/30 border border-amber-400"></span> Inspected Thread
            </span>
          </div>
        </div>

        {/* Right Column: Mathematical Calculation & Code Inspector */}
        <div className="lg:col-span-5 space-y-5">
          {/* Thread Coordinate Inspector */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center justify-between">
              <span>Thread Coordinate Inspector</span>
              <span className="text-xs font-mono text-emerald-400 bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                Block {inspectionThread.block}, Thread {inspectionThread.thread}
              </span>
            </h3>

            {/* Coordinates table */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                <span className="text-slate-400 block text-[11px]">blockIdx() [1-based]</span>
                <span className="text-purple-400 font-mono font-bold text-sm">
                  ({bIdxX}, {bIdxY})
                </span>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                <span className="text-slate-400 block text-[11px]">threadIdx() [1-based]</span>
                <span className="text-emerald-400 font-mono font-bold text-sm">
                  ({tIdxX}, {tIdxY})
                </span>
              </div>
            </div>

            {/* Julia 1-based indexing equation */}
            <div className="bg-slate-950 p-4 rounded-xl border border-emerald-500/30 space-y-2">
              <div className="text-[11px] font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-emerald-400" />
                Julia 1-Based Global Index Formula
              </div>
              <div className="font-mono text-xs text-emerald-300 bg-slate-900/90 p-2.5 rounded-lg border border-slate-800">
                i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
              </div>
              <div className="font-mono text-xs text-slate-300">
                i = ({bIdxX} - 1) * {preset.blockDim.x} + {tIdxX} ={" "}
                <span className="text-emerald-400 font-bold text-sm">{globalIndex1D}</span>
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed pt-1 border-t border-slate-800/80">
                In C++ CUDA, indexing starts at 0: <code className="text-slate-300">blockIdx.x * blockDim.x + threadIdx.x</code>.
                Julia's 1-based convention requires subtracting 1 from <code className="text-purple-300">blockIdx().x</code> before multiplying by <code className="text-slate-300">blockDim().x</code>!
              </p>
            </div>
          </div>

          {/* Julia Kernel Code Viewer */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-400"></span>
              Executed Kernel Code
            </h4>
            <pre className="bg-slate-950 p-4 rounded-xl text-xs font-mono text-purple-300/90 overflow-x-auto border border-slate-800/80 leading-relaxed">
              <code>{preset.code}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
