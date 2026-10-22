import React, { useState } from "react";
import { MessageSquare, Sparkles, Send, Copy, Check, Terminal, HelpCircle, Code2, AlertTriangle } from "lucide-react";

interface AiAssistantProps {
  initialPrompt?: string;
  initialCode?: string;
}

export const AiAssistant: React.FC<AiAssistantProps> = ({ initialPrompt = "", initialCode = "" }) => {
  const [prompt, setPrompt] = useState<string>(initialPrompt);
  const [codeSnippet, setCodeSnippet] = useState<string>(initialCode);
  const [taskType, setTaskType] = useState<string>("kernel_optimization");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<string | null>(null);
  const [isCopied, setIsCopied] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Quick preset queries
  const promptPresets = [
    {
      label: "Fix Scalar Indexing",
      type: "debug_scalar",
      prompt: "I am getting 'ERROR: Scalar indexing is disallowed'. How do I fix this loop in Julia CUDA.jl?",
      code: `using CUDA
CUDA.allowscalar(false)

function my_loop(A)
    # Slow scalar loop
    for i in 1:length(A)
        A[i] = A[i] * 2.0f0
    end
end`,
    },
    {
      label: "Shared Memory 2D Tiling",
      type: "kernel_gen",
      prompt: "Show me how to write an efficient 2D matrix multiplication kernel in Julia using @cuStaticSharedMem and sync_threads().",
      code: ``,
    },
    {
      label: "Parallel Reduction",
      type: "kernel_gen",
      prompt: "How can I implement a logarithmic parallel sum reduction in Julia CUDA.jl using warp shuffles (shfl_down_sync)?",
      code: ``,
    },
    {
      label: "PyTorch to Julia CUDA",
      type: "convert",
      prompt: "Convert this PyTorch custom GPU tensor operation into idiomatic Julia CUDA.jl with fused broadcasting and a custom @cuda kernel fallback.",
      code: `import torch
def my_op(x, y):
    mask = x > 0.5
    return torch.where(mask, x * 2.0 + y, y ** 2)`,
    },
    {
      label: "Memory Coalescing & Profiling",
      type: "profiling",
      prompt: "What are the rules for memory coalescing in CUDA.jl, and how do I profile my Julia kernel using Nsight Systems and @device_code_ptx?",
      code: ``,
    },
  ];

  const handlePresetClick = (preset: typeof promptPresets[0]) => {
    setPrompt(preset.prompt);
    setCodeSnippet(preset.code);
    setTaskType(preset.type);
  };

  const handleSend = async () => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    setErrorMsg(null);
    setResponse(null);

    try {
      const res = await fetch("/api/cuda-ai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          taskType,
          codeSnippet: codeSnippet.trim() ? codeSnippet : undefined,
        }),
      });

      const data = await res.json();
      if (data.error) {
        setErrorMsg(data.error);
      } else {
        setResponse(data.response);
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to reach AI service.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyResponse = () => {
    if (!response) return;
    navigator.clipboard.writeText(response);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Overview */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-400" />
              Julia &amp; CUDA.jl AI Copilot &amp; Kernel Optimizer
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Ask deep questions about GPU architecture, debug scalar indexing traps, convert PyTorch/C++ code, and generate optimal Julia CUDA kernels.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono bg-slate-950 px-3 py-1.5 rounded-xl border border-slate-800 text-purple-400">
            <span>Powered by Gemini 3.8 Flash</span>
          </div>
        </div>
      </div>

      {/* Preset Buttons */}
      <div className="space-y-2">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
          High-Yield Julia CUDA.jl Presets:
        </span>
        <div className="flex flex-wrap gap-2">
          {promptPresets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => handlePresetClick(p)}
              className="text-xs px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-300 hover:text-white rounded-xl border border-slate-800 transition-all font-medium flex items-center gap-1.5"
            >
              <Code2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>{p.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4">
        {/* Prompt Input */}
        <div className="space-y-1.5">
          <label className="text-xs font-semibold text-slate-300">Your Julia / CUDA.jl Question or Instruction</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="e.g. How do I optimize this 2D finite difference wave equation in CUDA.jl with shared memory?"
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-purple-500/50 resize-y"
          />
        </div>

        {/* Optional Code Snippet Input */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center text-xs">
            <label className="text-slate-300 font-semibold flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-slate-400" />
              <span>Optional Code Context or Stack Trace</span>
            </label>
            {codeSnippet && (
              <button
                onClick={() => setCodeSnippet("")}
                className="text-[11px] text-slate-500 hover:text-slate-300"
              >
                Clear
              </button>
            )}
          </div>
          <textarea
            value={codeSnippet}
            onChange={(e) => setCodeSnippet(e.target.value)}
            rows={4}
            placeholder="# Paste your Julia code, C++ kernel, or CUDA stack trace here..."
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-emerald-300/90 placeholder-slate-600 focus:outline-none focus:border-purple-500/50 resize-y"
          />
        </div>

        {/* Submit */}
        <div className="flex justify-end items-center gap-3 pt-2">
          <button
            onClick={handleSend}
            disabled={isLoading || !prompt.trim()}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-emerald-500 to-purple-600 hover:from-emerald-400 hover:to-purple-500 disabled:opacity-50 text-white font-semibold text-xs rounded-xl shadow-md transition-all cursor-pointer"
          >
            {isLoading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                <span>Analyzing GPU Architecture...</span>
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5" />
                <span>Consult Julia CUDA Expert</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error state */}
      {errorMsg && (
        <div className="bg-rose-500/10 border border-rose-500/30 p-4 rounded-xl flex items-center gap-3 text-xs text-rose-300">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* AI Response Display */}
      {response && (
        <div className="bg-slate-900/90 border border-purple-500/30 rounded-2xl p-5 space-y-4 shadow-lg glow-julia">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
              <h3 className="text-sm font-bold text-white">Julia &amp; CUDA.jl Solution &amp; Recommendations</h3>
            </div>
            <button
              onClick={handleCopyResponse}
              className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 px-3 py-1.5 rounded-lg transition-colors"
            >
              {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{isCopied ? "Copied" : "Copy Solution"}</span>
            </button>
          </div>

          {/* Formatted Markdown / Code Output */}
          <div className="prose prose-invert max-w-none text-xs text-slate-300 space-y-3 leading-relaxed whitespace-pre-wrap font-sans">
            {response}
          </div>
        </div>
      )}
    </div>
  );
};
