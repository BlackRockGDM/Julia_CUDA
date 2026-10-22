import React, { useState } from "react";
import { CODE_COMPARISONS } from "../data/comparisonsData";
import { GitCompare, Copy, Check, CheckCircle2 } from "lucide-react";

export const CodeComparator: React.FC = () => {
  const [selectedId, setSelectedId] = useState<string>("vector-add");
  const [copiedLang, setCopiedLang] = useState<string | null>(null);

  const comparison = CODE_COMPARISONS.find((c) => c.id === selectedId) || CODE_COMPARISONS[0];

  const handleCopy = (lang: string, code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedLang(lang);
    setTimeout(() => setCopiedLang(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <GitCompare className="w-5 h-5 text-emerald-400" />
              Julia CUDA.jl vs C++ CUDA vs Python Numba
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Direct side-by-side technical comparison showing why Julia provides the speed of C++ CUDA with the expressiveness of high-level scripting.
            </p>
          </div>

          {/* Example selector */}
          <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
            {CODE_COMPARISONS.map((c) => (
              <button
                key={c.id}
                onClick={() => setSelectedId(c.id)}
                className={`text-xs px-3 py-1.5 rounded-lg font-medium transition-all ${
                  selectedId === c.id
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {c.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Side-by-Side 3-column view */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Julia CUDA.jl (Highlight) */}
        <div className="bg-slate-900/90 border-2 border-emerald-500/40 rounded-2xl p-5 space-y-3 relative glow-julia">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-emerald-400"></span>
              <span className="text-sm font-bold text-white font-mono">Julia (CUDA.jl)</span>
            </div>
            <button
              onClick={() => handleCopy("julia", comparison.juliaCode)}
              className="text-xs text-slate-400 hover:text-white bg-slate-800 px-2 py-1 rounded flex items-center gap-1"
            >
              {copiedLang === "julia" ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
              <span>{copiedLang === "julia" ? "Copied" : "Copy"}</span>
            </button>
          </div>
          <pre className="bg-slate-950 p-3.5 rounded-xl text-xs font-mono text-emerald-300/90 overflow-x-auto border border-slate-800/80 leading-relaxed max-h-[460px]">
            <code>{comparison.juliaCode}</code>
          </pre>
          <div className="text-[11px] text-emerald-400 font-medium">
            ✓ JIT compiled directly to PTX | ✓ Zero boilerplate | ✓ Native 1-based indexing
          </div>
        </div>

        {/* C++ CUDA */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-blue-400"></span>
              <span className="text-sm font-bold text-white font-mono">C++ (CUDA Runtime)</span>
            </div>
            <button
              onClick={() => handleCopy("cpp", comparison.cppCode)}
              className="text-xs text-slate-400 hover:text-white bg-slate-800 px-2 py-1 rounded flex items-center gap-1"
            >
              {copiedLang === "cpp" ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
              <span>{copiedLang === "cpp" ? "Copied" : "Copy"}</span>
            </button>
          </div>
          <pre className="bg-slate-950 p-3.5 rounded-xl text-xs font-mono text-blue-300/90 overflow-x-auto border border-slate-800/80 leading-relaxed max-h-[460px]">
            <code>{comparison.cppCode}</code>
          </pre>
          <div className="text-[11px] text-slate-400">
            • Requires offline <code className="text-slate-300">nvcc</code> compiler &amp; header files.
          </div>
        </div>

        {/* Python Numba / CuPy */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-amber-400"></span>
              <span className="text-sm font-bold text-white font-mono">Python (Numba / CuPy)</span>
            </div>
            <button
              onClick={() => handleCopy("python", comparison.pythonCode)}
              className="text-xs text-slate-400 hover:text-white bg-slate-800 px-2 py-1 rounded flex items-center gap-1"
            >
              {copiedLang === "python" ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
              <span>{copiedLang === "python" ? "Copied" : "Copy"}</span>
            </button>
          </div>
          <pre className="bg-slate-950 p-3.5 rounded-xl text-xs font-mono text-amber-300/90 overflow-x-auto border border-slate-800/80 leading-relaxed max-h-[460px]">
            <code>{comparison.pythonCode}</code>
          </pre>
          <div className="text-[11px] text-slate-400">
            • Limited Python subset inside kernels; dynamic dispatch overhead on host.
          </div>
        </div>
      </div>

      {/* Julia Advantages Callout */}
      <div className="bg-slate-950 p-5 rounded-2xl border border-emerald-500/30 space-y-3">
        <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          Why Julia Outperforms in GPU Scientific Computing
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {comparison.advantages.map((adv, idx) => (
            <div key={idx} className="bg-slate-900/70 p-3 rounded-xl border border-slate-800 text-xs text-slate-300">
              {adv}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
