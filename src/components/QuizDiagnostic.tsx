import React, { useState } from "react";
import { Award, CheckCircle, XCircle, RotateCcw, AlertCircle, HelpCircle } from "lucide-react";

interface QuizQuestion {
  id: number;
  question: string;
  codeSnippet?: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
}

const QUESTIONS: QuizQuestion[] = [
  {
    id: 1,
    question: "What is the correct Julia 1-based indexing idiom for a 1D CUDA kernel?",
    options: [
      "i = blockIdx().x * blockDim().x + threadIdx().x",
      "i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x",
      "i = (blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1)",
      "i = threadIdx().x"
    ],
    correctAnswer: 1,
    explanation: "Julia arrays are 1-indexed! blockIdx().x starts at 1, so we subtract 1 before multiplying by blockDim().x, and add threadIdx().x (which also starts at 1), producing a valid 1-based index from 1 to N."
  },
  {
    id: 2,
    question: "What does calling `CUDA.allowscalar(false)` achieve, and why is it recommended for all Julia GPU scripts?",
    options: [
      "It disables scalar indexing from the host CPU, throwing an immediate error rather than causing disastrous PCIe round-trip stalls.",
      "It restricts Julia from using Float32 scalars.",
      "It forces the GPU to only execute SIMD instructions.",
      "It prevents CUDA from allocating arrays larger than 4GB."
    ],
    correctAnswer: 0,
    explanation: "Accessing CuArray elements from CPU code (e.g. arr[i] in a CPU loop) generates an individual PCIe transfer for each element. allowscalar(false) guarantees that accidental scalar fallbacks fail loudly rather than silently running 1000x slower."
  },
  {
    id: 3,
    question: "What must every custom Julia GPU kernel launched via `@cuda` return?",
    options: [
      "return 0",
      "return true",
      "return nothing",
      "return arr"
    ],
    correctAnswer: 2,
    explanation: "In CUDA.jl, GPU kernels execute asynchronously on the device and cannot return a value to the host CPU. Any custom kernel must explicitly terminate with `return nothing`."
  },
  {
    id: 4,
    question: "Why does evaluating `y .= 2.0f0 .* x .+ 1.0f0` on a CuArray outperform traditional separate operations in other languages?",
    options: [
      "Julia's compiler automatically fuses the entire chained expression into a single PTX GPU kernel without temporary array allocations.",
      "CuArray runs on Tensor Cores exclusively.",
      "Julia bypasses the GPU driver directly.",
      "It converts the numbers to 8-bit integers."
    ],
    correctAnswer: 0,
    explanation: "Julia's broadcast fusion fuses all '.' operations into a single GPU kernel loop. This minimizes GPU global memory round-trips from multiple kernel passes to just one pass!"
  },
  {
    id: 5,
    question: "On consumer NVIDIA GPUs (e.g., RTX 4090), why should you prefer `Float32` (or `1.0f0`) over `Float64` (`1.0`)?",
    options: [
      "Float64 is not supported on NVIDIA hardware.",
      "Consumer GPUs have a 1:64 FP64 to FP32 compute throughput ratio, making Float64 up to 64x slower.",
      "Julia cannot represent Float64 on GPU.",
      "Float32 saves GPU power but runs at the same compute speed."
    ],
    correctAnswer: 1,
    explanation: "GeForce GPUs dedicate their silicon area to FP32 and Tensor cores, with FP64 capped at 1/64th speed. Unless you are running on datacenter GPUs like the A100 or H100 (which have 1:2 FP64), single-precision Float32 is dramatically faster."
  }
];

export const QuizDiagnostic: React.FC = () => {
  const [selectedAnswers, setSelectedAnswers] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState<boolean>(false);

  const handleSelect = (questionId: number, optionIdx: number) => {
    if (submitted) return;
    setSelectedAnswers((prev) => ({ ...prev, [questionId]: optionIdx }));
  };

  const handleReset = () => {
    setSelectedAnswers({});
    setSubmitted(false);
  };

  const score = QUESTIONS.reduce((acc, q) => {
    return selectedAnswers[q.id] === q.correctAnswer ? acc + 1 : acc;
  }, 0);

  return (
    <div className="space-y-6">
      {/* Intro */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Award className="w-5 h-5 text-emerald-400" />
              Julia &amp; CUDA.jl Architectural Mastery Check
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Verify your knowledge of CUDA.jl indexing conventions, memory bandwidth principles, and common pitfalls.
            </p>
          </div>

          {submitted && (
            <div className="flex items-center gap-3 bg-slate-950 px-4 py-2 rounded-xl border border-slate-800 font-mono text-sm">
              <span className="text-slate-400">Score:</span>
              <span className={`font-bold ${score === QUESTIONS.length ? "text-emerald-400" : "text-amber-400"}`}>
                {score} / {QUESTIONS.length} ({Math.round((score / QUESTIONS.length) * 100)}%)
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Questions */}
      <div className="space-y-5">
        {QUESTIONS.map((q) => {
          const userAnswer = selectedAnswers[q.id];
          const isCorrect = userAnswer === q.correctAnswer;

          return (
            <div
              key={q.id}
              className={`bg-slate-900/80 border rounded-2xl p-5 space-y-3 transition-all ${
                submitted
                  ? isCorrect
                    ? "border-emerald-500/40 bg-emerald-950/10"
                    : "border-rose-500/40 bg-rose-950/10"
                  : "border-slate-800"
              }`}
            >
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-300 font-mono text-xs flex items-center justify-center shrink-0 mt-0.5">
                  {q.id}
                </span>
                <div className="space-y-1">
                  <h3 className="text-sm font-semibold text-white">{q.question}</h3>
                  {q.codeSnippet && (
                    <pre className="bg-slate-950 p-2.5 rounded-lg text-xs font-mono text-emerald-300 border border-slate-800">
                      <code>{q.codeSnippet}</code>
                    </pre>
                  )}
                </div>
              </div>

              {/* Options */}
              <div className="space-y-2 pl-9">
                {q.options.map((opt, optIdx) => {
                  const isSelected = userAnswer === optIdx;
                  let optionClass = "bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700";

                  if (submitted) {
                    if (optIdx === q.correctAnswer) {
                      optionClass = "bg-emerald-500/20 border-emerald-500/50 text-emerald-200 font-medium";
                    } else if (isSelected && !isCorrect) {
                      optionClass = "bg-rose-500/20 border-rose-500/50 text-rose-200";
                    } else {
                      optionClass = "bg-slate-950/30 border-slate-900 text-slate-600";
                    }
                  } else if (isSelected) {
                    optionClass = "bg-emerald-500/20 border-emerald-500/50 text-emerald-300";
                  }

                  return (
                    <button
                      key={optIdx}
                      onClick={() => handleSelect(q.id, optIdx)}
                      className={`w-full text-left p-3 rounded-xl border text-xs transition-all flex items-center justify-between ${optionClass}`}
                    >
                      <span>{opt}</span>
                      {submitted && optIdx === q.correctAnswer && (
                        <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 ml-2" />
                      )}
                      {submitted && isSelected && !isCorrect && (
                        <XCircle className="w-4 h-4 text-rose-400 shrink-0 ml-2" />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Explanation after submit */}
              {submitted && (
                <div className="pl-9 pt-2 text-xs text-slate-400 bg-slate-950/40 p-3 rounded-xl border border-slate-800/60">
                  <span className="font-semibold text-slate-300 block mb-0.5">Explanation:</span>
                  {q.explanation}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Action Footer */}
      <div className="flex justify-end gap-3 pt-2">
        {submitted ? (
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-semibold transition-all"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Retake Check</span>
          </button>
        ) : (
          <button
            onClick={() => setSubmitted(true)}
            disabled={Object.keys(selectedAnswers).length < QUESTIONS.length}
            className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold transition-all shadow-md cursor-pointer"
          >
            Submit Answers
          </button>
        )}
      </div>
    </div>
  );
};
