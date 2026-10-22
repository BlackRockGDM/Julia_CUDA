import React, { useState } from "react";
import { COOKBOOK_RECIPES } from "../data/cookbookData";
import { CodeRecipe } from "../types";
import { Search, Copy, Check, Sparkles, Filter, Code, Terminal, AlertCircle } from "lucide-react";

interface CookbookViewerProps {
  onSelectForAi: (prompt: string, code: string) => void;
}

export const CookbookViewer: React.FC<CookbookViewerProps> = ({ onSelectForAi }) => {
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const categories = [
    "All",
    "Quickstart",
    "CuArray Basics",
    "Custom Kernels",
    "Shared Memory",
    "Atomics & Warps",
    "Libraries",
    "Profiling & Traps",
  ];

  const filteredRecipes = COOKBOOK_RECIPES.filter((recipe) => {
    const matchesCategory = selectedCategory === "All" || recipe.category === selectedCategory;
    const term = searchTerm.toLowerCase();
    const matchesSearch =
      recipe.title.toLowerCase().includes(term) ||
      recipe.summary.toLowerCase().includes(term) ||
      recipe.tags.some((tag) => tag.toLowerCase().includes(term)) ||
      recipe.code.toLowerCase().includes(term);

    return matchesCategory && matchesSearch;
  });

  const handleCopy = (id: string, code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Search and Filters */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4">
        <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Code className="w-5 h-5 text-emerald-400" />
              Julia &amp; CUDA.jl Production Recipe Book
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Curated, verified code patterns for high-performance GPU development in Julia.
            </p>
          </div>

          {/* Search Input */}
          <div className="relative w-full sm:w-72">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search recipes, keywords, tags..."
              className="w-full pl-9 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500/50"
            />
          </div>
        </div>

        {/* Categories Bar */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`text-xs px-3 py-1.5 rounded-lg whitespace-nowrap font-medium transition-all ${
                selectedCategory === cat
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Recipes Grid */}
      <div className="space-y-6">
        {filteredRecipes.length === 0 ? (
          <div className="text-center py-16 bg-slate-900/40 border border-slate-800 rounded-2xl">
            <AlertCircle className="w-8 h-8 text-slate-500 mx-auto mb-2" />
            <p className="text-sm text-slate-400">No recipes matched your search criteria.</p>
          </div>
        ) : (
          filteredRecipes.map((recipe) => {
            const isCopied = copiedId === recipe.id;
            return (
              <div
                key={recipe.id}
                className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 hover:border-slate-700 transition-all shadow-sm"
              >
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                        {recipe.category}
                      </span>
                      <span
                        className={`text-xs font-semibold px-2 py-0.5 rounded-full border ${
                          recipe.difficulty === "Beginner"
                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                            : recipe.difficulty === "Intermediate"
                            ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                            : "bg-purple-500/10 text-purple-400 border-purple-500/30"
                        }`}
                      >
                        {recipe.difficulty}
                      </span>
                    </div>
                    <h3 className="text-base font-bold text-white mt-1.5">{recipe.title}</h3>
                    <p className="text-xs text-slate-400 mt-0.5">{recipe.summary}</p>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-2 self-start sm:self-center">
                    <button
                      onClick={() =>
                        onSelectForAi(
                          `Can you help me adapt and extend this Julia CUDA.jl recipe for '${recipe.title}'?`,
                          recipe.code
                        )
                      }
                      className="flex items-center gap-1.5 text-xs text-purple-300 hover:text-white bg-purple-950/60 hover:bg-purple-900/60 border border-purple-800/80 px-3 py-1.5 rounded-lg transition-colors"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                      <span>Ask AI to Adapt</span>
                    </button>
                    <button
                      onClick={() => handleCopy(recipe.id, recipe.code)}
                      className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg transition-colors"
                    >
                      {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{isCopied ? "Copied" : "Copy"}</span>
                    </button>
                  </div>
                </div>

                {/* Code Block */}
                <div className="relative">
                  <pre className="bg-slate-950 p-4 rounded-xl text-xs font-mono text-emerald-300/90 overflow-x-auto border border-slate-800/80 leading-relaxed max-h-[380px]">
                    <code>{recipe.code}</code>
                  </pre>
                </div>

                {/* Key Takeaways */}
                <div className="bg-slate-950/50 p-3.5 rounded-xl border border-slate-800/60 space-y-1.5">
                  <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
                    Key Performance Takeaways:
                  </div>
                  <ul className="list-disc list-inside space-y-1 text-xs text-slate-300">
                    {recipe.explanation.map((item, i) => (
                      <li key={i} className="leading-relaxed">
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Tags */}
                <div className="flex items-center gap-1.5 flex-wrap pt-1">
                  {recipe.tags.map((tag) => (
                    <span
                      key={tag}
                      className="text-[10px] font-mono bg-slate-950 px-2 py-0.5 rounded text-slate-400 border border-slate-800"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
