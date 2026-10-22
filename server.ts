import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;

app.use(express.json({ limit: "5mb" }));

// Server-side Gemini AI setup
const apiKey = process.env.GEMINI_API_KEY;
let ai: GoogleGenAI | null = null;
if (apiKey) {
  ai = new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

// System prompt specialized in Julia + CUDA.jl
const SYSTEM_PROMPT = `You are the world's foremost expert in the Julia programming language and the CUDA.jl GPU computing ecosystem.
Your role is to assist developers, scientists, and engineers in writing, optimizing, debugging, and understanding GPU-accelerated Julia programs.

Key CUDA.jl Knowledge & Idioms:
1. Indexing & Dimensions: Julia uses 1-based indexing! In CUDA kernels:
   - 1D linear global index: \`i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x\`
   - 2D grid: \`i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x\`, \`j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y\`
   - Always cast \`Int32(1)\` or use Int32 for index math inside GPU kernels to avoid 64-bit integer conversion overhead.
   - Always guard array boundaries: \`if i <= length(arr) ... end\`.

2. Memory Hierarchy:
   - Global memory: CuArray{T, N}
   - Shared memory: \`@cuStaticSharedMem(Float32, (16, 16))\` or \`@cuDynamicSharedMem(Float32, len)\` with \`shmem=...\` in \`@cuda\`.
   - Synchronization: \`sync_threads()\` and memory fences.
   - Warp primitives: \`shfl_down_sync(0xffffffff, val, delta)\`, \`vote_all_sync\`, \`activemask()\`.

3. Performance & Common Traps:
   - Scalar indexing: \`CUDA.allowscalar(false)\` should ALWAYS be recommended. Scalar indexing brings GPU memory to host byte-by-byte causing catastrophic slowdowns.
   - Broadcast fusion: \`y .= 2.0f0 .* x .+ 1.0f0\` generates a fused PTX kernel automatically without custom kernels!
   - Type Stability: Type instability inside GPU kernels causes fatal compilation errors or dynamic dispatch. Everything must be concretely typed.
   - Precision: Float32 (single precision) is drastically faster than Float64 on GeForce/consumer GPUs (up to 32x-64x faster).
   - Asynchronous execution: GPU kernels launch asynchronously. Call \`CUDA.@sync\` or \`CUDA.synchronize()\` before benchmarking or timing with \`@btime\`.

Always output pristine, production-ready Julia code with syntax highlighting, clear explanations, and performance annotations.`;

// API endpoint for Julia + CUDA.jl AI assistance
app.post("/api/cuda-ai", async (req, res) => {
  try {
    const { prompt, taskType, codeSnippet } = req.body;

    if (!prompt) {
      return res.status(400).json({ error: "Prompt is required" });
    }

    if (!ai) {
      // Fallback response generator if API key is not configured in environment
      return res.json({
        response: generateFallbackResponse(taskType, prompt, codeSnippet),
        fallback: true,
      });
    }

    const fullPrompt = `Task Type: ${taskType || "General Question"}
${codeSnippet ? `User Provided Code / Error Context:\n\`\`\`julia\n${codeSnippet}\n\`\`\`\n` : ""}
User Inquiry: ${prompt}

Please provide an in-depth, idiomatic Julia + CUDA.jl response with well-commented Julia code, memory layout insights, and performance recommendations.`;

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",
      contents: fullPrompt,
      config: {
        systemInstruction: SYSTEM_PROMPT,
        temperature: 0.2,
      },
    });

    const text = response.text || "No response generated";
    return res.json({ response: text, fallback: false });
  } catch (error: any) {
    console.error("Gemini API error:", error);
    // Return fallback guidance so user experience is never blocked
    return res.status(200).json({
      response: `An error occurred contacting Gemini (${error?.message || "connection error"}). Here is standard CUDA.jl guidance:\n\n` +
        generateFallbackResponse(req.body.taskType, req.body.prompt, req.body.codeSnippet),
      fallback: true,
      errorDetails: error?.message,
    });
  }
});

function generateFallbackResponse(taskType?: string, prompt?: string, codeSnippet?: string): string {
  const p = (prompt || "").toLowerCase();
  
  if (p.includes("scalar") || taskType === "debug_scalar") {
    return `### Fixing Scalar Indexing in CUDA.jl

**Problem**: In CUDA.jl, indexing into a \`CuArray\` from the CPU (like \`arr[1]\` or in standard \`for i in 1:length(arr)\` CPU loops) is called **scalar indexing**. This forces a synchronous PCIe memory transfer for every single element, causing massive performance drops.

\`\`\`julia
using CUDA

# 1. Enforce no scalar indexing (Best practice for all CUDA.jl scripts)
CUDA.allowscalar(false)

# 2. INCORRECT (Triggers Scalar Indexing Exception):
# a = CUDA.zeros(Float32, 1000)
# for i in 1:1000
#     a[i] = i * 2.0f0 # SLOW & FAILS with allowscalar(false)
# end

# 3. CORRECT - Solution A: Vectorized Broadcasting (Kernel Fusion)
a = CUDA.zeros(Float32, 1000)
indices = CuArray(1:1000)
a .= indices .* 2.0f0

# 4. CORRECT - Solution B: Custom GPU Kernel (@cuda)
function init_kernel!(a)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(a)
        @inbounds a[i] = Float32(i) * 2.0f0
    end
    return nothing
end

threads = 256
blocks = cld(length(a), threads)
@cuda threads=threads blocks=blocks init_kernel!(a)
CUDA.synchronize()
\`\`\``;
  }

  if (p.includes("kernel") || p.includes("vector") || taskType === "kernel_gen") {
    return `### Idiomatic Custom Kernel with \`@cuda\`

Here is the standard pattern for a custom 1D kernel in CUDA.jl with 1-based indexing:

\`\`\`julia
using CUDA

# Custom element-wise addition kernel
function vmul_add_kernel!(c, a, b, alpha)
    # Calculate global 1-based index (Julia style)
    # Note: Use Int32 to prevent 64-bit address calculation overhead
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    
    # Boundary guard: essential because blocks * threads >= N
    if i <= length(c)
        @inbounds c[i] = a[i] * b[i] + alpha
    end
    
    return nothing  # GPU kernels must return nothing
end

function run_vmul_add(N = 1_000_000)
    a_d = CUDA.rand(Float32, N)
    b_d = CUDA.rand(Float32, N)
    c_d = CUDA.zeros(Float32, N)
    alpha = 4.2f0
    
    # Launch configuration
    threads = 256
    blocks = cld(N, threads)
    
    # Launch kernel on the GPU
    @cuda threads=threads blocks=blocks vmul_add_kernel!(c_d, a_d, b_d, alpha)
    CUDA.synchronize()
    
    println("Kernel executed successfully across \$N elements.")
    return c_d
end
\`\`\`

**Key Points:**
- \`blockIdx().x - Int32(1)\` handles Julia's 1-based indexing.
- Use \`@inbounds\` inside the boundary check for max speed.
- The return statement in any CUDA.jl kernel **must** be \`return nothing\`.`;
  }

  return `### Julia + CUDA.jl Core Architecture & Quick Start

CUDA.jl provides zero-overhead GPU computing in Julia, compiling directly to PTX via LLVM.

\`\`\`julia
using CUDA

# 1. Device inspection
println("CUDA Functional? ", CUDA.functional())
println("Active Device:   ", CUDA.name(CUDA.device()))
CUDA.memory_status()

# 2. High-Level CuArray operations (Fused broadcasting)
N = 10_000_000
x = CUDA.rand(Float32, N)
y = CUDA.rand(Float32, N)

# Fuses into a single GPU kernel automatically!
CUDA.@sync y .= 3.0f0 .* x .^ 2 .+ y

# 3. Parallel Reduction
total = sum(y) # Uses optimized GPU reduction
println("Sum: ", total)
\`\`\`

Feel free to ask specific questions about shared memory tiling, warp shuffles, CUBLAS integration, or debugging illegal memory access!`;
}

// Dev server Vite middleware or production static files
async function startServer() {
  const isProd = process.env.NODE_ENV === "production";

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Julia CUDA.jl Studio server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
