export interface CudaToolkitSpec {
  version: string;
  ptxIsaVersion: string;
  minDriverVersion: string;
  features: string[];
  juliaArtifactVersion: string;
  isLatest?: boolean;
  notes: string;
}

export const CUDA_TOOLKITS: CudaToolkitSpec[] = [
  {
    version: "13.1.1",
    ptxIsaVersion: "9.1",
    minDriverVersion: ">= 580.45",
    features: [
      "Native Blackwell B200 / GB200 architecture support (sm_100, sm_103)",
      "PTX ISA 9.1 with enhanced Tensor Memory Accelerator (TMA) primitives",
      "Dynamic Distributed Shared Memory & Asynchronous Cluster Barriers",
      "Next-Gen LLVM 19/20 code generation pipeline in Julia CUDA.jl",
      "Sub-microsecond kernel launch latency reduction"
    ],
    juliaArtifactVersion: "CUDA_Runtime_jll v13.1.1",
    isLatest: true,
    notes: "Latest flagship toolkit release. Full support for Blackwell and enhanced Hopper FP8/FP4 tensor operations."
  },
  {
    version: "12.8",
    ptxIsaVersion: "8.6",
    minDriverVersion: ">= 560.35",
    features: [
      "Blackwell early preview (sm_100a)",
      "Enhanced Hopper DPX instructions for dynamic programming",
      "Improved CUBLASLt tensor layout transformations"
    ],
    juliaArtifactVersion: "CUDA_Runtime_jll v12.8.0",
    notes: "High stability production release for Ampere, Ada, and Hopper."
  },
  {
    version: "12.6",
    ptxIsaVersion: "8.5",
    minDriverVersion: ">= 550.54",
    features: [
      "Full Ada Lovelace (RTX 4090 / RTX 4080) and Ampere (RTX 3050 / RTX 3080) support",
      "CUDA Driver 550.x ABI compatibility",
      "CUDA.jl v5.4+ default artifact"
    ],
    juliaArtifactVersion: "CUDA_Runtime_jll v12.6.2",
    notes: "Widely deployed enterprise and cloud distribution."
  },
  {
    version: "11.8",
    ptxIsaVersion: "7.8",
    minDriverVersion: ">= 520.61",
    features: [
      "Broad compatibility across older Linux clusters and Kepler/Turing systems",
      "CUSOLVER legacy interfaces"
    ],
    juliaArtifactVersion: "CUDA_Runtime_jll v11.8.0",
    notes: "Legacy compatibility version for older HPC nodes."
  }
];
