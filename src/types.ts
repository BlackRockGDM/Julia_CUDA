export interface GpuSpec {
  id: string;
  name: string;
  architecture: string;
  computeCapability: string;
  maxThreadsPerBlock: number;
  maxWarpsPerSm: number;
  maxThreadsPerSm: number;
  maxBlocksPerSm: number;
  maxRegistersPerSm: number;
  maxRegistersPerThread: number;
  maxSharedMemoryPerSmKb: number;
  maxSharedMemoryPerBlockKb: number;
  memoryBandwidthGBs: number;
  vramMiB: number;
  fp32TFlops: number;
  fp64TFlops: number;
  notes: string;
}

export interface OccupancyResult {
  threadsPerBlock: number;
  totalBlocks: number;
  activeWarpsPerSm: number;
  theoreticalOccupancy: number; // percentage
  limitingFactor: "Warps" | "Registers" | "Shared Memory" | "Blocks" | "Optimal";
  activeBlocksPerSm: number;
  sharedMemPerBlock: number;
  registersPerBlock: number;
  warpAllocation: number;
  warnings: string[];
}

export interface CodeRecipe {
  id: string;
  title: string;
  category: "Quickstart" | "CuArray Basics" | "Custom Kernels" | "Shared Memory" | "Atomics & Warps" | "Libraries" | "Profiling & Traps";
  summary: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  code: string;
  explanation: string[];
  outputExample?: string;
  tags: string[];
}

export interface SimulationStep {
  stepIndex: number;
  blockIdx: { x: number; y?: number };
  threadIdx: { x: number; y?: number };
  globalIdx: number;
  formula: string;
  explanation: string;
  shmemState?: Record<string, any>;
  activeWarp: number;
}
