//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#include "MatrixKernels.cuh"
#include <cuda_runtime.h>

// Constant memory for values that are fixed for the entire execution
__constant__ int d_K; // number of states (4 for DNA, 20 for protein)
__constant__ int d_P; // number of sites/patterns

void setKernelConstants(int K, int P) {
  cudaMemcpyToSymbol(d_K, &K, sizeof(int));
  cudaMemcpyToSymbol(d_P, &P, sizeof(int));
}

__global__ void hadamardKernel(const double *A, const double *B, double *C,
                               int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] * B[idx];
  }
}

__global__ void
hadamard_scale_kernel(const double *__restrict__ ab, // size: d_K * d_P
                      const double *__restrict__ cd, // size: d_K * d_P
                      double *__restrict__ r,        // size: d_K * d_P
                      uint8_t *__restrict__ scale_count,
                      double scaling_threshold, int scaling_exp) {
  int j = blockIdx.x; // one block per column (site)
  if (j >= d_P)
    return;

  // 1) compute r = ab * cd and local max
  double local_max = 0.0;

  for (int i = threadIdx.x; i < d_K; i += blockDim.x) {
    double v = ab[j * d_K + i] * cd[j * d_K + i];
    r[j * d_K + i] = v;
    double av = fabs(v);
    if (av > local_max)
      local_max = av;
  }

  // 2) reduce max within block
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  sdata[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }

  double max_abs = sdata[0];

  // 3) rescale if below threshold
  if (max_abs < scaling_threshold) {
    for (int i = threadIdx.x; i < d_K; i += blockDim.x) {
      r[j * d_K + i] = scalbn(r[j * d_K + i], scaling_exp);
    }
    if (tid == 0)
      scale_count[j] += 1;
  }
}

/**
 * Fused Composite Hadamard + Scaling Kernel
 * Matches OpenACC two-loop pattern:
 *   Loop 1: compute R = (A*B) ⊙ (C*D)
 *   Loop 2: per-site max reduction + conditional scaling
 *
 * Two reduction strategies based on K vs warp size:
 *   - K divides 32 (e.g. K=4): warp-shuffle reduction (no __syncthreads)
 *   - K doesn't divide 32 (e.g. K=20): shared memory reduction
 *
 * For DNA (K=4):     256 threads → 64 sites per block (warp-shuffle path)
 * For Protein (K=20): 240 threads → 12 sites per block (shared-mem path)
 */
__global__ void composite_hadamard_fused_kernel(
    const double *__restrict__ a,      // d_K x d_K, column-major
    const double *__restrict__ b,      // d_K x d_P, site-major columns
    const double *__restrict__ c,      // d_K x d_K, column-major
    const double *__restrict__ d,      // d_K x d_P, site-major columns
    double *__restrict__ r,            // d_K x d_P, site-major columns
    uint8_t *__restrict__ scale_count, // Per-site scale counter
    int sites_per_block, double scaling_threshold, int scaling_exp) {

  // Shared memory layout:
  //   [A matrix: d_K*d_K] [C matrix: d_K*d_K] [reduction: blockDim.x (only if
  //   K%32!=0)]
  extern __shared__ double smem[];
  double *s_a = smem;             // d_K*d_K elements
  double *s_c = smem + d_K * d_K; // d_K*d_K elements

  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Cooperatively load A and C into shared memory (once per block)
  for (int idx = tid; idx < d_K * d_K; idx += threads_per_block) {
    s_a[idx] = a[idx];
    s_c[idx] = c[idx];
  }
  __syncthreads();

  // Each thread handles one (site, state) pair
  int local_site = tid / d_K;
  int state_i = tid % d_K;
  int global_site_j = blockIdx.x * sites_per_block + local_site;

  // --- Loop 1: Compute Hadamard product ---
  double val = 0.0;
  bool active = (global_site_j < d_P && local_site < sites_per_block);

  if (active) {
    const double *bj = &b[global_site_j * d_K];
    const double *dj = &d[global_site_j * d_K];

    double s1 = 0.0;
    double s2 = 0.0;
    for (int k = 0; k < d_K; ++k) {
      s1 += s_a[k * d_K + state_i] * bj[k];
      s2 += s_c[k * d_K + state_i] * dj[k];
    }
    val = s1 * s2;
  }

  // --- Loop 2: Per-site max-abs reduction + scaling ---
  double max_abs;

  if ((32 % d_K) == 0) {
    // FAST PATH: K divides 32 (e.g. K=4, K=8, K=16)
    // Site groups never straddle warp boundaries → use warp shuffles.
    // K=4: 8 sites per warp, K=8: 4 sites per warp, K=16: 2 sites per warp.
    int lane = threadIdx.x % 32;
    int group_start = (lane / d_K) * d_K;
    unsigned group_mask = (((unsigned)1 << d_K) - 1) << group_start;

    double max_val = fabs(val);
    for (int stride = 16; stride >= 1; stride >>= 1) {
      if (stride < d_K) {
        double other = __shfl_down_sync(group_mask, max_val, stride);
        max_val = fmax(max_val, other);
      }
    }
    // Broadcast from group leader (state_i==0)
    max_abs = __shfl_sync(group_mask, max_val, group_start);
  } else {
    // SAFE PATH: K doesn't divide 32 (e.g. K=20)
    // Site groups can straddle warp boundaries → use shared memory reduction.
    double *s_reduce = smem + 2 * d_K * d_K;

    s_reduce[tid] = fabs(val);
    __syncthreads();

    int site_base = local_site * d_K;
    for (int stride = d_K / 2; stride > 0; stride >>= 1) {
      if (state_i < stride) {
        s_reduce[site_base + state_i] =
            fmax(s_reduce[site_base + state_i],
                 s_reduce[site_base + state_i + stride]);
      }
      __syncthreads();
    }
    // Handle odd K: fold in the last element
    if ((d_K & 1) && state_i == 0 && d_K > 1) {
      s_reduce[site_base] =
          fmax(s_reduce[site_base], s_reduce[site_base + d_K - 1]);
    }
    __syncthreads();

    max_abs = s_reduce[site_base];
  }

  // Rescale and write
  if (active) {
    if (max_abs < scaling_threshold && max_abs > 0.0) {
      val = scalbn(val, scaling_exp);
      if (state_i == 0)
        scale_count[global_site_j] += 1;
    }
    r[global_site_j * d_K + state_i] = val;
  }
}

/**
 * Scaling kernel - runs AFTER the fused kernel to match OpenACC pattern
 * One block per site (column), finds max and rescales if needed
 */
__global__ void
scaling_kernel(double *__restrict__ r, // d_K x d_P, site-major columns
               uint8_t *__restrict__ scale_count, // Per-site scale counter
               double scaling_threshold, int scaling_exp) {
  int j = blockIdx.x; // one block per column (site)
  if (j >= d_P)
    return;

  //  Find max absolute value in this column
  double max_abs = 0.0;
  for (int i = threadIdx.x; i < d_K; i += blockDim.x) {
    double v = fabs(r[j * d_K + i]);
    if (v > max_abs)
      max_abs = v;
  }

  //  Reduce max within block
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  sdata[tid] = max_abs;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }

  max_abs = sdata[0];

  // scale if below threshold
  if (max_abs < scaling_threshold && max_abs > 0.0) {
    for (int i = threadIdx.x; i < d_K; i += blockDim.x) {
      r[j * d_K + i] = scalbn(r[j * d_K + i], scaling_exp);
    }
    if (tid == 0)
      scale_count[j] += 1;
  }
}

void launchHadamard(const double *A, const double *B, double *C, int size,
                    int blockSize) {
  int gridSize = (size + blockSize - 1) / blockSize;
  hadamardKernel<<<gridSize, blockSize>>>(A, B, C, size);
}

void launchCompositeHadamard(const double *d_AB, const double *d_CD,
                             double *d_R, uint8_t *d_scale_count, int numStates,
                             int P, int blockSize, double scaling_threshold,
                             int scaling_exp) {
  int gridSize = P;
  size_t sharedBytes = blockSize * sizeof(double);

  hadamard_scale_kernel<<<gridSize, blockSize, sharedBytes>>>(
      d_AB, d_CD, d_R, d_scale_count, scaling_threshold, scaling_exp);
}

void launchCompositeHadamardFused(const double *d_A, const double *d_B,
                                  const double *d_C, const double *d_D,
                                  double *d_R, uint8_t *d_scale_count, int K,
                                  int P, double scaling_threshold,
                                  int scaling_exp, cudaStream_t stream) {
  // 256 target threads per block for better occupancy
  // DNA (K=4):     256 threads → 64 sites per block  (warp-shuffle reduction)
  // Protein (K=20): 240 threads → 12 sites per block (shared-mem reduction)
  int target_threads = 256;
  int sites_per_block = target_threads / K;
  if (sites_per_block < 1)
    sites_per_block = 1;

  int threads_per_block = sites_per_block * K;
  int num_blocks = (P + sites_per_block - 1) / sites_per_block;

  // Shared memory: 2 matrices (K*K each)
  // + reduction workspace (threads_per_block) only if K doesn't divide 32
  size_t smem_size = (2 * K * K) * sizeof(double);
  if ((32 % K) != 0) {
    smem_size += threads_per_block * sizeof(double);
  }

  composite_hadamard_fused_kernel<<<num_blocks, threads_per_block, smem_size,
                                    stream>>>(d_A, d_B, d_C, d_D, d_R,
                                              d_scale_count, sites_per_block,
                                              scaling_threshold, scaling_exp);
}

// ========= Fused Log-Likelihood Kernel =========
// Replaces cuBLAS DGEMM + separate reduction with a single kernel.
// Uses grid-stride loop so each thread processes multiple sites,
// increasing arithmetic intensity per thread.
//   1. Dot product: siteL = sum_k baseFreq[k] * rootL[k + j*K]
//   2. val += freq[j] * (log(siteL) + scale_count[j] * log_scaling_threshold)
//   3. Block-level warp-shuffle reduction -> atomicAdd to global result
__global__ void fused_log_likelihood_kernel(
    const double *__restrict__ baseFreq, // K values (cached on GPU)
    const double *__restrict__ rootL,    // K*P column-major (already on GPU)
    const uint8_t *__restrict__ scale_count, // P-length (GPU-resident)
    const int *__restrict__ freq,            // P-length pattern frequencies
    double *__restrict__ d_result,           // single output scalar
    int K, int P, double log_scaling_threshold) {

  extern __shared__ double sdata[];

  int tid = threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  // Grid-stride loop: each thread accumulates over multiple sites
  double val = 0.0;
  for (int gid = blockIdx.x * blockDim.x + tid; gid < P; gid += total_threads) {
    // Dot product replaces cuBLAS DGEMM for M=1
    // For DNA (K=4): just 4 MADs, compiler unrolls this
    double siteL = 0.0;
    const double *col = rootL + gid * K; // column-major: column j starts at j*K
    for (int k = 0; k < K; ++k) {
      siteL += baseFreq[k] * col[k];
    }
    val += freq[gid] * (log(siteL) + scale_count[gid] * log_scaling_threshold);
  }

  sdata[tid] = val;
  __syncthreads();

  // Block-level tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(d_result, sdata[0]);
  }
}

void launchFusedLogLikelihood(const double *d_baseFreq, const double *d_rootL,
                              const uint8_t *d_scale_count, const int *d_freq,
                              double *d_result, int K, int P,
                              double log_scaling_threshold,
                              cudaStream_t stream) {
  cudaMemsetAsync(d_result, 0, sizeof(double), stream);

  // 128 threads with capped grid: grid-stride loop handles overflow
  int threads = 128;
  int blocks = (P + threads - 1) / threads;
  // Cap grid size to avoid excessive blocks for very large P
  if (blocks > 256)
    blocks = 256;
  size_t smem = threads * sizeof(double);

  fused_log_likelihood_kernel<<<blocks, threads, smem, stream>>>(
      d_baseFreq, d_rootL, d_scale_count, d_freq, d_result, K, P,
      log_scaling_threshold);
}
