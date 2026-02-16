//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#include "MatrixKernels.cuh"
#include <cuda_runtime.h>

__global__ void hadamardKernel(const double *A, const double *B, double *C,
                               int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    C[idx] = A[idx] * B[idx];
  }
}

__global__ void
hadamard_scale_kernel(const double *__restrict__ ab, // size: numStates * P
                      const double *__restrict__ cd, // size: numStates * P
                      double *__restrict__ r,        // size: numStates * P
                      uint8_t *__restrict__ scale_count, int numStates, int P,
                      double scaling_threshold, int scaling_exp) {
  int j = blockIdx.x; // one block per column (site)
  if (j >= P)
    return;

  // 1) compute r = ab * cd and local max
  double local_max = 0.0;

  for (int i = threadIdx.x; i < numStates; i += blockDim.x) {
    double v = ab[j * numStates + i] * cd[j * numStates + i];
    r[j * numStates + i] = v;
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
    for (int i = threadIdx.x; i < numStates; i += blockDim.x) {
      r[j * numStates + i] = scalbn(r[j * numStates + i], scaling_exp);
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
 * For DNA (K=4): 32 sites per block with 128 threads
 * For Protein (K=20): 6 sites per block with 120 threads
 */
__global__ void composite_hadamard_fused_kernel(
    const double *__restrict__ a, // KxK, column-major: a[k*K + i] = A(i,k)
    const double
        *__restrict__ b, // KxP, site-major columns: b[j*K + k] = B(k,j)
    const double *__restrict__ c,      // KxK, column-major
    const double *__restrict__ d,      // KxP, site-major columns
    double *__restrict__ r,            // KxP, site-major columns: r[j*K + i]
    uint8_t *__restrict__ scale_count, // Per-site scale counter
    int K, int P, int sites_per_block, double scaling_threshold,
    int scaling_exp) {

  // Shared memory layout: [A matrix (K*K)] [C matrix (K*K)] [reduction
  // (blockDim.x)]
  extern __shared__ double smem[];
  double *s_a = smem;                  // K*K elements
  double *s_c = smem + K * K;          // K*K elements
  double *s_reduce = smem + 2 * K * K; // blockDim.x elements

  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Cooperatively load A and C into shared memory (once per block)
  for (int idx = tid; idx < K * K; idx += threads_per_block) {
    s_a[idx] = a[idx];
    s_c[idx] = c[idx];
  }
  __syncthreads();

  // Each thread handles one (site, state) pair
  int local_site = tid / K;
  int state_i = tid % K;
  int global_site_j = blockIdx.x * sites_per_block + local_site;

  // --- Loop 1: Compute Hadamard product ---
  // Inactive threads keep val = 0 so they can safely participate in reduction
  double val = 0.0;
  bool active = (global_site_j < P && local_site < sites_per_block);

  if (active) {
    const double *bj = &b[global_site_j * K];
    const double *dj = &d[global_site_j * K];

    double s1 = 0.0;
    double s2 = 0.0;
    for (int k = 0; k < K; ++k) {
      s1 += s_a[k * K + state_i] * bj[k];
      s2 += s_c[k * K + state_i] * dj[k];
    }
    val = s1 * s2;
  }

  // --- Loop 2: Per-site max-abs reduction + scaling ---
  // (matches OpenACC: find max per column, rescale if below threshold)
  s_reduce[tid] = fabs(val);
  __syncthreads();

  // Reduce within each group of K threads (one group per site)
  int site_base = local_site * K;
  for (int stride = K / 2; stride > 0; stride >>= 1) {
    if (state_i < stride) {
      s_reduce[site_base + state_i] =
          fmax(s_reduce[site_base + state_i],
               s_reduce[site_base + state_i + stride]);
    }
    __syncthreads();
  }
  // Handle odd K (e.g. K=5): fold in the last element
  if ((K & 1) && state_i == 0 && K > 1) {
    s_reduce[site_base] =
        fmax(s_reduce[site_base], s_reduce[site_base + K - 1]);
  }
  __syncthreads();

  double max_abs = s_reduce[site_base];

  // Rescale and write
  if (active) {
    if (max_abs < scaling_threshold && max_abs > 0.0) {
      val = scalbn(val, scaling_exp);
      if (state_i == 0)
        scale_count[global_site_j] += 1;
    }
    r[global_site_j * K + state_i] = val;
  }
}

/**
 * Scaling kernel - runs AFTER the fused kernel to match OpenACC pattern
 * One block per site (column), finds max and rescales if needed
 */
__global__ void
scaling_kernel(double *__restrict__ r,            // KxP, site-major columns
               uint8_t *__restrict__ scale_count, // Per-site scale counter
               int K, int P, double scaling_threshold, int scaling_exp) {
  int j = blockIdx.x; // one block per column (site)
  if (j >= P)
    return;

  //  Find max absolute value in this column
  double max_abs = 0.0;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    double v = fabs(r[j * K + i]);
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
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
      r[j * K + i] = scalbn(r[j * K + i], scaling_exp);
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
      d_AB, d_CD, d_R, d_scale_count, numStates, P, scaling_threshold,
      scaling_exp);
}

void launchCompositeHadamardFused(const double *d_A, const double *d_B,
                                  const double *d_C, const double *d_D,
                                  double *d_R, uint8_t *d_scale_count, int K,
                                  int P, double scaling_threshold,
                                  int scaling_exp, cudaStream_t stream) {
  // 128 threads per block
  // DNA (K=4):    128 threads → 32 sites per block
  // Protein (K=20): ~6 sites per block
  int target_threads = 128;
  int sites_per_block = target_threads / K;
  if (sites_per_block < 1)
    sites_per_block = 1;

  int threads_per_block = sites_per_block * K;
  int num_blocks = (P + sites_per_block - 1) / sites_per_block;

  // Shared memory: 2 matrices (K*K each) + reduction workspace
  // (threads_per_block)
  size_t smem_size = (2 * K * K + threads_per_block) * sizeof(double);

  composite_hadamard_fused_kernel<<<num_blocks, threads_per_block, smem_size,
                                    stream>>>(
      d_A, d_B, d_C, d_D, d_R, d_scale_count, K, P, sites_per_block,
      scaling_threshold, scaling_exp);
}
