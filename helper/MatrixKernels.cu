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
 * OPTIMIZED Fused Composite Hadamard Kernel
 *
 * Computes R = (A*B) ⊙ (C*D) in a single kernel with:
 * - Shared memory caching for transition matrices A and C
 * - Multiple sites processed per block for better occupancy
 * - Coalesced memory access patterns
 *
 * For DNA (K=4): 32 sites per block with 128 threads
 * For Protein (K=20): 6 sites per block with 120 threads
 */
__global__ void composite_hadamard_fused_kernel(
    const double *__restrict__ a, // KxK, column-major: a[k*K + i] = A(i,k)
    const double
        *__restrict__ b, // KxP, site-major columns: b[j*K + k] = B(k,j)
    const double *__restrict__ c, // KxK, column-major
    const double *__restrict__ d, // KxP, site-major columns
    double *__restrict__ r,       // KxP, site-major columns: r[j*K + i]
    int K, int P, int sites_per_block) {
  // Shared memory layout: [A matrix][C matrix]
  extern __shared__ double smem[];
  double *s_a = smem;         // K*K elements for transition matrix A
  double *s_c = smem + K * K; // K*K elements for transition matrix C

  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;

  // Cooperatively load A and C into shared memory (once per block)
  // All threads participate in loading for coalesced access
  for (int idx = tid; idx < K * K; idx += threads_per_block) {
    s_a[idx] = a[idx];
    s_c[idx] = c[idx];
  }
  __syncthreads();

  // Each thread handles one (site, state) pair
  int local_site = tid / K; // Which site within this block
  int state_i = tid % K;    // Which state (row of output)
  int global_site_j = blockIdx.x * sites_per_block + local_site;

  // Bounds check
  if (global_site_j >= P || local_site >= sites_per_block)
    return;

  // Pointers to columns of B and D for this site
  const double *bj = &b[global_site_j * K];
  const double *dj = &d[global_site_j * K];

  // Compute dot products using cached transition matrices
  double s1 = 0.0;
  double s2 = 0.0;

#pragma unroll
  for (int k = 0; k < K; ++k) {
    s1 += s_a[k * K + state_i] * bj[k]; // (A*B)[i,j]
    s2 += s_c[k * K + state_i] * dj[k]; // (C*D)[i,j]
  }

  // Write result: Hadamard product
  r[global_site_j * K + state_i] = s1 * s2;
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
                                  double *d_R, int K, int P,
                                  cudaStream_t stream) {
  // Target ~128 threads per block for good occupancy
  // sites_per_block = floor(128 / K)
  int target_threads = 128;
  int sites_per_block = target_threads / K;
  if (sites_per_block < 1)
    sites_per_block = 1; // At least 1 site per block

  int threads_per_block = sites_per_block * K;
  int num_blocks = (P + sites_per_block - 1) / sites_per_block;

  // Shared memory: 2 matrices of size K*K
  size_t smem_size = 2 * K * K * sizeof(double);

  composite_hadamard_fused_kernel<<<num_blocks, threads_per_block, smem_size,
                                    stream>>>(d_A, d_B, d_C, d_D, d_R, K, P,
                                              sites_per_block);
}
