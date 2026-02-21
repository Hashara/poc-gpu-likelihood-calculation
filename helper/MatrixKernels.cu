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

// =========================================================================
// Template-specialized Fused Composite Hadamard + Scaling Kernel
// =========================================================================
//
// Templated on K (number of states) for compile-time optimizations:
//   - Full loop unrolling (#pragma unroll)
//   - if constexpr selects warp-shuffle vs shared-mem reduction at compile time
//   - K=4 specialization: register-cached A/C matrices, double2 vectorized
//     loads, no shared memory for matrices, no __syncthreads for matrix load
//
// For DNA (K=4):     256 threads → 64 sites/block, register-cached,
// warp-shuffle For Protein (K=20): 240 threads → 12 sites/block, shared-mem
// cached, shared-mem reduction

template <int K>
__global__ void composite_hadamard_fused_kernel(
    const double *__restrict__ a,      // K x K, column-major
    const double *__restrict__ b,      // K x P, site-major columns
    const double *__restrict__ c,      // K x K, column-major
    const double *__restrict__ d,      // K x P, site-major columns
    double *__restrict__ r,            // K x P, site-major columns
    uint8_t *__restrict__ scale_count, // Per-site scale counter
    int sites_per_block, int P, double scaling_threshold, int scaling_exp) {

  int tid = threadIdx.x;
  int local_site = tid / K;
  int state_i = tid % K;
  int global_site_j = blockIdx.x * sites_per_block + local_site;
  bool active = (global_site_j < P && local_site < sites_per_block);

  // -----------------------------------------------------------------------
  // Load A and C matrices — strategy depends on K
  // -----------------------------------------------------------------------
  double val = 0.0;

  if constexpr (K <= 4) {
    // =====================================================================
    // K=4 (DNA) — register-cached, vectorized, no shared memory
    // =====================================================================
    // Each thread loads its own row of A and C into registers.
    // 4×4 = 16 doubles per matrix = 128 bytes — fits easily in registers.
    // No __syncthreads() needed since each thread has its own copy.
    double a_row[K], c_row[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
      a_row[k] = __ldg(&a[k * K + state_i]);
      c_row[k] = __ldg(&c[k * K + state_i]);
    }

    if (active) {
      const double *bj = &b[global_site_j * K];
      const double *dj = &d[global_site_j * K];

      // Vectorized loads: load K doubles as K/2 double2's
      double s1 = 0.0, s2 = 0.0;
#pragma unroll
      for (int k = 0; k < K; k += 2) {
          // Vectorize the dot products by loading two doubles at once (double2), so each loop step handles k and k+1 together.
          //  This reduces memory transactions and computes s1/s2 two terms at a time, which is faster for small K (e.g., 4).
        double2 bv = *reinterpret_cast<const double2 *>(&bj[k]);
        double2 dv = *reinterpret_cast<const double2 *>(&dj[k]);
        s1 += a_row[k] * bv.x + a_row[k + 1] * bv.y;
        s2 += c_row[k] * dv.x + c_row[k + 1] * dv.y;
      }
      val = s1 * s2;
    }
  } else {
    // =====================================================================
    //  K>4 (e.g. K=20 for AA) — shared memory cached
    // =====================================================================
    // smem is one contiguous dynamic shared memory buffer.
    // s_a uses the first K*K doubles (first KxK matrix).
    // s_c is offset by K*K doubles, so it starts immediately after s_a (second KxK matrix).

    extern __shared__ double smem[];
    double *s_a = smem;
    double *s_c = smem + K * K;

    int threads_per_block = blockDim.x;

// Cooperatively load A and C into shared memory
#pragma unroll 4
    for (int idx = tid; idx < K * K; idx += threads_per_block) {
        // Use __ldg for read-only global memory access to leverage the read-only data
        // https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html?utm_source=openai#read-only-data-cache-load-function
      s_a[idx] = __ldg(&a[idx]);
      s_c[idx] = __ldg(&c[idx]);
    }
    __syncthreads();

    if (active) {
      const double *bj = &b[global_site_j * K];
      const double *dj = &d[global_site_j * K];

      double s1 = 0.0, s2 = 0.0;
#pragma unroll
      for (int k = 0; k < K; ++k) {
        double bk = __ldg(&bj[k]);
        double dk = __ldg(&dj[k]);
        s1 += s_a[k * K + state_i] * bk;
        s2 += s_c[k * K + state_i] * dk;
      }
      val = s1 * s2;
    }
  }

  // -----------------------------------------------------------------------
  // Per-site max-abs reduction + scaling
  // -----------------------------------------------------------------------
  double max_abs;

  if constexpr ((32 % K) == 0) {
    // WARP-SHUFFLE PATH: K divides 32 — site groups stay within one warp
    int lane = threadIdx.x % 32;
    int group_start = (lane / K) * K;
    unsigned group_mask = (((unsigned)1 << K) - 1) << group_start;

    double max_val = fabs(val);
#pragma unroll
    for (int stride = 16; stride >= 1; stride >>= 1) {
      if (stride < K) { // K is compile-time; compiler dead-code-eliminates
        double other = __shfl_down_sync(group_mask, max_val, stride);
        max_val = fmax(max_val, other);
      }
    }
    // _shfl_down_sync is a CUDA warp‑shuffle intrinsic provided by NVIDIA,
    // https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec4.pdf
    // It’s a built‑in GPU instruction used to exchange values between threads in
    // the same warp without shared memory
    max_abs = __shfl_sync(group_mask, max_val, group_start);
  } else {
    // SHARED-MEMORY PATH: K doesn't divide 32
    extern __shared__ double smem2[];
    double *s_reduce = smem2 + 2 * K * K;

    s_reduce[tid] = fabs(val);
    __syncthreads();

    int site_base = local_site * K;
#pragma unroll
    for (int stride = K / 2; stride > 0; stride >>= 1) {
      if (state_i < stride) {
        s_reduce[site_base + state_i] =
            fmax(s_reduce[site_base + state_i],
                 s_reduce[site_base + state_i + stride]);
      }
      __syncthreads();
    }
    if constexpr ((K & 1) != 0) {
      if (state_i == 0 && K > 1) {
        s_reduce[site_base] =
            fmax(s_reduce[site_base], s_reduce[site_base + K - 1]);
      }
      __syncthreads();
    }
    max_abs = s_reduce[site_base];
  }

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

__global__ void composite_hadamard_fused_kernel_dna4(
        const double *__restrict__ a,      // 4 x 4, column-major
        const double *__restrict__ b,      // 4 x P, site-major
        const double *__restrict__ c,      // 4 x 4, column-major
        const double *__restrict__ d,      // 4 x P, site-major
        double *__restrict__ r,            // 4 x P, site-major
        uint8_t *__restrict__ scale_count, // per-site
        int sites_per_block, int P, double scaling_threshold, int scaling_exp)
{
    constexpr int K = 4;

    int tid = threadIdx.x;
    int local_site = tid / K;
    int state_i    = tid % K;
    int global_site_j = blockIdx.x * sites_per_block + local_site;

    bool active = (global_site_j < P && local_site < sites_per_block);

    // ---- Load A,C rows into registers (each thread holds one row/col slice) ----
    double a_row[K], c_row[K];
#pragma unroll
    for (int k = 0; k < K; ++k) {
        a_row[k] = __ldg(&a[k * K + state_i]);
        c_row[k] = __ldg(&c[k * K + state_i]);
    }

    double val = 0.0;
    if (active) {
        const double *bj = &b[global_site_j * K];
        const double *dj = &d[global_site_j * K];

        // Vectorized dot (two doubles at once)
        double s1 = 0.0, s2 = 0.0;
#pragma unroll
        for (int k = 0; k < K; k += 2) {
            double2 bv = *reinterpret_cast<const double2 *>(&bj[k]);
            double2 dv = *reinterpret_cast<const double2 *>(&dj[k]);
            s1 += a_row[k]     * bv.x + a_row[k + 1] * bv.y;
            s2 += c_row[k]     * dv.x + c_row[k + 1] * dv.y;
        }
        val = s1 * s2;
    }

    // ---- Per-site max-abs reduction (warp shuffle path, since 32 % 4 == 0) ----
    int lane = tid % 32;
    int group_start = (lane / K) * K;
    unsigned group_mask = (((unsigned)1 << K) - 1) << group_start;

    double max_val = fabs(val);
#pragma unroll
    for (int stride = 16; stride >= 1; stride >>= 1) {
        if (stride < K) {
            double other = __shfl_down_sync(group_mask, max_val, stride);
            max_val = fmax(max_val, other);
        }
    }
    double max_abs = __shfl_sync(group_mask, max_val, group_start);

    // ---- Rescale and write ----
    if (active) {
        if (max_abs < scaling_threshold && max_abs > 0.0) {
            val = scalbn(val, scaling_exp);
            if (state_i == 0) scale_count[global_site_j] += 1;
        }
        r[global_site_j * K + state_i] = val;
    }
}

__global__ void composite_hadamard_fused_kernel_aa20(
        const double *__restrict__ a,      // 20 x 20, column-major
        const double *__restrict__ b,      // 20 x P, site-major
        const double *__restrict__ c,      // 20 x 20, column-major
        const double *__restrict__ d,      // 20 x P, site-major
        double *__restrict__ r,            // 20 x P, site-major
        uint8_t *__restrict__ scale_count, // per-site
        int sites_per_block, int P, double scaling_threshold, int scaling_exp)
{
    constexpr int K = 20;

    int tid = threadIdx.x;
    int local_site = tid / K;
    int state_i    = tid % K;
    int global_site_j = blockIdx.x * sites_per_block + local_site;

    bool active = (global_site_j < P && local_site < sites_per_block);

    extern __shared__ double smem[];
    // Layout: [A (K*K)] [C (K*K)] [reduce (threads_per_block)]
    double *s_a      = smem;                 // K*K
    double *s_c      = smem + K * K;         // K*K
    double *s_reduce = smem + 2 * K * K;     // blockDim.x

    int threads_per_block = blockDim.x;

    // ---- Cooperative load A/C ----
    for (int idx = tid; idx < K * K; idx += threads_per_block) {
        s_a[idx] = __ldg(&a[idx]);
        s_c[idx] = __ldg(&c[idx]);
    }
    __syncthreads();

    // ---- Compute ----
    double val = 0.0;
    if (active) {
        const double *bj = &b[global_site_j * K];
        const double *dj = &d[global_site_j * K];

        double s1 = 0.0, s2 = 0.0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            double bk = __ldg(&bj[k]);
            double dk = __ldg(&dj[k]);
            s1 += s_a[k * K + state_i] * bk;
            s2 += s_c[k * K + state_i] * dk;
        }
        val = s1 * s2;
    }

    // ---- Per-site reduction in shared memory (since 32 % 20 != 0) ----
    s_reduce[tid] = fabs(val);
    __syncthreads();

    int site_base = local_site * K;

    // Tree reduction assumes power-of-two; handle remainder carefully for K=20
    // Reduce 20 -> 10 -> 5 -> 2 -> 1 (with cleanup)
    if (state_i < 10) s_reduce[site_base + state_i] = fmax(s_reduce[site_base + state_i],
                                                           s_reduce[site_base + state_i + 10]);
    __syncthreads();

    if (state_i < 5)  s_reduce[site_base + state_i] = fmax(s_reduce[site_base + state_i],
                                                           s_reduce[site_base + state_i + 5]);
    __syncthreads();

    if (state_i < 2)  s_reduce[site_base + state_i] = fmax(s_reduce[site_base + state_i],
                                                           s_reduce[site_base + state_i + 2]);
    __syncthreads();

    if (state_i == 0) s_reduce[site_base] = fmax(s_reduce[site_base],
                                                 s_reduce[site_base + 1]);
    __syncthreads();

    double max_abs = s_reduce[site_base];

    // ---- Rescale and write ----
    if (active) {
        if (max_abs < scaling_threshold && max_abs > 0.0) {
            val = scalbn(val, scaling_exp);
            if (state_i == 0) scale_count[global_site_j] += 1;
        }
        r[global_site_j * K + state_i] = val;
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

// =========================================================================
// Template-specialized Fused Log-Likelihood Kernel
// =========================================================================
// Templated on K for full loop unrolling of the dot product.
// Uses grid-stride loop so each thread processes multiple sites.

template <int K>
__global__ void fused_log_likelihood_kernel(
    const double *__restrict__ baseFreq, // K values (cached on GPU)
    const double *__restrict__ rootL,    // K*P column-major (already on GPU)
    const uint8_t *__restrict__ scale_count, // P-length (GPU-resident)
    const int *__restrict__ freq,            // P-length pattern frequencies
    double *__restrict__ d_result,           // single output scalar
    int P, double log_scaling_threshold) {

  extern __shared__ double sdata[];

  int tid = threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  // Load base frequencies into registers (constant, K values)
  double bf[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    bf[k] = __ldg(&baseFreq[k]);
  }

  // Grid-stride loop: each thread accumulates over multiple sites
  double val = 0.0;
  for (int gid = blockIdx.x * blockDim.x + tid; gid < P; gid += total_threads) {
    const double *col = rootL + gid * K;

    double siteL = 0.0;
    if constexpr (K <= 4) {
// Vectorized loads for small K
#pragma unroll
      for (int k = 0; k < K; k += 2) {
        double2 rv = *reinterpret_cast<const double2 *>(&col[k]);
        siteL += bf[k] * rv.x + bf[k + 1] * rv.y;
      }
    } else {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        siteL += bf[k] * __ldg(&col[k]);
      }
    }

    val += __ldg(&freq[gid]) *
           (log(siteL) + __ldg(&scale_count[gid]) * log_scaling_threshold);
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

// =========================================================================
// Launch functions
// =========================================================================

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
  int target_threads = 256;
  int sites_per_block = target_threads / K;
  if (sites_per_block < 1)
    sites_per_block = 1;

  int threads_per_block = sites_per_block * K;
  int num_blocks = (P + sites_per_block - 1) / sites_per_block;

  // Shared memory: K<=4 uses registers (no smem for matrices), K>4 uses smem
  // + reduction workspace only if K doesn't divide 32
  size_t smem_size = 0;
  if (K > 4) {
    smem_size = (2 * K * K) * sizeof(double);
  }
  if ((32 % K) != 0) { // CUDA warp is 32 threads; if K doesn't divide 32, we need shared memory for reduction
    smem_size += threads_per_block * sizeof(double);
  }

  // Template dispatch on K
  switch (K) {
  case 4:
    composite_hadamard_fused_kernel<4>
        <<<num_blocks, threads_per_block, smem_size, stream>>>(
            d_A, d_B, d_C, d_D, d_R, d_scale_count, sites_per_block, P,
            scaling_threshold, scaling_exp);
    break;
  case 20:
    composite_hadamard_fused_kernel<20>
        <<<num_blocks, threads_per_block, smem_size, stream>>>(
            d_A, d_B, d_C, d_D, d_R, d_scale_count, sites_per_block, P,
            scaling_threshold, scaling_exp);
    break;
  default:
    // Fallback: instantiate for a generic K would require runtime K;
    // for now only DNA (4) and protein (20) are supported.
    // If needed, add more cases here.
    composite_hadamard_fused_kernel<20>
        <<<num_blocks, threads_per_block, smem_size, stream>>>(
            d_A, d_B, d_C, d_D, d_R, d_scale_count, sites_per_block, P,
            scaling_threshold, scaling_exp);
    break;
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
  if (blocks > 256)
    blocks = 256;
  size_t smem = threads * sizeof(double);

  // Template dispatch on K
  switch (K) {
  case 4:
    fused_log_likelihood_kernel<4><<<blocks, threads, smem, stream>>>(
        d_baseFreq, d_rootL, d_scale_count, d_freq, d_result, P,
        log_scaling_threshold);
    break;
  case 20:
    fused_log_likelihood_kernel<20><<<blocks, threads, smem, stream>>>(
        d_baseFreq, d_rootL, d_scale_count, d_freq, d_result, P,
        log_scaling_threshold);
    break;
  default:
    fused_log_likelihood_kernel<20><<<blocks, threads, smem, stream>>>(
        d_baseFreq, d_rootL, d_scale_count, d_freq, d_result, P,
        log_scaling_threshold);
    break;
  }
}

void launchCompositeHadamardFused_DNA4(
        const double *d_A, const double *d_B,
        const double *d_C, const double *d_D,
        double *d_R, uint8_t *d_scale_count,
        int P, double scaling_threshold, int scaling_exp,
        cudaStream_t stream)
{
    constexpr int K = 4;
    int target_threads = 256;
    int sites_per_block = target_threads / K; // 64
    if (sites_per_block < 1) sites_per_block = 1;

    int threads_per_block = sites_per_block * K;
    int num_blocks = (P + sites_per_block - 1) / sites_per_block;

    composite_hadamard_fused_kernel_dna4<<<
    num_blocks, threads_per_block, 0, stream>>>(
            d_A, d_B, d_C, d_D, d_R, d_scale_count,
            sites_per_block, P, scaling_threshold, scaling_exp);
}

void launchCompositeHadamardFused_AA20(
        const double *d_A, const double *d_B,
        const double *d_C, const double *d_D,
        double *d_R, uint8_t *d_scale_count,
        int P, double scaling_threshold, int scaling_exp,
        cudaStream_t stream)
{
    constexpr int K = 20;
    int target_threads = 256;
    int sites_per_block = target_threads / K; // 12 (since 256/20=12)
    if (sites_per_block < 1) sites_per_block = 1;

    int threads_per_block = sites_per_block * K; // 240
    int num_blocks = (P + sites_per_block - 1) / sites_per_block;

    // [A 400] + [C 400] + [reduce 240] doubles
    size_t smem_size = sizeof(double) * (2 * K * K + threads_per_block);

    composite_hadamard_fused_kernel_aa20<<<
    num_blocks, threads_per_block, smem_size, stream>>>(
            d_A, d_B, d_C, d_D, d_R, d_scale_count,
            sites_per_block, P, scaling_threshold, scaling_exp);
}
