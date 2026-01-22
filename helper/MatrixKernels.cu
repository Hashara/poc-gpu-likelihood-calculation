//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#include "MatrixKernels.cuh"
#include <cuda_runtime.h>


__global__ void hadamardKernel(const double* A, const double* B, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void hadamard_scale_kernel(
        const double* __restrict__ ab,   // size: numStates * P
        const double* __restrict__ cd,   // size: numStates * P
        double* __restrict__ r,          // size: numStates * P
        uint8_t* __restrict__ scale_count,
        int numStates,
        int P,
        double scaling_threshold,
        int scaling_exp
) {
    int j = blockIdx.x; // one block per column (site)
    if (j >= P) return;

    // 1) compute r = ab * cd and local max
    double local_max = 0.0;

    for (int i = threadIdx.x; i < numStates; i += blockDim.x) {
        double v = ab[j * numStates + i] * cd[j * numStates + i];
        r[j * numStates + i] = v;
        double av = fabs(v);
        if (av > local_max) local_max = av;
    }

    // 2) reduce max within block
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }

    double max_abs = sdata[0];

    // 3) rescale if below threshold
    if (max_abs < scaling_threshold) {
        for (int i = threadIdx.x; i < numStates; i += blockDim.x) {
            r[j * numStates + i] = scalbn(r[j * numStates + i], scaling_exp);
        }
        if (tid == 0) scale_count[j] += 1;
    }
}

__global__ void composite_hadamard_fused_kernel(
        const double* __restrict__ a, // KxK, column-major: a[k*K + i] = A(i,k)
        const double* __restrict__ b, // KxP, site-major columns: b[j*K + k] = B(k,j)
        const double* __restrict__ c, // KxK, column-major
        const double* __restrict__ d, // KxP, site-major columns
        double* __restrict__ r,       // KxP, site-major columns: r[j*K + i]
        int K,
        int P
) {
    int j = blockIdx.x;  // one block per site/column
    int i = threadIdx.x; // one thread per state

    if (j >= P || i >= K) return;

    const double* bj = &b[j * K];
    const double* dj = &d[j * K];

    double s1 = 0.0;
    double s2 = 0.0;

    // dot over k
    for (int k = 0; k < K; ++k) {
        s1 += a[k * K + i] * bj[k];
        s2 += c[k * K + i] * dj[k];
    }

    r[j * K + i] = s1 * s2;
}


void launchHadamard(const double* A, const double* B, double* C, int size, int blockSize) {
    int gridSize = (size + blockSize - 1) / blockSize;
    hadamardKernel<<<gridSize, blockSize>>>(A, B, C, size);
}

void launchCompositeHadamard(
        const double* d_AB,
        const double* d_CD,
        double* d_R,
        uint8_t* d_scale_count,
        int numStates,
        int P,
        int blockSize,
        double scaling_threshold,
        int scaling_exp
) {
    int gridSize = P;
    size_t sharedBytes = blockSize * sizeof(double);

    hadamard_scale_kernel<<<gridSize, blockSize, sharedBytes>>>(
            d_AB, d_CD, d_R, d_scale_count,
            numStates, P,
            scaling_threshold, scaling_exp
    );
}

void launchCompositeHadamardFused(
        const double* d_A,
        const double* d_B,
        const double* d_C,
        const double* d_D,
        double* d_R,
        int K,
        int P,
        cudaStream_t stream = 0
) {
    dim3 grid(P);
    dim3 block(K);               // K threads per block (K <= 1024)
    composite_hadamard_fused_kernel<<<grid, block, 0, stream>>>(
            d_A, d_B, d_C, d_D, d_R, K, P
    );
}
