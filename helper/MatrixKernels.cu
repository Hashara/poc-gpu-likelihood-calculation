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

void launchHadamard(const double* A, const double* B, double* C, int size, int blockSize) {
    int gridSize = (size + blockSize - 1) / blockSize;
    hadamardKernel<<<gridSize, blockSize>>>(A, B, C, size);
}

