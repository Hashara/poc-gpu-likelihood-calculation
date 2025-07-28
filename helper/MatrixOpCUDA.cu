//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#include "MatrixOpCUDA.cuh"
#include <cuda_runtime.h>

__global__ void matMulKernel(const double* A, const double* B, double* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < P) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

Matrix MatrixOpCUDA::multiply(const Matrix& A, const Matrix& B) {
    int M = A.rows(), N = A.cols(), P = B.cols();
    if (N != B.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix C(M, P);
    const double* h_A = A.data();
    const double* h_B = B.data();
    double* h_C = C.data();

    double *d_A, *d_B, *d_C;
    size_t sizeA = M * N * sizeof(double);
    size_t sizeB = N * P * sizeof(double);
    size_t sizeC = M * P * sizeof(double);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((P + 15) / 16, (M + 15) / 16);
    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, P);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

__global__ void hadamardKernel(const double* A, const double* B, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

Matrix MatrixOpCUDA::hadamard(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
    }

    int M = A.rows(), N = A.cols();
    int size = M * N;
    Matrix C(M, N);

    const double* h_A = A.data();
    const double* h_B = B.data();
    double* h_C = C.data();

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(double));
    cudaMalloc(&d_B, size * sizeof(double));
    cudaMalloc(&d_C, size * sizeof(double));

    cudaMemcpy(d_A, h_A, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    hadamardKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);

    cudaMemcpy(h_C, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}
