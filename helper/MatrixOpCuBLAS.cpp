//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#include "MatrixOpCuBLAS.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include "MatrixKernels.cuh"

MatrixOpCuBLAS::MatrixOpCuBLAS() {
    cublasCreate(&handle);
}

MatrixOpCuBLAS::~MatrixOpCuBLAS() {
    cublasDestroy(handle);
}

Matrix MatrixOpCuBLAS::multiply(const Matrix& A, const Matrix& B) {
    int M = A.rows(), N = A.cols(), P = B.cols();
    if (N != B.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix C(M, P);

    const double alpha = 1.0;
    const double beta = 0.0;

    double *d_A, *d_B, *d_C;
    size_t sizeA = M * N * sizeof(double);
    size_t sizeB = N * P * sizeof(double);
    size_t sizeC = M * P * sizeof(double);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice);

    // cuBLASDgemm is the double precision matrix multiplication function
    /**
     * cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)
       CUBLAS_OP_N means no transpose, CUBLAS_OP_T means transpose
       alpha -> Scalar multiplier for A * B
       lda, ldb, ldc -> Leading dimensions of A, B, C respectively
       beta -> Scalar multiplier for C
     */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                P, M, N,
                &alpha,
                d_B, P,
                d_A, N,
                &beta,
                d_C, P);

    cudaMemcpy(C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

Matrix MatrixOpCuBLAS::hadamard(const Matrix& A, const Matrix& B) {
    // For now, reuse the CUDA kernel hadamard (same as MatrixOpCUDA)
    int M = A.rows(), N = A.cols();
    if (M != B.rows() || N != B.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
    }

    Matrix C(M, N);
    double *d_A, *d_B, *d_C;
    int size = M * N;
    size_t bytes = size * sizeof(double);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    launchHadamard(d_A, d_B, d_C, size, blockSize);

    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

