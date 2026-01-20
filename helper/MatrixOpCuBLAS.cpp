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
    // Column-major GEMM: C(MxP) = A(MxN) * B(NxP)
    cublasDgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
            /* m */ M,
            /* n */ P,
            /* k */ N,
                             &alpha,
            /* A */ d_A, /* lda */ M,
            /* B */ d_B, /* ldb */ N,
                             &beta,
            /* C */ d_C, /* ldc */ M);

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

void MatrixOpCuBLAS::compositehadamard(
        const Matrix& A, const Matrix& B,
        const Matrix& C, const Matrix& D,
        Matrix& R, uint8_t* scale_count
) {
    const int P = (int) B.cols();

    R.resize(numStates, P);

    size_t Asz = numStates * numStates;
    size_t Bsz = numStates * P;

    const size_t bytesA = Asz * sizeof(double);
    const size_t bytesB = Bsz * sizeof(double);


    double *d_B = B.deviceData();   // <<< NO malloc, NO memcpy
    double *d_D = D.deviceData();   // <<< NO malloc, NO memcpy

    double *d_A = nullptr, *d_C = nullptr;
    double *d_AB = nullptr, *d_CD = nullptr, *d_R = nullptr;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_C, bytesA);
    cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), bytesA, cudaMemcpyHostToDevice);

    cudaMalloc(&d_AB, bytesB);
    cudaMalloc(&d_CD, bytesB);
    R.allocDevice();                 // allocate device memory for R
    d_R = R.deviceData();           // device pointer for R

    const double alpha = 1.0;
    const double beta0 = 0.0;

    // AB = A * B
    auto st1 = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           numStates, P, numStates,
                           &alpha,
                           d_A, numStates,
                           d_B, numStates,
                           &beta0,
                           d_AB, numStates);
    if (st1 != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasDgemm(A*B) failed");

    // CD = C * D
    auto st2 = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           numStates, P, numStates,
                           &alpha,
                           d_C, numStates,
                           d_D, numStates,
                           &beta0,
                           d_CD, numStates);
    if (st2 != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasDgemm(C*D) failed");

    // R = AB ⊙ CD (NO scaling)
    int blockSize = 256;
    launchHadamard(d_AB, d_CD, d_R, numStates * P, blockSize);

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_AB);
    cudaFree(d_CD);

}

void MatrixOpCuBLAS::multiplyInPlace(const Matrix& A, const Matrix& B, Matrix& R) {
    // A = baseFreq, B = root likelihood, R = siteLikelihoods,
    int M=1, N = A.cols(), P = B.cols();
    if (N != B.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    R.resize(M, P); // R is a column vector of size P

    const double alpha = 1.0;
    const double beta = 0.0;

    double *d_A, *d_B, *d_R;
    size_t sizeA = N * sizeof(double);
    size_t sizeB = N * P * sizeof(double);
    size_t sizeC = P * sizeof(double);

    cudaMalloc(&d_A, sizeA);

    d_B = B.deviceData();   // <<< NO malloc, NO memcpy
    R.allocDevice();                 // allocate device memory for R
    d_R = R.deviceData();           // device pointer for R

    cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice);


    // Column-major GEMM: C(MxP) = A(MxN) * B(NxP)
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
            /* m */ M,
            /* n */ P,
            /* k */ N,
                &alpha,
            /* A */ d_A, /* lda */ M,
            /* B */ d_B, /* ldb */ N,
                &beta,
            /* C */ d_R, /* ldc */ M);

    cudaMemcpy(R.data(), d_R, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_R);

}