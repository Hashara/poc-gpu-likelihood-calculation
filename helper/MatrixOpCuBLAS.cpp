//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#include "MatrixOpCuBLAS.h"
#include "MatrixKernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

MatrixOpCuBLAS::MatrixOpCuBLAS() {
  cublasCreate(&handle);
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
}

MatrixOpCuBLAS::~MatrixOpCuBLAS() {
  cublasDestroy(handle);
  // Free cached buffers
  if (d_A_cache)
    cudaFree(d_A_cache);
  if (d_C_cache)
    cudaFree(d_C_cache);
  if (d_AB_cache)
    cudaFree(d_AB_cache);
  if (d_CD_cache)
    cudaFree(d_CD_cache);
  // Destroy streams
  if (stream1)
    cudaStreamDestroy(stream1);
  if (stream2)
    cudaStreamDestroy(stream2);
}

void MatrixOpCuBLAS::ensureBuffer(double *&buf, size_t &current_elems,
                                  size_t needed_elems) {
  if (buf && current_elems >= needed_elems)
    return;
  if (buf)
    cudaFree(buf);
  cudaMalloc(&buf, needed_elems * sizeof(double));
  current_elems = needed_elems;
}

Matrix MatrixOpCuBLAS::multiply(const Matrix &A, const Matrix &B) {
  int M = A.rows(), N = A.cols(), P = B.cols();
  if (N != B.rows()) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication.");
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
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              /* m */ M,
              /* n */ P,
              /* k */ N, &alpha,
              /* A */ d_A, /* lda */ M,
              /* B */ d_B, /* ldb */ N, &beta,
              /* C */ d_C, /* ldc */ M);

  cudaMemcpy(C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}

Matrix MatrixOpCuBLAS::hadamard(const Matrix &A, const Matrix &B) {
  // For now, reuse the CUDA kernel hadamard (same as MatrixOpCUDA)
  int M = A.rows(), N = A.cols();
  if (M != B.rows() || N != B.cols()) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for Hadamard product.");
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

void MatrixOpCuBLAS::compositehadamard(const Matrix &A, const Matrix &B,
                                       const Matrix &C, const Matrix &D,
                                       Matrix &R, uint8_t *scale_count) {
  const int P = (int)B.cols();

  R.resize(numStates, P);

  size_t Asz = numStates * numStates;
  size_t Bsz = numStates * P;

  const size_t bytesA = Asz * sizeof(double);

  double *d_B = B.deviceData(); // Pre-uploaded by buildTipLikelihood
  double *d_D = D.deviceData(); // Pre-uploaded by buildTipLikelihood

  // Use cached buffers instead of malloc/free each call
  ensureBuffer(d_A_cache, cache_elems_A, Asz);
  ensureBuffer(d_C_cache, cache_elems_A, Asz); // Same size as A
  ensureBuffer(d_AB_cache, cache_elems_B, Bsz);
  ensureBuffer(d_CD_cache, cache_elems_B, Bsz);

  R.allocDevice();
  double *d_R = R.deviceData();

  // Async copy transition matrices to device (overlap with computation)
  cudaMemcpyAsync(d_A_cache, A.data(), bytesA, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_C_cache, C.data(), bytesA, cudaMemcpyHostToDevice, stream2);

  const double alpha = 1.0;
  const double beta0 = 0.0;

  // AB = A * B on stream1 (waits for d_A_cache copy)
  cublasSetStream(handle, stream1);
  auto st1 = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, numStates, P,
                         numStates, &alpha, d_A_cache, numStates, d_B,
                         numStates, &beta0, d_AB_cache, numStates);
  if (st1 != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasDgemm(A*B) failed");

  // CD = C * D on stream2 (runs in parallel with A*B)
  cublasSetStream(handle, stream2);
  auto st2 = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, numStates, P,
                         numStates, &alpha, d_C_cache, numStates, d_D,
                         numStates, &beta0, d_CD_cache, numStates);
  if (st2 != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cublasDgemm(C*D) failed");

  // Synchronize both streams before Hadamard
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  // R = AB ⊙ CD (element-wise product)
  int blockSize = 256;
  launchHadamard(d_AB_cache, d_CD_cache, d_R, numStates * P, blockSize);

  // No cudaFree calls - buffers are cached for reuse!
}

void MatrixOpCuBLAS::multiplyInPlace(const Matrix &A, const Matrix &B,
                                     Matrix &R) {
  // A = baseFreq, B = root likelihood, R = siteLikelihoods,
  int M = 1, N = A.cols(), P = B.cols();
  if (N != B.rows()) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication.");
  }

  R.resize(M, P); // R is a column vector of size P

  const double alpha = 1.0;
  const double beta = 0.0;

  double *d_A, *d_B, *d_R;
  size_t sizeA = N * sizeof(double);

  cudaMalloc(&d_A, sizeA);

  d_B = B.deviceData(); // <<< NO malloc, NO memcpy
  R.allocDevice();      // allocate device memory for R
  d_R = R.deviceData(); // device pointer for R

  cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice);

  // Column-major GEMM: C(MxP) = A(MxN) * B(NxP)
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              /* m */ M,
              /* n */ P,
              /* k */ N, &alpha,
              /* A */ d_A, /* lda */ M,
              /* B */ d_B, /* ldb */ N, &beta,
              /* C */ d_R, /* ldc */ M);

  R.copyDtoHAsync(0); // copy R back to host

  cudaFree(d_A); // Only free locally allocated buffer
}