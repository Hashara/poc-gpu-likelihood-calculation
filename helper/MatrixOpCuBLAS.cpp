//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#include "MatrixOpCuBLAS.h"
#include "../tree/Scaling.h"
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
  // Free cached base frequency buffer
  if (d_baseFreq_cache)
    cudaFree(d_baseFreq_cache);
  // Destroy streams
  if (stream1)
    cudaStreamDestroy(stream1);
  if (stream2)
    cudaStreamDestroy(stream2);
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

  // All matrices are now pre-uploaded to GPU:
  // - A, C: uploaded by Model::buildTransitionMatrix()
  // - B, D: uploaded by buildTipLikelihood() or previous compositehadamard()
  double *d_A = A.deviceData(); // Pre-uploaded by buildTransitionMatrix
  double *d_B = B.deviceData(); // Pre-uploaded by buildTipLikelihood
  double *d_C = C.deviceData(); // Pre-uploaded by buildTransitionMatrix
  double *d_D = D.deviceData(); // Pre-uploaded by buildTipLikelihood

  // Wait for transition matrices to finish uploading (they use stream 0)
  A.waitHtoD(stream1);
  C.waitHtoD(stream1);

  R.allocDevice();
  double *d_R = R.deviceData();

  // ===== scale_count handling =====
  // scale_count accumulates across ALL internal node computations.
  // Pattern: H->D copy (with accumulated counts), kernel increments, D->H copy
  static uint8_t *d_scale_count = nullptr;
  static size_t d_scale_count_size = 0;

  if (!d_scale_count || d_scale_count_size < (size_t)P) {
    if (d_scale_count)
      cudaFree(d_scale_count);
    cudaMalloc(&d_scale_count, P * sizeof(uint8_t));
    d_scale_count_size = P;
  }

  // Copy scale_count from host to GPU (preserves accumulated counts)
  cudaMemcpyAsync(d_scale_count, scale_count, P * sizeof(uint8_t),
                  cudaMemcpyHostToDevice, stream1);

  // FUSED KERNEL + SCALING KERNEL - matches OpenACC two-loop pattern!
  launchCompositeHadamardFused(d_A,           // Transition matrix P1
                               d_B,           // Left child partial likelihood
                               d_C,           // Transition matrix P2
                               d_D,           // Right child partial likelihood
                               d_R,           // Output: result
                               d_scale_count, // Scale count per site
                               numStates,     // K = number of states
                               P,             // P = number of sites/patterns
                               SCALING_THRESHOLD,     // Threshold for scaling
                               SCALING_THRESHOLD_EXP, // Exponent for scalbn
                               stream1                // CUDA stream
  );

  // Copy scale_count back to host (with new increments from this call)
  cudaMemcpyAsync(scale_count, d_scale_count, P * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost, stream1);

  // Sync to ensure scale_count is available on host
  cudaStreamSynchronize(stream1);
}

void MatrixOpCuBLAS::multiplyInPlace(const Matrix &A, const Matrix &B,
                                     Matrix &R) {
  // A = baseFreq (constant), B = root likelihood, R = siteLikelihoods
  int M = 1, N = A.cols(), P = B.cols();
  if (N != B.rows()) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication.");
  }

  R.resize(M, P); // R is a row vector of size P

  const double alpha = 1.0;
  const double beta = 0.0;

  size_t needed_elems = (size_t)N;

  // Cache base frequency buffer - it's constant for the entire tree traversal
  if (!d_baseFreq_cache || baseFreq_elems < needed_elems) {
    if (d_baseFreq_cache)
      cudaFree(d_baseFreq_cache);
    cudaMalloc(&d_baseFreq_cache, needed_elems * sizeof(double));
    cudaMemcpy(d_baseFreq_cache, A.data(), needed_elems * sizeof(double),
               cudaMemcpyHostToDevice);
    baseFreq_elems = needed_elems;
  }

  double *d_B = B.deviceData(); // Root likelihood already on GPU
  R.allocDevice();
  double *d_R = R.deviceData();

  // Column-major GEMM: R(1xP) = baseFreq(1xN) * rootL(NxP)
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              /* m */ M,
              /* n */ P,
              /* k */ N, &alpha,
              /* A */ d_baseFreq_cache, /* lda */ M,
              /* B */ d_B, /* ldb */ N, &beta,
              /* C */ d_R, /* ldc */ M);

  R.copyDtoHAsync(0); // copy R back to host

  // No cudaFree - base frequency is cached for reuse!
}