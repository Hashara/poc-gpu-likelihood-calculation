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
  // Free GPU-resident scale_count
  if (d_scale_count)
    cudaFree(d_scale_count);
  // Free cached freq + result buffers
  if (d_freq_cache)
    cudaFree(d_freq_cache);
  if (d_logL_result)
    cudaFree(d_logL_result);
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

  // Set constant memory once (K and P are fixed for the entire execution)
  static bool constants_set = false;
  if (!constants_set) {
    setKernelConstants(numStates, P);
    constants_set = true;
  }

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

  // ===== scale_count: GPU-resident, no per-call copies =====
  // d_scale_count is zeroed once per traversal by resetScaleCount()
  // and copied D->H once by syncScaleCount() after the traversal.
  if (!d_scale_count || d_scale_count_size < (size_t)P) {
    if (d_scale_count)
      cudaFree(d_scale_count);
    cudaMalloc(&d_scale_count, P * sizeof(uint8_t));
    d_scale_count_size = P;
  }

  // FUSED KERNEL + SCALING KERNEL - matches OpenACC two-loop pattern!
  launchCompositeHadamardFused(
      d_A,                   // Transition matrix P1
      d_B,                   // Left child partial likelihood
      d_C,                   // Transition matrix P2
      d_D,                   // Right child partial likelihood
      d_R,                   // Output: result
      d_scale_count,         // Scale count per site (GPU-resident)
      numStates,             // K = number of states
      P,                     // P = number of sites/patterns
      SCALING_THRESHOLD,     // Threshold for scaling
      SCALING_THRESHOLD_EXP, // Exponent for scalbn
      stream1                // CUDA stream
  );
  // No D->H copy or sync here -- scale_count stays on GPU until
  // syncScaleCount()
}

void MatrixOpCuBLAS::resetScaleCount(int P) {
  if (!d_scale_count || d_scale_count_size < (size_t)P) {
    if (d_scale_count)
      cudaFree(d_scale_count);
    cudaMalloc(&d_scale_count, P * sizeof(uint8_t));
    d_scale_count_size = P;
  }
  cudaMemsetAsync(d_scale_count, 0, P * sizeof(uint8_t), stream1);
}

void MatrixOpCuBLAS::syncScaleCount(uint8_t *host_ptr, int P) {
  cudaMemcpyAsync(host_ptr, d_scale_count, P * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost, stream1);
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

double MatrixOpCuBLAS::computeLogLikelihood(const Matrix &baseFreq,
                                            const Matrix &rootL,
                                            const int *freq, int numPatterns,
                                            double log_scaling_threshold) {
  int K = (int)baseFreq.cols(); // number of states
  int P = numPatterns;

  // --- 1. Cache base frequency on GPU (constant, uploaded once) ---
  size_t needed_elems = (size_t)K;
  if (!d_baseFreq_cache || baseFreq_elems < needed_elems) {
    if (d_baseFreq_cache)
      cudaFree(d_baseFreq_cache);
    cudaMalloc(&d_baseFreq_cache, needed_elems * sizeof(double));
    cudaMemcpy(d_baseFreq_cache, baseFreq.data(), needed_elems * sizeof(double),
               cudaMemcpyHostToDevice);
    baseFreq_elems = needed_elems;
  }

  // --- 2. Cache freq data on GPU (constant, uploaded once) ---
  if (!d_freq_cache || d_freq_elems < (size_t)P) {
    if (d_freq_cache)
      cudaFree(d_freq_cache);
    cudaMalloc(&d_freq_cache, P * sizeof(int));
    cudaMemcpyAsync(d_freq_cache, freq, P * sizeof(int), cudaMemcpyHostToDevice,
                    stream1);
    d_freq_elems = P;
  }

  // --- 3. Allocate result scalar on GPU (reused) ---
  if (!d_logL_result) {
    cudaMalloc(&d_logL_result, sizeof(double));
  }

  // --- 4. Single fused kernel: dot product + log + scale + reduction ---
  // Replaces cuBLAS DGEMM + separate reduction kernel.
  // For DNA (K=4), each thread does 4 MADs instead of launching cuBLAS
  // which has ~15-25us overhead for an M=1 DGEMM.
  double *d_rootL = rootL.deviceData(); // already on GPU from compositehadamard
  launchFusedLogLikelihood(d_baseFreq_cache, d_rootL, d_scale_count,
                           d_freq_cache, d_logL_result, K, P,
                           log_scaling_threshold, stream1);

  // --- 5. Copy single scalar back to host ---
  double logL = 0.0;
  cudaMemcpyAsync(&logL, d_logL_result, sizeof(double), cudaMemcpyDeviceToHost,
                  stream1);
  cudaStreamSynchronize(stream1);

  return logL;
}