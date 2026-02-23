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
  cudaEventCreateWithFlags(&stream1_ready_event, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&d2h_done_event, cudaEventDisableTiming);
  if (cublasSetStream(handle, stream1) != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to bind cuBLAS handle to stream1.");
  }
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
  if (h_logL_result_pinned)
    cudaFreeHost(h_logL_result_pinned);
  // Free eigenspace buffers
  if (d_eigenvectors_) cudaFree(d_eigenvectors_);
  if (d_eigenvalues_) cudaFree(d_eigenvalues_);
  if (d_inv_evec_) cudaFree(d_inv_evec_);
  if (d_echild_left_) cudaFree(d_echild_left_);
  if (d_echild_right_) cudaFree(d_echild_right_);
  // Destroy streams
  if (stream1)
    cudaStreamDestroy(stream1);
  if (stream2)
    cudaStreamDestroy(stream2);
  if (stream1_ready_event)
    cudaEventDestroy(stream1_ready_event);
  if (d2h_done_event)
    cudaEventDestroy(d2h_done_event);
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
  switch (numStates) {
  case 4:
    compositehadamardSpecialized<4>(A, B, C, D, R, scale_count);
    break;
  case 20:
    compositehadamardSpecialized<20>(A, B, C, D, R, scale_count);
    break;
  default:
    throw std::invalid_argument(
        "MatrixOpCuBLAS::compositehadamard only supports numStates 4 or 20");
  }
}

template <int K>
void MatrixOpCuBLAS::compositehadamardSpecialized(
    const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D,
    Matrix &R, uint8_t *scale_count) {
  static_assert(K == 4 || K == 20,
                "compositehadamardSpecialized only supports K=4 or K=20");

  (void)scale_count; // scale_count is GPU-resident in d_scale_count for CUBLAS.
  const int P = (int)B.cols();

  // Set constant memory once per K path.
  static bool constants_set = false;
  if (!constants_set) {
    setKernelConstants(K, P);
    constants_set = true;
  }

  R.resize(K, P);

  // All matrices are now pre-uploaded to GPU:
  // - A, C: uploaded by Model::buildTransitionMatrix()
  // - B, D: uploaded by buildTipLikelihood() or previous compositehadamard()
  double *d_A = A.deviceData();
  double *d_B = B.deviceData();
  double *d_C = C.deviceData();
  double *d_D = D.deviceData();

  // Transition matrices are produced on stream1 in the CUBLAS path.
  // No cross-stream wait is needed here.

  R.allocDevice();
  double *d_R = R.deviceData();

  // d_scale_count is zeroed once per traversal by resetScaleCount()
  // and copied D->H once by syncScaleCount() after traversal.
  if (!d_scale_count || d_scale_count_size < (size_t)P) {
    if (d_scale_count)
      cudaFree(d_scale_count);
    cudaMalloc(&d_scale_count, P * sizeof(uint8_t));
    d_scale_count_size = P;
  }

  if constexpr (K == 4) {
    launchCompositeHadamardFused_DNA4(
        d_A, d_B, d_C, d_D, d_R, d_scale_count, P, SCALING_THRESHOLD,
        SCALING_THRESHOLD_EXP, stream1);
  } else {
    launchCompositeHadamardFused_AA20(
        d_A, d_B, d_C, d_D, d_R, d_scale_count, P, SCALING_THRESHOLD,
        SCALING_THRESHOLD_EXP, stream1);
  }
}

template void MatrixOpCuBLAS::compositehadamardSpecialized<4>(
    const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D,
    Matrix &R, uint8_t *scale_count);
template void MatrixOpCuBLAS::compositehadamardSpecialized<20>(
    const Matrix &A, const Matrix &B, const Matrix &C, const Matrix &D,
    Matrix &R, uint8_t *scale_count);

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
  cudaEventRecord(stream1_ready_event, stream1);
  cudaStreamWaitEvent(stream2, stream1_ready_event, 0);
  cudaMemcpyAsync(host_ptr, d_scale_count, P * sizeof(uint8_t),
                  cudaMemcpyDeviceToHost, stream2);
  cudaEventRecord(d2h_done_event, stream2);
  cudaEventSynchronize(d2h_done_event);
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
  if (!h_logL_result_pinned) {
    if (cudaMallocHost(&h_logL_result_pinned, sizeof(double)) != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate pinned host buffer for log-likelihood result.");
    }
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
  cudaEventRecord(stream1_ready_event, stream1);
  cudaStreamWaitEvent(stream2, stream1_ready_event, 0);
  cudaMemcpyAsync(h_logL_result_pinned, d_logL_result, sizeof(double),
                  cudaMemcpyDeviceToHost, stream2);
  cudaEventRecord(d2h_done_event, stream2);
  cudaEventSynchronize(d2h_done_event);

  return *h_logL_result_pinned;
}

// =========================================================================
// Eigenspace methods
// =========================================================================

void MatrixOpCuBLAS::uploadEigenData(const double *eigenvectors,
                                      const double *eigenvalues,
                                      const double *inv_eigenvectors,
                                      int K) {
  eigen_K_ = K;
  size_t mat_bytes = K * K * sizeof(double);
  size_t vec_bytes = K * sizeof(double);

  // Allocate device buffers
  if (d_eigenvectors_) cudaFree(d_eigenvectors_);
  if (d_eigenvalues_) cudaFree(d_eigenvalues_);
  if (d_inv_evec_) cudaFree(d_inv_evec_);
  if (d_echild_left_) cudaFree(d_echild_left_);
  if (d_echild_right_) cudaFree(d_echild_right_);

  cudaMalloc(&d_eigenvectors_, mat_bytes);
  cudaMalloc(&d_eigenvalues_, vec_bytes);
  cudaMalloc(&d_inv_evec_, mat_bytes);
  cudaMalloc(&d_echild_left_, mat_bytes);
  cudaMalloc(&d_echild_right_, mat_bytes);

  // Upload constant data
  cudaMemcpyAsync(d_eigenvectors_, eigenvectors, mat_bytes,
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_eigenvalues_, eigenvalues, vec_bytes,
                  cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_inv_evec_, inv_eigenvectors, mat_bytes,
                  cudaMemcpyHostToDevice, stream1);
}

void MatrixOpCuBLAS::buildEChildrenGPU(double t, double *d_echildren) {
  launchBuildEChildren(d_eigenvectors_, d_eigenvalues_, d_echildren,
                       eigen_K_, t, stream1);
}

void MatrixOpCuBLAS::eigenCompositeHadamard(
    const double *d_echild_left,
    const Matrix &L_left,
    const double *d_echild_right,
    const Matrix &L_right,
    Matrix &R)
{
  int K = eigen_K_;
  int P = (int)L_left.cols();

  R.resize(K, P);
  R.allocDevice();

  // Ensure scale_count is allocated
  if (!d_scale_count || d_scale_count_size < (size_t)P) {
    if (d_scale_count) cudaFree(d_scale_count);
    cudaMalloc(&d_scale_count, P * sizeof(uint8_t));
    d_scale_count_size = P;
  }

  // L_left and L_right are already device-resident (from previous compositeHadamard
  // or buildTipLikelihood). R is also device-allocated.
  launchEigenPartialLikelihood(
      d_echild_left,
      L_left.deviceData(),
      d_echild_right,
      L_right.deviceData(),
      d_inv_evec_,
      R.deviceData(),
      d_scale_count,
      K, P,
      SCALING_THRESHOLD,
      SCALING_THRESHOLD_EXP,
      stream1
  );
}

void MatrixOpCuBLAS::transformToEigenSpace(Matrix &L) {
  // L_out = U⁻¹ · L_in   (K×K) * (K×P) = (K×P)
  int K = eigen_K_;
  int P = (int)L.cols();

  // Allocate temp buffer for result
  double *d_temp = nullptr;
  cudaMalloc(&d_temp, K * P * sizeof(double));

  double alpha = 1.0, beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              K, P, K, &alpha,
              d_inv_evec_, K,
              L.deviceData(), K, &beta,
              d_temp, K);

  // Copy result back to L's device buffer
  cudaMemcpyAsync(L.deviceData(), d_temp, K * P * sizeof(double),
                   cudaMemcpyDeviceToDevice, stream1);
  cudaFree(d_temp);
}

void MatrixOpCuBLAS::transformFromEigenSpace(Matrix &L) {
  // L_out = U · L_in   (K×K) * (K×P) = (K×P)
  int K = eigen_K_;
  int P = (int)L.cols();

  // Allocate temp buffer for result
  double *d_temp = nullptr;
  cudaMalloc(&d_temp, K * P * sizeof(double));

  double alpha = 1.0, beta = 0.0;
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              K, P, K, &alpha,
              d_eigenvectors_, K,
              L.deviceData(), K, &beta,
              d_temp, K);

  // Copy result back to L's device buffer
  cudaMemcpyAsync(L.deviceData(), d_temp, K * P * sizeof(double),
                   cudaMemcpyDeviceToDevice, stream1);
  cudaFree(d_temp);
}
