//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H

#include "MatrixOp.h"
#include <cublas_v2.h>

class MatrixOpCuBLAS : public MatrixOp {
public:
  MatrixOpCuBLAS();
  ~MatrixOpCuBLAS() override;
  Matrix multiply(const Matrix &A, const Matrix &B) override;
  Matrix hadamard(const Matrix &A, const Matrix &B) override;

#ifdef USE_CUBLAS
  void compositehadamard(const Matrix &A, const Matrix &B, const Matrix &C,
                         const Matrix &D, Matrix &R,
                         uint8_t *scale_count) override;
  void multiplyInPlace(const Matrix &A, const Matrix &B, Matrix &R) override;

  // GPU-resident scale_count lifecycle management
  void resetScaleCount(int P); // Zero on GPU at start of traversal
  void syncScaleCount(uint8_t *host_ptr, int P); // Copy D→H at end of traversal
#endif                                           // USE_CUBLAS
private:
  cublasHandle_t handle;

  // Cached base frequency buffer for multiplyInPlace (constant during tree
  // traversal)
  double *d_baseFreq_cache = nullptr;
  size_t baseFreq_elems = 0;

  // GPU-resident scale_count (persists across compositehadamard calls)
  uint8_t *d_scale_count = nullptr;
  size_t d_scale_count_size = 0;

  // Streams for async execution
  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;

public:
  // Expose stream for reuse by LikelihoodCalculator
  cudaStream_t getStream() const { return stream1; }
};

#endif // POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H
