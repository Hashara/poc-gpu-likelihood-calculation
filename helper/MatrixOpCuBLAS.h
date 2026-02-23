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
  template <int K>
  void compositehadamardSpecialized(const Matrix &A, const Matrix &B,
                                    const Matrix &C, const Matrix &D,
                                    Matrix &R, uint8_t *scale_count);
  void multiplyInPlace(const Matrix &A, const Matrix &B, Matrix &R) override;

  // GPU-resident scale_count lifecycle management
  void resetScaleCount(int P); // Zero on GPU at start of traversal
  void syncScaleCount(uint8_t *host_ptr, int P); // Copy D→H at end of traversal

  // GPU reduction: DGEMM + log-likelihood sum, returns scalar logL
  double computeLogLikelihood(const Matrix &baseFreq, const Matrix &rootL,
                              const int *freq, int numPatterns,
                              double log_scaling_threshold);

  // ---- Eigenspace partial likelihood (IQ-TREE formulation) ----

  // Eigenspace partial likelihood:
  //   L_dad = inv_evec * ((E_left * L_left) ⊙ (E_right * L_right))
  void eigenCompositeHadamard(
      const double *d_echild_left,   // [K*K] device ptr
      const Matrix &L_left,          // [K*P] left child (device-resident)
      const double *d_echild_right,  // [K*K] device ptr
      const Matrix &L_right,         // [K*P] right child (device-resident)
      Matrix &R                      // [K*P] output (device-allocated)
  );

  // Upload eigendecomposition data to GPU (call once after model init)
  void uploadEigenData(const double *eigenvectors, const double *eigenvalues,
                       const double *inv_eigenvectors, int K);

  // Build echildren matrix on GPU for a given branch length
  void buildEChildrenGPU(double t, double *d_echildren);

  // Transform partial likelihoods between original and eigenspace bases
  // L_out = U⁻¹ · L_in  (original → eigenspace)
  void transformToEigenSpace(Matrix &L);
  // L_out = U · L_in  (eigenspace → original)
  void transformFromEigenSpace(Matrix &L);

  // Get device pointers
  double* getDeviceInvEvec() const { return d_inv_evec_; }
  double* getDeviceEigenvectors() const { return d_eigenvectors_; }
  double* getDeviceEigenvalues() const { return d_eigenvalues_; }
  double* getDeviceEChildLeft() const { return d_echild_left_; }
  double* getDeviceEChildRight() const { return d_echild_right_; }
  int getEigenK() const { return eigen_K_; }
#endif // USE_CUBLAS
private:
  cublasHandle_t handle;

  // Cached base frequency buffer for multiplyInPlace (constant during tree
  // traversal)
  double *d_baseFreq_cache = nullptr;
  size_t baseFreq_elems = 0;

  // GPU-resident scale_count (persists across compositehadamard calls)
  uint8_t *d_scale_count = nullptr;
  size_t d_scale_count_size = 0;

  // Cached freq + result buffers for GPU log-likelihood reduction
  int *d_freq_cache = nullptr;
  size_t d_freq_elems = 0;
  double *d_logL_result = nullptr; // single scalar on GPU
  double *h_logL_result_pinned = nullptr;

  // Eigenspace buffers (allocated once, reused for all branches)
  double *d_eigenvectors_ = nullptr;   // [K*K] U
  double *d_eigenvalues_ = nullptr;    // [K]   λ
  double *d_inv_evec_ = nullptr;       // [K*K] U⁻¹
  double *d_echild_left_ = nullptr;    // [K*K] working buffer
  double *d_echild_right_ = nullptr;   // [K*K] working buffer
  int eigen_K_ = 0;

  // Streams for async execution
  cudaStream_t stream1 = nullptr;
  cudaStream_t stream2 = nullptr;
  cudaEvent_t stream1_ready_event = nullptr;
  cudaEvent_t d2h_done_event = nullptr;

public:
  // Expose stream for reuse by LikelihoodCalculator
  cudaStream_t getStream() const { return stream1; }
};

#endif // POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H
