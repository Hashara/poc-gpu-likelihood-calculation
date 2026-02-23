//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
#include <cstdint> // for uint8_t

#pragma once

// Set constant memory values for K (numStates) and P (num sites)
// Must be called once before any kernel launch
void setKernelConstants(int K, int P);
__global__ void hadamardKernel(const double *A, const double *B, double *C,
                               int size);
__global__ void hadamard_scale_kernel(const double *__restrict__ ab,
                                      const double *__restrict__ cd,
                                      double *__restrict__ r,
                                      uint8_t *__restrict__ scale_count,
                                      double scaling_threshold,
                                      int scaling_exp);
__global__ void composite_hadamard_fused_kernel(
    const double *__restrict__ a, const double *__restrict__ b,
    const double *__restrict__ c, const double *__restrict__ d,
    double *__restrict__ r, uint8_t *__restrict__ scale_count,
    int sites_per_block, double scaling_threshold, int scaling_exp);

__global__ void scaling_kernel(double *__restrict__ r,
                               uint8_t *__restrict__ scale_count,
                               double scaling_threshold, int scaling_exp);

void launchHadamard(const double *A, const double *B, double *C, int size,
                    int blockSize);
void launchCompositeHadamard(const double *d_AB, const double *d_CD,
                             double *d_R, uint8_t *d_scale_count, int numStates,
                             int P, int blockSize, double scaling_threshold,
                             int scaling_exp);
void launchCompositeHadamardFused(const double *d_A, const double *d_B,
                                  const double *d_C, const double *d_D,
                                  double *d_R, uint8_t *d_scale_count, int K,
                                  int P, double scaling_threshold,
                                  int scaling_exp, cudaStream_t stream);

void launchFusedLogLikelihood(const double *d_baseFreq, const double *d_rootL,
                              const uint8_t *d_scale_count, const int *d_freq,
                              double *d_result, int K, int P,
                              double log_scaling_threshold,
                              cudaStream_t stream);
__global__ void composite_hadamard_fused_kernel_dna4(
        const double *__restrict__ a,      // 4 x 4, column-major
        const double *__restrict__ b,      // 4 x P, site-major
        const double *__restrict__ c,      // 4 x 4, column-major
        const double *__restrict__ d,      // 4 x P, site-major
        double *__restrict__ r,            // 4 x P, site-major
        uint8_t *__restrict__ scale_count, // per-site
        int sites_per_block, int P, double scaling_threshold, int scaling_exp);

__global__ void composite_hadamard_fused_kernel_aa20(
        const double *__restrict__ a,      // 20 x 20, column-major
        const double *__restrict__ b,      // 20 x P, site-major
        const double *__restrict__ c,      // 20 x 20, column-major
        const double *__restrict__ d,      // 20 x P, site-major
        double *__restrict__ r,            // 20 x P, site-major
        uint8_t *__restrict__ scale_count, // per-site
        int sites_per_block, int P, double scaling_threshold, int scaling_exp);

void launchCompositeHadamardFused_DNA4(
        const double *d_A, const double *d_B,
        const double *d_C, const double *d_D,
        double *d_R, uint8_t *d_scale_count,
        int P, double scaling_threshold, int scaling_exp,
        cudaStream_t stream);

void launchCompositeHadamardFused_AA20(
        const double *d_A, const double *d_B,
        const double *d_C, const double *d_D,
        double *d_R, uint8_t *d_scale_count,
        int P, double scaling_threshold, int scaling_exp,
        cudaStream_t stream);

// Build transition matrices directly on device (row-major layout used by
// existing CUBLAS path in Model::buildTransitionMatrix).
void launchBuildTransitionMatrixJC(double *d_P, double t, cudaStream_t stream);
void launchBuildTransitionMatrixPOISSON(double *d_P, double t,
                                        cudaStream_t stream);

// =========================================================================
// Eigenspace partial likelihood kernel (IQ-TREE-style formulation)
// =========================================================================
//
// For each pattern:
//   L_dad = inv_evec * ((echild_left * L_left) ⊙ (echild_right * L_right))
//
// Mathematically equivalent to (P1*L_left) ⊙ (P2*L_right) but uses
// eigendecomposition: P(t) = U·diag(exp(λt))·U⁻¹ split into
// echildren = U·diag(exp(λt)) and a final inv_evec back-transform.
//
void launchEigenPartialLikelihood(
    const double *d_echild_left,       // [K * K] column-major
    const double *d_partial_lh_left,   // [K * P] site-major
    const double *d_echild_right,      // [K * K] column-major
    const double *d_partial_lh_right,  // [K * P] site-major
    const double *d_inv_evec,          // [K * K] column-major
    double       *d_partial_lh_dad,    // [K * P] output site-major
    uint8_t      *d_scale_count,       // [P] scale counter (accumulated)
    int           K,                   // number of states
    int           P,                   // number of patterns
    double        scaling_threshold,
    int           scaling_exp,
    cudaStream_t  stream
);

// Build echildren matrix on GPU: E[i,j] = U[i,j] * exp(eigenval[j] * t)
void launchBuildEChildren(
    const double *d_eigenvectors,  // [K * K] column-major U
    const double *d_eigenvalues,   // [K] eigenvalues
    double       *d_echildren,     // [K * K] output column-major
    int           K,
    double        t,
    cudaStream_t  stream
);


#endif // POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
