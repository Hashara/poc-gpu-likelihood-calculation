//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
#include <cstdint> // for uint8_t

#pragma once
__global__ void hadamardKernel(const double *A, const double *B, double *C,
                               int size);
__global__ void
hadamard_scale_kernel(const double *__restrict__ ab, // size: numStates * P
                      const double *__restrict__ cd, // size: numStates * P
                      double *__restrict__ r,        // size: numStates * P
                      uint8_t *__restrict__ scale_count, int numStates, int P,
                      double scaling_threshold, int scaling_exp);
__global__ void composite_hadamard_fused_kernel(
    const double *__restrict__ a, // KxK, column-major: a[k*K + i] = A(i,k)
    const double
        *__restrict__ b, // KxP, site-major columns: b[j*K + k] = B(k,j)
    const double *__restrict__ c, // KxK, column-major
    const double *__restrict__ d, // KxP, site-major columns
    double *__restrict__ r,       // KxP, site-major columns: r[j*K + i]
    int K, int P,
    int sites_per_block // Number of sites processed per block
);

__global__ void
scaling_kernel(double *__restrict__ r,            // KxP, site-major columns
               uint8_t *__restrict__ scale_count, // Per-site scale counter
               int K, int P, double scaling_threshold, int scaling_exp);

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
#endif // POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
