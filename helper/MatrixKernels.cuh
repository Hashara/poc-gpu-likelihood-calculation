//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH


#pragma once
__global__ void hadamardKernel(const double* A, const double* B, double* C, int size);
void launchHadamard(const double* A, const double* B, double* C, int size, int blockSize);


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXKERNELS_CUH
