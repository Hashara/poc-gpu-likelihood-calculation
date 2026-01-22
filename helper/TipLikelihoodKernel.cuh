//
// Created by Hashara Kumarasinghe on 17/12/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_TIPLIKELIHOODKERNEL_CUH
#define POC_GPU_LIKELIHOOD_CALCULATIONS_TIPLIKELIHOODKERNEL_CUH

#include <cstdint>   // for uint8_t
#include <cuda_runtime.h>


#pragma once
__global__ void build_tip_partials_kernel(
        const uint8_t* __restrict__ tip8, // device pointer
        double* __restrict__ l,           // device pointer
        int numStates,
        int numPatterns
);

void launchBuildTipPartials(
        const uint8_t* d_tip8,
        double* d_l,
        int numStates,
        int numPatterns,
        int blockSize,
        cudaStream_t stream
);


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_TIPLIKELIHOODKERNEL_CUH
