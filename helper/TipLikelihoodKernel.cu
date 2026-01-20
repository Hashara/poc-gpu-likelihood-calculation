//
// Created by Hashara Kumarasinghe on 17/12/2025.
//

#include "TipLikelihoodKernel.cuh"
#include <stdint.h>

__global__ void build_tip_partials_kernel(
        const uint8_t* __restrict__ tip8, // device pointer
        double* __restrict__ l,           // device pointer
        int numStates,
        int numPatterns
) {
    int p = (int)blockIdx.x;                          // pattern index
    int s = (int)(blockIdx.y * blockDim.x + threadIdx.x); // state index

    if (p >= numPatterns || s >= numStates) return;

    uint8_t ss = tip8[p];
    l[(size_t)p * (size_t)numStates + (size_t)s] =
            (ss == 0xFF) ? 1.0 : ((s == (int)ss) ? 1.0 : 0.0);
}

void launchBuildTipPartials(
        const uint8_t* d_tip8,
        double* d_l,
        int numStates,
        int numPatterns,
        int blockSize,
        cudaStream_t stream
) {
    dim3 grid(numPatterns, (numStates + blockSize - 1) / blockSize);
    dim3 block(blockSize);

    build_tip_partials_kernel<<<grid, block, 0, stream>>>(
            d_tip8,
            d_l,
            numStates,
            numPatterns
    );
}