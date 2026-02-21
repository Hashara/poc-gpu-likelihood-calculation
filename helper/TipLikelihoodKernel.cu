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
    int p = (int)(blockIdx.x * blockDim.x + threadIdx.x); // pattern index
    if (p >= numPatterns) return;

    uint8_t ss = tip8[p];
    size_t base = (size_t)p * (size_t)numStates;

    if (ss == 0xFF) {
        // Ambiguous/missing state: all ones.
        for (int s = 0; s < numStates; ++s) {
            l[base + (size_t)s] = 1.0;
        }
        return;
    }

    // One-hot state vector.
    for (int s = 0; s < numStates; ++s) {
        l[base + (size_t)s] = 0.0;
    }
    if ((int)ss < numStates) {
        l[base + (size_t)ss] = 1.0;
    }
}

void launchBuildTipPartials(
        const uint8_t* d_tip8,
        double* d_l,
        int numStates,
        int numPatterns,
        int blockSize,
        cudaStream_t stream
) {
    dim3 block(blockSize);
    dim3 grid((numPatterns + blockSize - 1) / blockSize);

    build_tip_partials_kernel<<<grid, block, 0, stream>>>(
            d_tip8,
            d_l,
            numStates,
            numPatterns
    );
}
