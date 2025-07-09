//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPFACTORY_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPFACTORY_H

#include "MatrixOp.h"

enum class MatrixOpType {
    CPU,
    CUBLAS,
    CUDA_KERNEL,
    OPENACC
};

class MatrixOpFactory {
public:
    static std::unique_ptr<MatrixOp> create(MatrixOpType type);

};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPFACTORY_H
