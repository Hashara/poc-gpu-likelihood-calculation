//
// Created by Hashara Kumarasinghe on 16/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPDISPATCHER_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPDISPATCHER_H

#include "MatrixOp.h"
#include "MatrixOpType.h"
#include "MatrixOpCPU.h"

#ifdef USE_OPENACC
#include "MatrixOpOpenACC.h"
#endif

// Function to return a backend singleton based on MatrixOpType
inline MatrixOp* getBackend(MatrixOpType type) {
    switch (type) {
        case MatrixOpType::CPU: {
            static MatrixOpCPU op;
            return &op;
        }
#ifdef USE_OPENACC
            case MatrixOpType::OPENACC: {
            static MatrixOpOpenACC op;
            return &op;
        }
#endif
        default:
            throw std::invalid_argument("Unsupported MatrixOpType in getBackend");
    }
}

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPDISPATCHER_H
