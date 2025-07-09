//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#include "MatrixOpFactory.h"
#include "MatrixOpCPU.h"

#ifdef USE_OPENACC
#include "MatrixOpOpenACC.h"
#endif

std::unique_ptr<MatrixOp> MatrixOpFactory::create(MatrixOpType type) {
    switch (type) {
        case MatrixOpType::CPU:
            return std::make_unique<MatrixOpCPU>();
#ifdef USE_OPENACC
        case MatrixOpType::OPENACC:
            return std::make_unique<MatrixOpOpenACC>();
#endif
        default:
            throw std::invalid_argument("Unsupported MatrixOpType");
    }
}
