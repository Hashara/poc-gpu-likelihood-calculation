//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#include "MatrixOpFactory.h"
#include "MatrixOpCPU.h"

std::unique_ptr<MatrixOp> MatrixOpFactory::create(MatrixOpType type) {
    switch (type) {
        case MatrixOpType::CPU:
            return std::make_unique<MatrixOpCPU>();
        default:
            throw std::invalid_argument("Unsupported MatrixOpType");
    }
}
