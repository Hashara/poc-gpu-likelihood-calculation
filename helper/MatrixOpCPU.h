//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCPU_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCPU_H


#include "MatrixOp.h"

class MatrixOpCPU : public MatrixOp {
public:
    Matrix multiply(const Matrix& A, const Matrix& B) override;
    Matrix hadamard(const Matrix& A, const Matrix& B) override;
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCPU_H
