//
// Created by Hashara Kumarasinghe on 28/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUDA_CUH
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUDA_CUH


#include "MatrixOp.h"

class MatrixOpCUDA : public MatrixOp {
public:
    Matrix multiply(const Matrix& A, const Matrix& B) override;
    Matrix hadamard(const Matrix& A, const Matrix& B) override;
    virtual ~MatrixOpCUDA() override = default;
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUDA_CUH
