//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOP_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOP_H

#include "Matrix.h"

#ifndef USE_EIGEN

class MatrixOp {
public:
    virtual Matrix multiply(const Matrix& A, const Matrix& B) = 0;
    virtual Matrix hadamard(const Matrix& A, const Matrix& B) = 0;
    virtual ~MatrixOp() = default;
};

#endif // USE_EIGEN
#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOP_H
