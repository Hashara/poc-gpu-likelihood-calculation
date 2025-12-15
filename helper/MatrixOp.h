//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOP_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOP_H

#include "Matrix.h"
#include "../tree/Scaling.h"
#ifndef USE_EIGEN

class MatrixOp {
public:
    virtual Matrix multiply(const Matrix& A, const Matrix& B) = 0;
    virtual Matrix hadamard(const Matrix& A, const Matrix& B) = 0;
#if defined(USE_OPENACC) || defined(USE_CUBLAS)
    virtual void compositehadamard(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, Matrix& R, uint8_t* scale_count) = 0;
    virtual void multiplyInPlace(const Matrix& A, const Matrix& B, Matrix& R) = 0;
#endif // USE_OPENACC
    virtual ~MatrixOp() = default;
    inline static int numStates = 0;
    //setter for the numstates
    static void setNumStates(int n) {
        numStates = n;
    }
};

#endif // USE_EIGEN
#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOP_H
