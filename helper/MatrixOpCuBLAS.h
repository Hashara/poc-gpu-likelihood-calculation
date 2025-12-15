//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H

#include "MatrixOp.h"
#include <cublas_v2.h>

class MatrixOpCuBLAS : public MatrixOp {
public:
    MatrixOpCuBLAS();
    ~MatrixOpCuBLAS() override;
    Matrix multiply(const Matrix& A, const Matrix& B) override;
    Matrix hadamard(const Matrix& A, const Matrix& B) override;

#ifdef USE_CUBLAS
    void compositehadamard(const Matrix& A, const Matrix& B, const Matrix& C, const Matrix& D, Matrix& R, uint8_t* scale_count) override;
    void multiplyInPlace(const Matrix& A, const Matrix& B, Matrix& R) override;
#endif // USE_CUBLAS
private:
    cublasHandle_t handle;
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H
