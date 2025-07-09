//
// Created by Hashara Kumarasinghe on 9/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXUTILS_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXUTILS_H



#include "Matrix.h"
#include "MatrixOp.h"

/**
 * @brief Compute (A * B) ∘ (C * D) using the given MatrixOp.
 *
 * @param op   The backend (CPU, OpenACC, CUDA, etc.) to use.
 * @param A, B Matrices to multiply for the first term.
 * @param C, D Matrices to multiply for the second term.
 * @return    The hadamard (element-wise) product of (A*B) and (C*D).
 */
inline Matrix compositeHadamard(MatrixOp &op,
                                const Matrix &A, const Matrix &B,
                                const Matrix &C, const Matrix &D)
{
    Matrix AB = op.multiply(A, B);
    Matrix CD = op.multiply(C, D);
    return op.hadamard(AB, CD);
}

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXUTILS_H
