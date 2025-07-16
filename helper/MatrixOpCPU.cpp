//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#include "MatrixOpCPU.h"

Matrix MatrixOpCPU::multiply(const Matrix &A, const Matrix &B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

Matrix MatrixOpCPU::hadamard(const Matrix &A, const Matrix &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
    }

    Matrix C(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            C(i, j) = A(i, j) * B(i, j);
        }
    }
    return C;
}