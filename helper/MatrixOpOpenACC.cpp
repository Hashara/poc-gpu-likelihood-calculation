//
// Created by Hashara Kumarasinghe on 9/7/2025.
//

#include "MatrixOpOpenACC.h"

Matrix MatrixOpOpenACC::multiply(const Matrix &A, const Matrix &B) {
    size_t M = A.rows(), N = A.cols(), P = B.cols();
    size_t Asz = M * N, Bsz = N * P, Csz = M * P;

    if (N != B.rows())
        throw std::invalid_argument("Matrix dimensions do not match");

    Matrix C(M, P);

    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

#pragma acc data copyin(a[0:Asz], b[0:Bsz]) copyout(c[0:Csz])
    {
#pragma acc parallel loop collapse(2) gang vector
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                double sum = 0.0;
#pragma acc loop seq
                for (size_t k = 0; k < N; ++k) {
                    sum += a[i * N + k] * b[k * P + j];
                }
                c[i * P + j] = sum;
            }
        }
    }

    return C;
}


Matrix MatrixOpOpenACC::hadamard(const Matrix &A, const Matrix &B) {
    size_t M = A.rows(), N = A.cols();
    size_t Asz = M * N, Bsz = M * N;

    if (M != B.rows() || N != B.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
    }

    Matrix C(M, N);
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();


#pragma acc data copyin(a[0:Asz], b[0:Bsz]) copyout(c[0:Csz])
    {
        #pragma acc parallel loop collapse(2) gang vector
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                C.at(i, j) = A.at(i, j) * B.at(i, j);
            }
        }
    }
    return C;
}