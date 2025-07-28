//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#include "MatrixOpOpenMPGPU.h"

Matrix MatrixOpOpenMPGPU::hadamard(const Matrix &A, const Matrix &B) {
    size_t M = A.rows(), N = A.cols();
    size_t Asz = M * N;

    if (M != B.rows() || N != B.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
    }

    Matrix C(M, N);
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

#pragma omp target data map(to: a[0:Asz], b[0:Asz]) map(from: c[0:Asz])
    {
#pragma omp target teams distribute parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c[i * N + j] = a[i * N + j] * b[i * N + j];  // row-major layout
            }
        }
    }
    return C;
}

Matrix MatrixOpOpenMPGPU::multiply(const Matrix &A, const Matrix &B) {
    size_t M = A.rows(), N = A.cols(), P = B.cols();
    size_t Asz = M * N, Bsz = N * P, Csz = M * P;

    if (N != B.rows())
        throw std::invalid_argument("Matrix dimensions do not match");

    Matrix C(M, P);

    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();

#pragma omp target data map(to: a[0:Asz], b[0:Bsz]) map(from: c[0:Csz])
    {
#pragma omp target teams distribute parallel for collapse(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < N; ++k) {
                    sum += a[i * N + k] * b[k * P + j];
                }
                c[i * P + j] = sum;
            }
        }
    }

    return C;
}
