//
// Created by Hashara Kumarasinghe on 9/7/2025.
//

#include "MatrixOpOpenACC.h"
#include <nvtx3/nvToolsExt.h>

Matrix MatrixOpOpenACC::multiply(const Matrix &A, const Matrix &B) {
    nvtxRangePushA("MatrixOpOpenACC::multiply");
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
#pragma acc loop vector reduction(+:sum)
                for (size_t k = 0; k < N; ++k) {
                    sum += a[i*N + k] * b[k*P + j];
                }
                c[i*P + j] = sum;
            }
        }
    }

    nvtxRangePop();
    return C;
}


Matrix MatrixOpOpenACC::hadamard(const Matrix &A, const Matrix &B) {
    nvtxRangePushA("MatrixOpOpenACC::hadamard");
    size_t M = A.rows(), N = A.cols();
    size_t Asz = M * N;

    if (M != B.rows() || N != B.cols()) {
        throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
    }

    Matrix C(M, N);
    const double *a = A.data();
    const double *b = B.data();
    double *c = C.data();


    #pragma acc data copyin(a[0:Asz], b[0:Asz]) copyout(c[0:Asz])
    {
        #pragma acc parallel loop collapse(2) gang vector
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                c[i*N + j] = a[i*N + j] * b[i*N + j];  // Row-major
            }
        }
    }
    nvtxRangePop();
    return C;
}


void MatrixOpOpenACC::compositehadamard(const Matrix &A, const Matrix &B,
                                          const Matrix &C, const Matrix &D, Matrix &R) {
    nvtxRangePushA("MatrixOpOpenACC::compositehadamard");

    size_t M = A.rows(), N = A.cols(), P = B.cols();

    Matrix R1(M, P);
    Matrix R2(M, P);
    R.resize(M, P);

    const double *a = A.data();
    const double *b = B.data();
    const double *c = C.data();
    const double *d = D.data();
    double *r1 = R1.data();
    double *r2 = R2.data();
    double *r = R.data();

    size_t Asz = M * N;
    size_t Bsz = N * P;
    size_t Csz = M * P;

    // One data region for all matrices
#pragma acc data copyin(a[0:Asz], b[0:Bsz], c[0:Asz], d[0:Bsz]) \
                     create(r1[0:Csz], r2[0:Csz]) copyout(r[0:Csz])
    {
        // First multiplication (async queue 1)
#pragma acc parallel loop collapse(2) gang vector async(1)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                double sum = 0.0;
#pragma acc loop reduction(+:sum)
                for (size_t k = 0; k < N; ++k) {
                    sum += a[i*N + k] * b[k*P + j];
                }
                r1[i*P + j] = sum;
            }
        }

        // Second multiplication (async queue 2)
#pragma acc parallel loop collapse(2) gang vector async(2)
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                double sum = 0.0;
#pragma acc loop reduction(+:sum)
                for (size_t k = 0; k < N; ++k) {
                    sum += c[i*N + k] * d[k*P + j];
                }
                r2[i*P + j] = sum;
            }
        }

        // Wait for both multiplies to finish
#pragma acc wait(1,2)

        // Hadamard product
#pragma acc parallel loop collapse(2) gang vector
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                r[i*P + j] = r1[i*P + j] * r2[i*P + j];
            }
        }
    }

    nvtxRangePop();
//    return R;
}

