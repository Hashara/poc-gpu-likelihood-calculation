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
                    // column-major indexing
                    sum += a[k * M + i] * b[j * N + k];
                }
                c[j * M + i] = sum;
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
        for (size_t j = 0; j < N; ++j) {
            for (size_t i = 0; i < M; ++i) {
                // column-major: idx = j*M + i
                size_t idx = j * M + i;
                c[idx] = a[idx] * b[idx];
            }
        }
    }
    nvtxRangePop();
    return C;
}


void MatrixOpOpenACC::compositehadamard(const Matrix &A, const Matrix &B,
                                        const Matrix &C, const Matrix &D, Matrix &R, uint8_t *scale_count) {
    // A, C are transition matrices
    // B, D are likelihood matrices
    nvtxRangePushA("MatrixOpOpenACC::compositehadamard");

    size_t P = B.cols();

    R.resize(numStates, P);

    // https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/#:~:text=Using%20the%20__restrict__,in%20the%20provided%20CPU%20example.
    const double *__restrict__ a = A.data();
    const double *__restrict__ b = B.data();
    const double *__restrict__ c = C.data();
    const double *__restrict__ d = D.data();
    double *__restrict__ r = R.data();

    size_t Asz = numStates * numStates;
    size_t Bsz = numStates * P;

#pragma acc enter data create(r[0:Bsz]) // Pre-create output matrix on device

// Single kernel for both multiplications and Hadamard product
#pragma acc data present(a[0:Asz], b[0:Bsz], c[0:Asz], d[0:Bsz], r[0:Bsz], scale_count[0:P])
    {
        // One kernel, two dot-products, one write

//#pragma acc kernels // doc: https://openacc-best-practices-guide.readthedocs.io/en/latest/04-Parallelize.html#the-kernels-construct
#pragma acc parallel loop gang vector vector_length(128) // doc: https://openacc-best-practices-guide.readthedocs.io/en/latest/06-Loops.html
        for (size_t j = 0; j < P; ++j) {
            const double *__restrict__ local_b = &b[j * numStates];
            const double *__restrict__ local_d = &d[j * numStates];

#pragma acc loop vector
            for (size_t i = 0; i < numStates; ++i) {
                double s1 = 0.0, s2 = 0.0;

#ifdef TRANSPOSED_RATE_MATRIX
                const double *local_a = &a[i*numStates]; // pointer to start of row i in A (transposed)
                const double *local_c = &c[i*numStates]; // pointer to start of row i in C (transposed)

                for (size_t k = 0; k < numStates; ++k) {
                    // column-major indexing
                    s1 += local_a[k] * local_b[k];  // A(i,k) * B(k,j)
                    s2 += local_c[k] * local_d[k];  // C(i,k) * D(k,j)
                }
#else
                for (size_t k = 0; k < numStates; ++k) {
                    // column-major indexing
                    s1 += a[k * numStates + i] * local_b[k];  // A(i,k) * B(k,j)
                    s2 += c[k * numStates + i] * local_d[k];  // C(i,k) * D(k,j)
                }
#endif
                r[j * numStates + i] = s1 * s2;
            }
        }


#pragma acc parallel loop gang vector present(r[0:Bsz], scale_count[0:P])
        for (size_t j = 0; j < P; ++j) {
            double max_abs = 0.0;
#pragma acc loop reduction(max:max_abs)
            for (size_t i = 0; i < numStates; ++i) {
                double v = fabs(r[j * numStates + i]);
                if (v > max_abs) max_abs = v;
            }
            // 2) rescale by 2^256 if below threshold
            if (max_abs < SCALING_THRESHOLD) {
//#pragma acc loop
                for (size_t i = 0; i < numStates; ++i) {
                    // scalbn(x,k) == x * 2^k (device-friendly)
                    r[j * numStates + i] = scalbn(r[j * numStates + i], SCALING_THRESHOLD_EXP);
                }
                scale_count[j] += 1;
            }
        }
    }
#pragma acc exit data delete(a[0:Asz], c[0:Asz]) async
    nvtxRangePop();
//    return R;
}

void MatrixOpOpenACC::multiplyInPlace(const Matrix &A, const Matrix &B, Matrix &R) {
    nvtxRangePushA("MatrixOpOpenACC::multiplyInPlace");
    size_t M = A.rows(), N = A.cols(), P = B.cols();
    size_t Asz = M * N, Bsz = N * P, Csz = M * P;

    if (N != B.rows())
        throw std::invalid_argument("Matrix dimensions do not match");

    R.resize(M, P);

    const double *a = A.data();
    const double *b = B.data();
    double *c = R.data();
#pragma acc enter data create(c[0:Csz]) // Pre-create output matrix on device
#pragma acc data present_or_copyin(a[0:Asz], b[0:Bsz], c[0:Csz])
    {
#pragma acc parallel loop collapse(2) gang vector
        for (size_t j = 0; j < P; ++j) {
            for (size_t i = 0; i < M; ++i) {
                double sum = 0.0;
#pragma acc loop vector reduction(+:sum)
                for (size_t k = 0; k < N; ++k) {
                    // column-major addressing
                    sum += a[k * M + i] * b[j * N + k];   // A(i,k) * B(k,j)
                }
                c[j * M + i] = sum;                     // R(i,j)
            }
        }
#pragma acc exit data delete(a[0:Asz], b[0:Bsz]) async
    }
}

