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

    // Cached buffers for compositehadamard (avoid repeated malloc/free)
    double* d_A_cache = nullptr;   // numStates x numStates transition matrix
    double* d_C_cache = nullptr;   // numStates x numStates transition matrix
    double* d_AB_cache = nullptr;  // numStates x P intermediate result
    double* d_CD_cache = nullptr;  // numStates x P intermediate result
    size_t cache_elems_A = 0;      // Current allocation size for A/C
    size_t cache_elems_B = 0;      // Current allocation size for AB/CD (max P seen)

    // Streams for parallel GEMM execution
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;

    // Helper to ensure buffer is large enough
    void ensureBuffer(double*& buf, size_t& current_elems, size_t needed_elems);
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPCUBLAS_H
