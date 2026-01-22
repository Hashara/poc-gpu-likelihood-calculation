//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H




#ifdef USE_EIGEN
#include <Eigen/Dense>
using Matrix = Eigen::MatrixXd;
Matrix hadamard(const Matrix& A, const Matrix& B);

#else
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <random>
#include "MatrixOpType.h"

#ifdef USE_CUBLAS
#include <cuda_runtime.h>
#endif

class Matrix {

public:
    Matrix(size_t rows, size_t cols);
    ~Matrix();
    Matrix() : m_rows(0), m_cols(0), m_data(nullptr) {}

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    double* data();             // Host pointer
    const double* data() const;

    size_t rows() const;
    size_t cols() const;

    // Access using (i, j) notation
    double& operator()(size_t i, size_t j);             // non-const access
    const double& operator()(size_t i, size_t j) const; // const access


    void fill(double val);

    void resize(size_t newRows, size_t newCols); // Resizes the matrix, preserving existing data where possible

    void fillRandom(unsigned int seed = 0);  // Fills matrix with values in [0, 1]

    // use operator loading
    /**
     * Matrix multiplication operator
     * @param other a Matrix
     * @return the product of this matrix and the other matrix
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * Hadamard (element-wise) product.
     * Example usage:
     *  Matrix A={{1,2}, {3,4}};
     *  Matrix B={{5,6}, {7,8}};
     *  Matrix C = A.hadamard(B);
     * @param other A Matrix
     * @return the Hadamard product of this matrix and the other matrix
     */
    Matrix hadamard(const Matrix& other) const;

#if defined(USE_OPENACC) || defined(USE_CUBLAS)
    void compositeHadamard(const Matrix& B, const Matrix& C, const Matrix& D, Matrix& R, uint8_t* scale_count) const;
    void multiplyInPlace(const Matrix& B, Matrix& R) const;
#endif

#if defined(USE_CUBLAS)
    void allocDevice();
    void copyHtoDAsync(cudaStream_t stream);
    void waitHtoD(cudaStream_t stream) const;
    double* deviceData() const;
    void freeDevice();
    void copyDtoHAsync(cudaStream_t stream);
    void waitDtoH(cudaStream_t stream) const;

#endif

private:
    size_t m_rows, m_cols;
    double* m_data;
    static MatrixOpType m_opType;
#if defined(USE_CUBLAS)
    double* d_data = nullptr; // Device pointer
    size_t d_elems = 0;      // Number of elements allocated on device
    cudaEvent_t h2d_event = nullptr; // Event for host-to-device copy
    cudaEvent_t d2h_event = nullptr; // Event for device-to-host copy
#endif
public:
    static void setMOpType(MatrixOpType mOpType);
    // Default to CPU operations

};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix); // replace print with the operator<<

#endif // USE_EIGEN
#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H
