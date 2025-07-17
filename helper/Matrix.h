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

private:
    size_t m_rows, m_cols;
    double* m_data;
    static MatrixOpType m_opType;
public:
    static void setMOpType(MatrixOpType mOpType);
    // Default to CPU operations

};

std::ostream& operator<<(std::ostream& os, const Matrix& matrix); // replace print with the operator<<

#endif // USE_EIGEN
#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H
