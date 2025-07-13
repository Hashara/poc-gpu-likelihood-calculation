//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H


#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <random>

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

    double& at(size_t i, size_t j);             // Access (row, col)
    const double& at(size_t i, size_t j) const;

    void fill(double val);
    void print() const;
    void fillRandom(unsigned int seed = 0);  // Fills matrix with values in [0, 1]
private:
    size_t m_rows, m_cols;
    double* m_data;
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIX_H
