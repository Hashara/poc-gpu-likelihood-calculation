//
// Created by Hashara Kumarasinghe on 8/7/2025.
//


#include "Matrix.h"

Matrix::Matrix(size_t rows, size_t cols)
        : m_rows(rows), m_cols(cols), m_data(nullptr) {
    m_data = new double[rows * cols]();
}

Matrix::~Matrix() {
    delete[] m_data;
}

Matrix::Matrix(const Matrix& other)
        : m_rows(other.m_rows), m_cols(other.m_cols) {
    m_data = new double[m_rows * m_cols];
    std::memcpy(m_data, other.m_data, sizeof(double) * m_rows * m_cols);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] m_data;
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = new double[m_rows * m_cols];
        std::memcpy(m_data, other.m_data, sizeof(double) * m_rows * m_cols);
    }
    return *this;
}

double* Matrix::data() { return m_data; }
const double* Matrix::data() const { return m_data; }

size_t Matrix::rows() const { return m_rows; }
size_t Matrix::cols() const { return m_cols; }

double& Matrix::at(size_t i, size_t j) {
    return m_data[j * m_rows + i];  // column-major access
}

const double& Matrix::at(size_t i, size_t j) const {
    return m_data[j * m_rows + i];
}

void Matrix::fill(double val) {
    std::fill(m_data, m_data + m_rows * m_cols, val);
}

void Matrix::print() const {
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            std::cout << at(i, j) << " ";
        }
        std::cout << "\n";
    }
}

void Matrix::fillRandom(unsigned int seed) {
    std::mt19937 gen(seed);                        // Seeded RNG
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] = dis(gen);
    }
}

Matrix Matrix::operator*(const Matrix &other) const {
    if (m_cols != other.m_rows)
        throw std::invalid_argument("Incompatible dimensions for multiplication.");

    Matrix result(m_rows, other.m_cols);

    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < other.m_cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < m_cols; ++k) {
                sum += at(i, k) * other.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix &other) const {
    if (m_rows != other.m_rows || m_cols != other.m_cols)
        throw std::invalid_argument("Incompatible dimensions for Hadamard product.");

    Matrix result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows * m_cols; ++i)
        result.data()[i] = m_data[i] * other.data()[i];  // flat access
    return result;
}
