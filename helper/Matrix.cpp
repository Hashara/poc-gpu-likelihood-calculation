//
// Created by Hashara Kumarasinghe on 8/7/2025.
//


#include "Matrix.h"

#ifdef USE_EIGEN
#include <Eigen/Dense>
using Matrix = Eigen::MatrixXd;


Matrix hadamard(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Hadamard: matrix size mismatch.");
    }
    return A.array() * B.array();
}

#else
#include "MatrixOpDispatcher.h"

// set default MatrixOpType
MatrixOpType Matrix::m_opType = MatrixOpType::CPU;

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

void Matrix::fill(double val) {
    std::fill(m_data, m_data + m_rows * m_cols, val);
}


void Matrix::fillRandom(unsigned int seed) {
    std::mt19937 gen(seed);                        // Seeded RNG
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < m_rows * m_cols; ++i) {
        m_data[i] = dis(gen);
    }
}

double& Matrix::operator()(size_t i, size_t j) {
    return m_data[i * m_cols + j];
}

const double& Matrix::operator()(size_t i, size_t j) const {
    return m_data[i * m_cols + j];
}

Matrix Matrix::operator*(const Matrix &other) const {
    return getBackend(m_opType)->multiply(*this, other);
}

Matrix Matrix::hadamard(const Matrix &other) const {
    return getBackend(m_opType)->hadamard(*this, other);
}

void Matrix::setMOpType(MatrixOpType mOpType) {
    m_opType = mOpType;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (size_t j = 0; j < matrix.cols(); ++j) {
            os << matrix(i, j) << " ";
        }
        os << "\n";
    }
    return os;
}

#endif