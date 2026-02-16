//
// Created by Hashara Kumarasinghe on 8/7/2025.
//

#include "Matrix.h"

#ifdef USE_EIGEN
#include <Eigen/Dense>
using Matrix = Eigen::MatrixXd;

Matrix hadamard(const Matrix &A, const Matrix &B) {
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
#if defined(USE_CUBLAS)
  freeDevice();
#endif
  delete[] m_data;
}

Matrix::Matrix(const Matrix &other)
    : m_rows(other.m_rows), m_cols(other.m_cols) {
  m_data = new double[m_rows * m_cols];
  std::memcpy(m_data, other.m_data, sizeof(double) * m_rows * m_cols);
}

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    delete[] m_data;
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_data = new double[m_rows * m_cols];
    std::memcpy(m_data, other.m_data, sizeof(double) * m_rows * m_cols);
  }
  return *this;
}

double *Matrix::data() { return m_data; }
const double *Matrix::data() const { return m_data; }

size_t Matrix::rows() const { return m_rows; }
size_t Matrix::cols() const { return m_cols; }

void Matrix::fill(double val) {
  std::fill(m_data, m_data + m_rows * m_cols, val);
}

void Matrix::fillRandom(unsigned int seed) {
  std::mt19937 gen(seed); // Seeded RNG
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (size_t i = 0; i < m_rows * m_cols; ++i) {
    m_data[i] = dis(gen);
  }
}

double &Matrix::operator()(size_t i, size_t j) {
  return m_data[j * m_rows + i]; // column-major
}

const double &Matrix::operator()(size_t i, size_t j) const {
  return m_data[j * m_rows + i]; // column-major
}

Matrix Matrix::operator*(const Matrix &other) const {
  return getBackend(m_opType)->multiply(*this, other);
}

Matrix Matrix::hadamard(const Matrix &other) const {
  return getBackend(m_opType)->hadamard(*this, other);
}

// New method
void Matrix::resize(size_t rows, size_t cols) {
  if (m_rows == rows && m_cols == cols && m_data)
    return; // already the right size, skip reallocation
  if (m_data) {
    delete[] m_data;
  }
  m_rows = rows;
  m_cols = cols;
  m_data = new double[rows * cols];
}

#if defined(USE_OPENACC) || defined(USE_CUBLAS)
void Matrix::compositeHadamard(const Matrix &B, const Matrix &C,
                               const Matrix &D, Matrix &R,
                               uint8_t *scale_count) const {
  return getBackend(m_opType)->compositehadamard(*this, B, C, D, R,
                                                 scale_count);
}

void Matrix::multiplyInPlace(const Matrix &B, Matrix &R) const {
  return getBackend(m_opType)->multiplyInPlace(*this, B, R);
}
#endif

#if defined(USE_CUBLAS)

void Matrix::allocDevice() {
  const size_t elems = m_rows * m_cols;
  if (d_data && d_elems == elems)
    return;

  // if size changed, reallocate
  if (d_data) {
    cudaFree(d_data);
    d_data = nullptr;
  }
  d_elems = elems;

  cudaMalloc(&d_data, d_elems * sizeof(double));

  if (!h2d_event) {
    cudaEventCreateWithFlags(&h2d_event, cudaEventDisableTiming);
  }
}

void Matrix::copyHtoDAsync(cudaStream_t stream) {
  allocDevice();
  cudaMemcpyAsync(d_data, m_data, d_elems * sizeof(double),
                  cudaMemcpyHostToDevice, stream);
  cudaEventRecord(h2d_event, stream);
}

void Matrix::waitHtoD(cudaStream_t stream) const {
  if (h2d_event)
    cudaStreamWaitEvent(stream, h2d_event, 0);
}

double *Matrix::deviceData() const { return d_data; }

void Matrix::freeDevice() {
  if (d_data)
    cudaFree(d_data);
  d_data = nullptr;
  d_elems = 0;

  if (h2d_event)
    cudaEventDestroy(h2d_event);
  h2d_event = nullptr;
}

void Matrix::copyDtoHAsync(cudaStream_t stream) {
  if (!d_data)
    return; // or allocDevice(), but usually you expect device to exist
  cudaMemcpyAsync(m_data, d_data, d_elems * sizeof(double),
                  cudaMemcpyDeviceToHost, stream);
  if (!d2h_event) {
    cudaEventCreateWithFlags(&d2h_event, cudaEventDisableTiming);
  }
  cudaEventRecord(d2h_event, stream);
}

void Matrix::waitDtoH(cudaStream_t stream) const {
  if (d2h_event)
    cudaStreamWaitEvent(stream, d2h_event, 0);
}

#endif

void Matrix::setMOpType(MatrixOpType mOpType) { m_opType = mOpType; }

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
  for (size_t i = 0; i < matrix.rows(); ++i) {
    for (size_t j = 0; j < matrix.cols(); ++j) {
      os << matrix(i, j) << " ";
    }
    os << "\n";
  }
  return os;
}

#endif