//
// Created by Hashara Kumarasinghe on 14/7/2025.
//

// EigenUtils.cpp
#include "EigenUtils.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace Eigen;

void computeEigenDecomposition(const ::Matrix& Qmatrix,
                               double* eigenvalues,
                               double* eigenvectors,
                               double* inv_eigenvectors,
                               double* inv_eigenvectors_transposed) {
    // Convert custom Matrix to Eigen::Matrix4d
    Matrix4d Q;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Q(i, j) = Qmatrix.at(i, j);

    // Compute Eigen decomposition
    EigenSolver<Matrix4d> solver(Q);

    Vector4cd evals = solver.eigenvalues();
    Matrix4cd evecs = solver.eigenvectors();
    Matrix4cd invEvecs = evecs.inverse();

    // Fill outputs
    for (int i = 0; i < 4; ++i)
        eigenvalues[i] = evals(i).real();

    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            double val = evecs(row, col).real();
            eigenvectors[row * 4 + col] = val;
            val = invEvecs(row, col).real();
            inv_eigenvectors[row * 4 + col] = val;
            inv_eigenvectors_transposed[col * 4 + row] = val;  // transposed
        }
    }
}
