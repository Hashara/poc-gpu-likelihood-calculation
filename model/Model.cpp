//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "Model.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstring>

#if defined(USE_EIGEN) && defined(DECOMP)
const Matrix &Model::getEigenvalues() const {
    return eigenvalues;
}

const Matrix &Model::getEigenvectors() const {
    return eigenvectors;
}

const Matrix &Model::getInvEigenvectors() const {
    return inv_eigenvectors;
}

#endif

// =========================================================================
// Eigendecomposition support (no Eigen library dependency)
// =========================================================================

int Model::getNumStates() const {
    return numStates_;
}

const double* Model::getEigenvaluesFlat() const {
    return eigenvalues_.data();
}

const double* Model::getEigenvectorsFlat() const {
    return eigenvectors_.data();
}

const double* Model::getInvEigenvectorsFlat() const {
    return invEigenvectors_.data();
}

/**
 * Build echildren matrix: E(t) = U · diag(exp(λ·t))
 *
 * E[i,j] = U[i,j] * exp(λ[j] * t)
 *
 * Result is K×K column-major (same as U), written into E.
 * This is the "echildren" matrix used in IQ-TREE's eigenspace formulation.
 */
void Model::buildEChildren(double t, Matrix &E) const {
    int K = numStates_;
    E.resize(K, K);
    double *e = E.data();
    const double *u = eigenvectors_.data();

    // Compute exp(λ[j] * t) for each eigenvalue
    for (int j = 0; j < K; ++j) {
        double expLambda = std::exp(eigenvalues_[j] * t);
        // Column j of E = Column j of U * exp(λ[j]*t)
        for (int i = 0; i < K; ++i) {
            e[j * K + i] = u[j * K + i] * expLambda;
        }
    }
}

/**
 * Jacobi eigenvalue algorithm for real symmetric matrices.
 *
 * Both JC and POISSON rate matrices are symmetric (equal rates model),
 * so this simple O(n³) algorithm works perfectly without external dependencies.
 *
 * Input:  A[n*n] symmetric matrix (column-major), destroyed on output
 * Output: eigenvals[n] eigenvalues (unsorted)
 *         V[n*n] eigenvectors as columns (column-major)
 */
static void jacobiEigen(double *A, int n, double *eigenvals, double *V) {
    // Initialize V = Identity
    std::memset(V, 0, n * n * sizeof(double));
    for (int i = 0; i < n; ++i)
        V[i * n + i] = 1.0;

    const int maxIter = 100;
    const double tol = 1e-15;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Find largest off-diagonal element
        double maxOffDiag = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double aij = std::fabs(A[j * n + i]); // column-major: A(i,j) = A[j*n+i]
                if (aij > maxOffDiag) {
                    maxOffDiag = aij;
                    p = i;
                    q = j;
                }
            }
        }

        if (maxOffDiag < tol)
            break;

        // Compute rotation angle
        double app = A[p * n + p]; // A(p,p)
        double aqq = A[q * n + q]; // A(q,q)
        double apq = A[q * n + p]; // A(p,q)

        double theta;
        if (std::fabs(app - aqq) < 1e-30) {
            theta = M_PI / 4.0;
        } else {
            theta = 0.5 * std::atan2(2.0 * apq, app - aqq);
        }

        double c = std::cos(theta);
        double s = std::sin(theta);

        // Apply Jacobi rotation: A' = J^T A J
        // Update rows/cols p and q
        for (int i = 0; i < n; ++i) {
            if (i == p || i == q) continue;
            double aip = A[p * n + i]; // A(i,p)
            double aiq = A[q * n + i]; // A(i,q)
            A[p * n + i] = c * aip + s * aiq;
            A[q * n + i] = -s * aip + c * aiq;
            // Symmetric: A(p,i) = A(i,p), A(q,i) = A(i,q)
            A[i * n + p] = A[p * n + i];
            A[i * n + q] = A[q * n + i];
        }

        // Update diagonal and off-diagonal (p,q)
        A[p * n + p] = c * c * app + 2 * s * c * apq + s * s * aqq;
        A[q * n + q] = s * s * app - 2 * s * c * apq + c * c * aqq;
        A[q * n + p] = 0.0;
        A[p * n + q] = 0.0;

        // Update eigenvectors: V' = V * J
        for (int i = 0; i < n; ++i) {
            double vip = V[p * n + i]; // V(i,p)
            double viq = V[q * n + i]; // V(i,q)
            V[p * n + i] = c * vip + s * viq;
            V[q * n + i] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues from diagonal
    for (int i = 0; i < n; ++i) {
        eigenvals[i] = A[i * n + i];
    }
}

/**
 * Simple matrix inverse for small matrices (Gauss-Jordan elimination).
 * A[n*n] column-major input, inv[n*n] column-major output.
 */
static void invertMatrix(const double *A, int n, double *inv) {
    // Augmented matrix [A | I] stored row-major for convenience
    std::vector<double> aug(n * 2 * n, 0.0);

    // Fill augmented matrix (convert column-major A to row-major aug)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i * 2 * n + j] = A[j * n + i]; // A(i,j) col-major → aug row-major
        }
        aug[i * 2 * n + n + i] = 1.0; // Identity on the right
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int maxRow = col;
        double maxVal = std::fabs(aug[col * 2 * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double v = std::fabs(aug[row * 2 * n + col]);
            if (v > maxVal) {
                maxVal = v;
                maxRow = row;
            }
        }
        if (maxVal < 1e-30) {
            throw std::runtime_error("Matrix is singular, cannot invert.");
        }

        // Swap rows
        if (maxRow != col) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(aug[col * 2 * n + j], aug[maxRow * 2 * n + j]);
            }
        }

        // Scale pivot row
        double pivot = aug[col * 2 * n + col];
        for (int j = 0; j < 2 * n; ++j) {
            aug[col * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug[row * 2 * n + col];
            for (int j = 0; j < 2 * n; ++j) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    // Extract inverse (right half of augmented matrix) → column-major output
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inv[j * n + i] = aug[i * 2 * n + n + j]; // row-major aug → col-major inv
        }
    }
}

void Model::computeEigenDecomp() {
#ifdef USE_EIGEN
    numStates_ = rateMatrix_.rows();
#else
    numStates_ = (int)rateMatrix_.rows();
#endif
    int K = numStates_;

    eigenvalues_.resize(K);
    eigenvectors_.resize(K * K);
    invEigenvectors_.resize(K * K);

    // Copy rate matrix into working buffer (Jacobi destroys input)
    std::vector<double> Acopy(K * K);
#ifdef USE_EIGEN
    // When USE_EIGEN is defined, Matrix = Eigen::MatrixXd
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            Acopy[j * K + i] = rateMatrix_(i, j);
#else
    const double *src = rateMatrix_.data();
    std::memcpy(Acopy.data(), src, K * K * sizeof(double));
#endif

    // Normalize the rate matrix so that the expected substitution rate = 1:
    //   norm = -Σ_i π_i Q(i,i)
    //   Q_normalized = Q / norm
    // This ensures eigenvalues match the branch-length scale used by
    // the analytical P(t) formulas (e.g., exp(-4t/3) for JC).
    double norm = 0.0;
    for (int i = 0; i < K; ++i) {
#ifdef USE_EIGEN
        norm -= baseFreq_(0, i) * Acopy[i * K + i];
#else
        norm -= baseFreq_.data()[i] * Acopy[i * K + i]; // baseFreq_ is 1×K col-major
#endif
    }
    if (norm > 0.0) {
        for (int i = 0; i < K * K; ++i) {
            Acopy[i] /= norm;
        }
    }

    // Compute eigendecomposition
    jacobiEigen(Acopy.data(), K, eigenvalues_.data(), eigenvectors_.data());

    // Compute U⁻¹
    invertMatrix(eigenvectors_.data(), K, invEigenvectors_.data());

    eigenDecompReady_ = true;

#ifdef VERBOSE
    std::cout << "Eigenvalues: ";
    for (int i = 0; i < K; ++i) std::cout << eigenvalues_[i] << " ";
    std::cout << std::endl;
#endif
}
