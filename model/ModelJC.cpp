//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "ModelJC.h"
#include <cmath>
#include <iostream>


std::string ModelJC::getName() const {
    return "JC69";
}

Matrix ModelJC::getRateMatrix() const {
    return rateMatrix_;
}

Matrix ModelJC::getBaseFrequencies() const {
    return baseFreq_;
}

#ifdef USE_EIGEN
ModelJC::ModelJC() {
    int n = 4;
    rateMatrix_ = Matrix(n, n);
    baseFreq_ = Matrix(1, n);

    rateMatrix_.fill(1.0);
// Zero diagonal and compute row sums
    for (int i = 0; i < n; ++i) {
        rateMatrix_(i, i) = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j)
                rateMatrix_(i, i) -= rateMatrix_(i, j);
        }
    }

    baseFreq_.fill(.25); // Equal base frequencies

#ifdef DECOMP
    decomposeRateMatrix();
#endif
}

Matrix ModelJC::getTransitionMatrix(double t) const {
#ifdef DECOMP
    return eigenvectors * getExpDiagMatrix(t) * inv_eigenvectors;
#else
    double e = std::exp(-4.0 * t / 3.0);
    Matrix P(4, 4);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                P(i, j) = 0.25 + 0.75 * e;
            else
                P(i, j) = 0.25 - 0.25 * e;
        }
    }
    return P;
#endif
}

#ifdef DECOMP
Matrix ModelJC::getExpDiagMatrix(double t) const {
    Eigen::VectorXd expLambda = (eigenvalues.array() * t).exp();
    return expLambda.asDiagonal();  // returns 4x4 diagonal matrix
}

void ModelJC::decomposeRateMatrix() {
    // rateMatrix_ already defined as Matrix (Eigen::MatrixXd)
    assert(rateMatrix_.rows() == 4 && rateMatrix_.cols() == 4);

    // Use symmetric eigendecomposition
    Eigen::SelfAdjointEigenSolver<Matrix> solver(rateMatrix_);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    eigenvalues = solver.eigenvalues();                   // VectorXd
    eigenvectors = solver.eigenvectors();                 // Matrix (U)
    inv_eigenvectors = eigenvectors.inverse();

#ifdef VERBOSE
    std::cout << "Eigenvalues:\n" << eigenvalues.transpose() << std::endl;
    std::cout << "Eigenvectors (U):\n" << eigenvectors << std::endl;
    std::cout << "Inverse Eigenvectors (U⁻¹):\n" << inv_eigenvectors << std::endl;
#endif
}

#endif // DECOMP

#else
ModelJC::ModelJC() {
    int n = 4;
    rateMatrix_ = Matrix(n, n);
    baseFreq_ = Matrix(1, n);

    rateMatrix_.fill(1.0);
// Zero diagonal and compute row sums
    for (int i = 0; i < n; ++i) {
        rateMatrix_(i, i) = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j)
                rateMatrix_(i, i) -= rateMatrix_(i, j);
        }
    }

    baseFreq_.fill(.25); // Equal base frequencies
}

// t is branch length
Matrix ModelJC::getTransitionMatrix(double t) const {
    double e = std::exp(-4.0 * t / 3.0);
    Matrix P(4, 4);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                P(i, j) = 0.25 + 0.75 * e;
            else
                P(i, j) = 0.25 - 0.25 * e;
        }
    }
    return P;
}
#endif

