//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "ModelJC.h"
#include <cmath>
#include <iostream>
//#define VERBOSE 1

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

    // JC69 rate matrix
//    rateMatrix_ = Matrix::Constant(n, n, 1.0);
//    rateMatrix_.diagonal().setZero();
//    rateMatrix_.diagonal() = -rateMatrix_.rowwise().sum();
    rateMatrix_ = Matrix(4, 4);  // Or rateMatrix_.resize(4, 4);

    rateMatrix_ << -3.0, 1.0, 1.0, 1.0,
            1.0, -3.0, 1.0, 1.0,
            1.0, 1.0, -3.0, 1.0,
            1.0, 1.0, 1.0, -3.0;

    rateMatrix_ /= 3.0;  // Normalize to make the row sums equal to 0
    std::cout << "Rate matrix:\n" << rateMatrix_ << std::endl;

    baseFreq_ = Matrix::Constant(1, n, 0.25);  // π = [0.25, 0.25, 0.25, 0.25]

    decomposeRateMatrix();

}

Matrix ModelJC::getTransitionMatrix(double t) const {
    double e = std::exp(-4.0 * t / 3.0);
    double diag = 0.25 + 0.75 * e;
    double off_diag = 0.25 - 0.25 * e;

    Matrix P = Matrix::Constant(4, 4, off_diag);  // Fill all with off-diagonal
    P.diagonal().setConstant(diag);              // Set diagonal

    return P;
}

Matrix ModelJC::getExpDiagMatrix(double t) const {
    Eigen::VectorXd expLambda = (eigenvalues.array() * t).exp();
#ifdef VERBOSE
    std::cout << "Eigenvalues: " << eigenvalues.transpose() << std::endl;
    std::cout << "Exp(eigenvalues * t): " << expLambda.transpose() << std::endl;
    Matrix EigenP = eigenvectors * expLambda.asDiagonal() * inv_eigenvectors;
    std::cout << "Eigen P(t):\n" << EigenP << std::endl;

    Matrix classicP = getTransitionMatrix(t);
    std::cout << "Classic P(t):\n" << classicP << std::endl;

    double error = (EigenP - classicP).norm();

    std::cout << "‖P_eigen - P_direct‖ = " << error << std::endl;
    if (error < 1e-10) {
        std::cout << "✅ accurate.\n";
    } else {
        std::cout << "⚠️ may be inaccurate.\n";
    }
#endif
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
//    inv_eigenvectors_transposed = inv_eigenvectors.transpose();
}



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

