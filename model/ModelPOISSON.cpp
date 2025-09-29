//
// Created by Hashara Kumarasinghe on 29/9/2025.
//

#include "ModelPOISSON.h"

#include <cmath>
#include <iostream>


std::string ModelPOISSON::getName() const {
    return "POISSON";
}

Matrix ModelPOISSON::getRateMatrix() const {
    return rateMatrix_;
}

Matrix ModelPOISSON::getBaseFrequencies() const {
    return baseFreq_;
}

#ifdef USE_EIGEN
ModelPOISSON::ModelPOISSON() {
    int n = 20;
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

    baseFreq_.fill(0.05); // Equal base frequencies

#ifdef DECOMP
    decomposeRateMatrix();

#endif
}

Matrix ModelPOISSON::getTransitionMatrix(double t) const {
#ifdef DECOMP
    return eigenvectors * getExpDiagMatrix(t) * inv_eigenvectors;
#else
    double e = std::exp(-20.0 * t / 19.0);
    Matrix P(20, 20);

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            if (i == j)
                P(i, j) = 0.05 + 0.95 * e;
            else
                P(i, j) = 0.05 - 0.05 * e;
        }
    }
    return P;
#endif
}

#ifdef DECOMP
Matrix ModelPOISSON::getExpDiagMatrix(double t) const {
    Eigen::Vector expLambda = (eigenvalues.array() * t).exp();
    return expLambda.asDiagonal();
}

void ModelPOISSON::decomposeRateMatrix() {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(rateMatrix_.toEigenMatrix());
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed for POISSON model.");
    }

    eigenvalues = solver.eigenvalues().real();
    eigenvectors = solver.eigenvectors().real();
    inv_eigenvectors = eigenvectors.inverse();

#ifdef VERBOSE
    std::cout << "Eigenvalues:\n" << eigenvalues.transpose() << std::endl;
    std::cout << "Eigenvectors (U):\n" << eigenvectors << std::endl;
    std::cout << "Inverse Eigenvectors (U⁻¹):\n" << inv_eigenvectors << std::endl;

#endif
}

#endif // DECOMP
#else
ModelPOISSON::ModelPOISSON() {
    int n = 20;
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

    baseFreq_.fill(0.05); // Equal base frequencies
}

Matrix ModelPOISSON::getTransitionMatrix(double t) const {
    double e = std::exp(-20.0 * t / 19.0);
    Matrix P(20, 20);

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            if (i == j)
                P(i, j) = 0.05 + 0.95 * e;
            else
                P(i, j) = 0.05 - 0.05 * e;
        }
    }
    return P;
}

void ModelPOISSON::buildTransitionMatrix(double t, Matrix &P) const {
    double e = std::exp(-20.0 * t / 19.0);

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            if (i == j)
                P(i, j) = 0.05 + 0.95 * e;
            else
                P(i, j) = 0.05 - 0.05 * e;
        }
    }
#ifdef USE_OPENACC
    double* p = P.data();
    #pragma acc enter data copyin(p[0:400])
#endif
}

#endif // USE_EIGEN




