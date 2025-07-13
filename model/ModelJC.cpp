//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "ModelJC.h"
#include <cmath>

ModelJC::ModelJC() : rateMatrix_(4, 4),
                     baseFreq_(1, 4) {
    int n = 4;

    rateMatrix_.fill(1.0);
// Zero diagonal and compute row sums
    for (int i = 0; i < n; ++i) {
        rateMatrix_.at(i, i) = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j)
                rateMatrix_.at(i, i) -= rateMatrix_.at(i, j);
        }
    }

    baseFreq_.fill(.25); // Equal base frequencies
}

std::string ModelJC::getName() const {
    return "JC69";
}

Matrix ModelJC::getRateMatrix() const {
    return rateMatrix_;
}

Matrix ModelJC::getBaseFrequencies() const {
    return baseFreq_;
}

// t is branch length
Matrix ModelJC::getTransitionMatrix(double t) const {
    double e = std::exp(-4.0 * t / 3.0);
    Matrix P(4, 4);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                P.at(i, j) = 0.25 + 0.75 * e;
            else
                P.at(i, j) = 0.25 - 0.25 * e;
        }
    }
    return P;
}