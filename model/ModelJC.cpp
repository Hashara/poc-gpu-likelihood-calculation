//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "ModelJC.h"
#include <cmath>

ModelJC::ModelJC() {
    int n = 4;
    rateMatrix_.resize(n, std::vector<double>(n, 1.0));
    for (int i = 0; i < n; ++i) {
        rateMatrix_[i][i] = 0.0;
        for (int j = 0; j < n; ++j)
            if (i != j)
                rateMatrix_[i][i] -= rateMatrix_[i][j];
    }

    baseFreq_ = std::vector<double>(4, 0.25); // Equal base frequencies
}

std::string ModelJC::getName() const {
    return "JC69";
}

std::vector<std::vector<double>> ModelJC::getRateMatrix() const {
    return rateMatrix_;
}

std::vector<double> ModelJC::getBaseFrequencies() const {
    return baseFreq_;
}

// t is branch length
std::vector<std::vector<double>> ModelJC::getTransitionMatrix(double t) const {
    double e = std::exp(-4.0 * t / 3.0);
    std::vector<std::vector<double>> P(4, std::vector<double>(4));

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j)
                P[i][j] = 0.25 + 0.75 * e;
            else
                P[i][j] = 0.25 - 0.25 * e;
        }
    }
    return P;
}
