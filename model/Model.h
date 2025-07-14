//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H

#include <vector>
#include <string>
#include "../helper/Matrix.h"

class Model {
public:
    virtual ~Model() = default;

    virtual std::string getName() const = 0;

    virtual Matrix getBaseFrequencies() const = 0;

    // Transition probability matrix P(t) for a given branch length t
    virtual Matrix getTransitionMatrix(double t) const = 0;


    // 4x4 substitution rate matrix for DNA
    virtual Matrix getRateMatrix() const = 0;
    // Eigen components shared by all models
    double* getEigenvalues() const;

    double* getEigenvectors() const;

    double* getInvEigenvectors() const;

    double* getInvEigenvectorsTransposed() const;

protected:
    Matrix rateMatrix_;
    Matrix baseFreq_;

    double* eigenvalues = nullptr;
    double* eigenvectors = nullptr;
    double* inv_eigenvectors = nullptr;
    double* inv_eigenvectors_transposed = nullptr;
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
