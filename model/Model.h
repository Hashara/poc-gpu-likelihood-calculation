//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H

#include <vector>
#include <string>
#include "../helper/Matrix.h"
//#define DECOMP 1

class Model {
public:
    virtual ~Model() = default;

    virtual std::string getName() const = 0;

    virtual Matrix getBaseFrequencies() const = 0;

    // Transition probability matrix P(t) for a given branch length t
    virtual Matrix getTransitionMatrix(double t) const = 0;


    // 4x4 substitution rate matrix for DNA
    virtual Matrix getRateMatrix() const = 0;

#if defined(USE_EIGEN) && defined(DECOMP)
    // Eigen components shared by all models
    const Matrix &getEigenvalues() const;
    const Matrix &getEigenvectors() const;
    const Matrix &getInvEigenvectors() const;
    virtual Matrix getExpDiagMatrix(double t) const = 0;
#endif

protected:
    Matrix rateMatrix_;
    Matrix baseFreq_;

#if defined(USE_EIGEN) && defined(DECOMP)
    Eigen::VectorXd eigenvalues;
    Matrix eigenvectors ;
    Matrix inv_eigenvectors ;
#endif
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
