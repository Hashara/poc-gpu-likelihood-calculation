//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MODELJC_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MODELJC_H

#include "Model.h"

class ModelJC : public Model {
public:
    ModelJC();

    std::string getName() const override;

    Matrix getRateMatrix() const override;
    Matrix getBaseFrequencies() const override;

    // Transition probability matrix P(t)
    Matrix getTransitionMatrix(double t) const override;

#ifdef USE_EIGEN
    Matrix getExpDiagMatrix(double t) const override;

    void decomposeRateMatrix();
#endif
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODELJC_H
