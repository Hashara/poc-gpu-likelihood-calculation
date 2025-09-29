//
// Created by Hashara Kumarasinghe on 29/9/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MODELPOISSON_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MODELPOISSON_H

#include "Model.h"

class ModelPOISSON : public Model {
public:
    ModelPOISSON();

    std::string getName() const override;

    Matrix getRateMatrix() const override;
    Matrix getBaseFrequencies() const override;

    // Transition probability matrix P(t)
    Matrix getTransitionMatrix(double t) const override;

    void buildTransitionMatrix(double t, Matrix &P) const override;

#if defined(USE_EIGEN) && defined(DECOMP)
    Matrix getExpDiagMatrix(double t) const override;

    void decomposeRateMatrix();
#endif

};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODELPOISSON_H
