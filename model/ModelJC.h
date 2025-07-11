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

    std::vector<std::vector<double>> getRateMatrix() const override;
    std::vector<double> getBaseFrequencies() const;

    // Transition probability matrix P(t)
    std::vector<std::vector<double>> getTransitionMatrix(double t) const;

private:
    std::vector<std::vector<double>> rateMatrix_;
    std::vector<double> baseFreq_;
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODELJC_H
