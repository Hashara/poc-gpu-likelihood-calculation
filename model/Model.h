//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H

#include <vector>
#include <string>

class Model {
public:
    virtual ~Model() = default;

    virtual std::string getName() const = 0;

    // 4x4 substitution rate matrix for DNA
    virtual std::vector<std::vector<double>> getRateMatrix() const = 0;

    // Optional: equilibrium frequencies, model parameters
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
