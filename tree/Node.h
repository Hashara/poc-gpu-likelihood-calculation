//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_NODE_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_NODE_H


#include <string>
#include <vector>
#include "../helper/Matrix.h"
#include <unordered_set>

class Node {
public:
    std::string name;
    double branchLength = 0.0;
    Node* parent = nullptr;
    std::vector<Node*> children;

    // Likelihood matrix (states x patterns)
    Matrix partialLikelihood;
    bool isPartialLikelihoodCalculated = false;
    std::unordered_set<int> completedPackets; // Tracks completed packet IDs for chunked computation

    bool isLeaf() const {
        return children.empty();
    }
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_NODE_H
