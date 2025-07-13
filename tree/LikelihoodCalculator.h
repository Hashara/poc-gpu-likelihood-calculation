//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_LIKELIHOODCALCULATOR_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_LIKELIHOODCALCULATOR_H


#include "Node.h"
#include "../helper/Matrix.h"
#include "../helper/MatrixOp.h"
#include "Tree.h"
#include "../alignment/Alignment.h"
#include "../model/Model.h"
#include "../helper/MatrixOpFactory.h"

class LikelihoodCalculator {

public:
    LikelihoodCalculator(Tree* tree, Alignment* aln, Model* model, MatrixOp* matrixOp);

    // Traverse the tree in post-order and compute partial likelihoods
    void traverseAndCompute(Node* node);

    Matrix buildTipLikelihood(const std::string& taxonName);

    // Compute one-hot likelihood matrix for a tip (leaf) node
    void computeTipLikelihood(Node* node);

    // Compute partial likelihood for an internal node using matrix ops
    void computeInternalLikelihood(Node* node);

    // Compute the overall log-likelihood of the tree
    double computeLogLikelihood();

private:
    Tree* tree_;
    Alignment* aln_;
    Model* model_;
    MatrixOp* matrixOp_;
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_LIKELIHOODCALCULATOR_H
