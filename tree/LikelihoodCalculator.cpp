//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "LikelihoodCalculator.h"
#define VERBOSE 1

LikelihoodCalculator::LikelihoodCalculator(Tree *tree, Alignment *aln, Model *model, MatrixOp *matrixOp) {

    tree_ = tree;
    aln_ = aln;
    model_ = model;
    matrixOp_ = matrixOp;

}

Matrix LikelihoodCalculator::buildTipLikelihood(const std::string& taxonName) {
    int numStates = 4;
    int numPatterns = aln_->patterns.size();

    int taxonIndex = -1;
    for (size_t i = 0; i < aln_->seq_names.size(); ++i) {
        if (aln_->seq_names[i] == taxonName) {
            taxonIndex = static_cast<int>(i);
            break;
        }
    }

    if (taxonIndex == -1) {
        throw std::runtime_error("Taxon name not found in alignment: " + taxonName);
    }

    Matrix L(numStates, numPatterns);
    for (int p = 0; p < numPatterns; ++p) {
        int state = (*aln_).patterns[p][taxonIndex] - '0';
        for (int s = 0; s < numStates; ++s) {
            L.at(s, p) = (s == state) ? 1.0 : 0.0;
        }
    }

#ifdef VERBOSE
    cout << "Tip likelihood for " << taxonName << ":\n";

    L.print();
#endif
    return L;
}


void LikelihoodCalculator::computeTipLikelihood(Node* node) {
    if (!node->isLeaf()) return;

    node->partialLikelihood = buildTipLikelihood(node->name);
    node->isPartialLikelihoodCalculated = true;
}


void LikelihoodCalculator::computeInternalLikelihood(Node* node) {
    if (node->isLeaf()) return;

    Node* left = node->children[0];
    Node* right = node->children[1];

    const Matrix& L1 = left->partialLikelihood;
    const Matrix& L2 = right->partialLikelihood;

    Matrix P1 = model_->getTransitionMatrix(left->branchLength);
    Matrix P2 = model_->getTransitionMatrix(right->branchLength);

    Matrix PL1 = matrixOp_->multiply(P1, L1);
    Matrix PL2 = matrixOp_->multiply(P2, L2);

    node->partialLikelihood =matrixOp_->hadamard(PL1, PL2);
    node->isPartialLikelihoodCalculated = true;

#ifdef VERBOSE
    cout << "Internal node " << node->name << ":\n";
    cout << "Left child (" << left->name << ") partial likelihood:\n";
    left->partialLikelihood.print();
    cout << "Right child (" << right->name << ") partial likelihood:\n";
    right->partialLikelihood.print();
    cout << "Transition matrix P1 for left child:\n";
    P1.print();
    cout << "Transition matrix P2 for right child:\n";
    P2.print();
    cout << "Partial likelihood after multiplication with transition matrices:\n";
    PL1.print();
    PL2.print();
    cout << "Hadamard product of left and right child likelihoods:\n";
    node->partialLikelihood.print();

    cout << "Internal likelihood for node " << node->name << ":\n";
    node->partialLikelihood.print();
#endif
}


void LikelihoodCalculator::traverseAndCompute(Node* node) {
    // Skip if already computed
    if (node->isPartialLikelihoodCalculated) return;

    // Recurse through children first (post-order)
    for (Node* child : node->children) {
        traverseAndCompute(child);
    }

    // Compute based on node type
    if (node->isLeaf()) {
        computeTipLikelihood(node);
    } else {
        computeInternalLikelihood(node);
    }
}

double LikelihoodCalculator::computeLogLikelihood() {
    // Traverse and compute partial likelihoods for all nodes
    traverseAndCompute(tree_->root);

    const Matrix& rootL = tree_->root->partialLikelihood;
    int numStates = rootL.rows();
    int numPatterns = rootL.cols();

    double logL = 0.0;

    Matrix baseFrequencies = model_->getBaseFrequencies();
    Matrix siteLikelihoods = matrixOp_->multiply(baseFrequencies, rootL);

    for (int j = 0; j < numPatterns; ++j) {
        double siteLikelihood = siteLikelihoods.at(0, j);  // Assuming siteLikelihoods is a 1-row matrix

        // Multiply by the pattern frequency
        int freq = aln_->patterns[j].frequency;
        logL += freq * std::log(siteLikelihood);
    }

    return logL;
}


