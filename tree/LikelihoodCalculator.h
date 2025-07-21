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

class LikelihoodCalculator {

public:
    LikelihoodCalculator(Tree* tree, Alignment* aln, Model* model);

    // Traverse the tree in post-order and compute partial likelihoods
    void traverseAndCompute(Node* node);

    void traverseAndComputeBounded(Node* node, size_t start, size_t end, int packet_id);

    Matrix buildTipLikelihood(const std::string& taxonName);

    Matrix buildTipLikelihoodBounded(const std::string& taxonName, size_t start, size_t end, int packet_id);

    // Compute one-hot likelihood matrix for a tip (leaf) node
    void computeTipLikelihood(Node* node);

    void computeTipLikelihoodBounded(Node* node, size_t start, size_t end, int packet_id);

    // Compute partial likelihood for an internal node using matrix ops
    void computeInternalLikelihood(Node* node);

    void computeInternalLikelihoodBounded(Node* node, int packet_id);

    // Compute the overall log-likelihood of the tree
    double computeLogLikelihood();

    /**
     * for distribute the alignment in to chunks
     * @param chunkSize
     * @param numThreads
     * @param limits
     */
    void computeBound(int chunkSize, int numThreads, vector<size_t>& limits);

    /**
     * Compute the likelihood for a given range of sites
     * @param start Start index of the site range
     * @param end End index of the site range
     * @param packet_id Packet ID for the current computation
     * @return The computed likelihood value
     */
    double computeLikelihoodFromBound(size_t start, size_t end, int packet_id);

    /**
     * Compute the likelihood for the root
     * @param rootL The likelihood matrix from the root node
     * @return The computed likelihood value
     */
    double computeSiteLikelihoodFromRoot(const Matrix& rootL);

    void setChunkSize(int chunkSize);

    void setNumThreads(int numThreads);

    void setIsBounded(bool isBounded);

private:
    Tree* tree_;
    Alignment* aln_;
    Model* model_;
    bool isBounded_ = false; // Flag to indicate if bounded computation is used
    int chunk_size_ = 1000; // Default chunk size for bounded computation
    int num_threads_ = 1; // Default number of threads for bounded computation
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_LIKELIHOODCALCULATOR_H
