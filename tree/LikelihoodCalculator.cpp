//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "LikelihoodCalculator.h"
#include <cmath>
#define COMPOSITE_HADAMARD

/**
 * Constructor
 * @param tree
 * @param aln
 * @param model
 */
LikelihoodCalculator::LikelihoodCalculator(Tree *tree, Alignment *aln, Model *model) {
    tree_ = tree;
    aln_ = aln;
    model_ = model;
}

/**
 * Build the one-hot likelihood matrix for a tip (leaf) node
 * @param node
 * @return
 */
void LikelihoodCalculator::buildTipLikelihood(Node *node) {
    int numStates = 4;
    int numPatterns = aln_->patterns.size();

    int taxonIndex = -1;
    for (size_t i = 0; i < aln_->seq_names.size(); ++i) {
        if (aln_->seq_names[i] == node->name) {
            taxonIndex = static_cast<int>(i);
            break;
        }
    }

    if (taxonIndex == -1) {
        throw std::runtime_error("Taxon name not found in alignment: " + node->name);
    }

    node->partialLikelihood.resize(numStates, numPatterns);

    Matrix& L = node->partialLikelihood;
//    node->partialLikelihood(numStates, numPatterns);
    for (int p = 0; p < numPatterns; ++p) {
        int state = (*aln_).patterns[p][taxonIndex] - '0';
        for (int s = 0; s < numStates; ++s) {
            L(s, p) = (s == state) ? 1.0 : 0.0;
        }
    }

#ifdef VERBOSE
    cout << "Tip likelihood for " << taxonName << ":\n";

    cout << L << endl;
#endif
    node->isPartialLikelihoodCalculated = true;
}

/**
 * Compute the partial likelihood for a tip (leaf) node
 * @param node
 */
void LikelihoodCalculator::computeTipLikelihood(Node *node) {
    if (!node->isLeaf()) return;

    buildTipLikelihood(node);
}

/**
 * Compute the partial likelihood for an internal node
 * computes L = (P1 * L1) hadamard (P2 * L2)
 * @param node
 */
void LikelihoodCalculator::computeInternalLikelihood(Node *node) {
    if (node->isLeaf()) return;

    Node *left = node->children[0];
    Node *right = node->children[1];


    const Matrix &L1 = left->partialLikelihood;
    const Matrix &L2 = right->partialLikelihood;

    Matrix P1 = model_->getTransitionMatrix(left->branchLength);
    Matrix P2 = model_->getTransitionMatrix(right->branchLength);

#if !defined(COMPOSITE_HADAMARD) || !defined(USE_OPENACC)
    Matrix PL1 = P1 * L1;
    Matrix PL2 = P2 * L2;

#ifdef USE_EIGEN
    node->partialLikelihood = hadamard(PL1, PL2);
#else
    node->partialLikelihood = PL1.hadamard(PL2);
#endif
#else
    P1.compositeHadamard(L1, P2, L2, node->partialLikelihood);
#endif
    node->isPartialLikelihoodCalculated = true;
}

/**
 * Compute the partial likelihood for an internal node for bounded computation
 * computes L = (P1 * L1) hadamard (P2 * L2)
 * @param node
 * @param packet_id
 */
void LikelihoodCalculator::computeInternalLikelihoodBounded(Node *node, int packet_id) {
    if (node->isLeaf()) return;

    Node *left = node->children[0];
    Node *right = node->children[1];


    const Matrix &L1 = left->partialLikelihood;
    const Matrix &L2 = right->partialLikelihood;

    Matrix P1 = model_->getTransitionMatrix(left->branchLength);
    Matrix P2 = model_->getTransitionMatrix(right->branchLength);
#if !defined(COMPOSITE_HADAMARD) || !defined(USE_OPENACC)
    Matrix PL1 = P1 * L1;
    Matrix PL2 = P2 * L2;

#ifdef USE_EIGEN
    node->partialLikelihood = hadamard(PL1, PL2);
#else
    node->partialLikelihood = PL1.hadamard(PL2);
#endif
#else
    P1.compositeHadamard(L1, P2, L2, node ->partialLikelihood);
#endif
    node->completedPackets.insert(packet_id);

}

/**
 * Traverse the tree in post-order and compute partial likelihoods
 * @param node --> pass the root node here
 */
void LikelihoodCalculator::traverseAndCompute(Node *node) {
    // Skip if already computed
    if (node->isPartialLikelihoodCalculated) return;

    // Recurse through children first (post-order)
    for (Node *child: node->children) {
        traverseAndCompute(child);
    }

    // Compute based on node type
    if (node->isLeaf()) {
        computeTipLikelihood(node);
    } else {
        computeInternalLikelihood(node);
    }
}

/**
 * Traverse the tree in post-order and compute partial likelihoods for bounded computation
 * @param node --> pass the root node here
 * @param start --> start index of the chunk
 * @param end --> end index of the chunk
 * @param packet_id --> id of the current chunk
 */
void LikelihoodCalculator::traverseAndComputeBounded(Node *node,
                                                     size_t start, size_t end, int packet_id) {
    // Skip if already computed
    if (node->completedPackets.find(packet_id) != node->completedPackets.end()) {
#ifdef VERBOSE
        cout << "Node " << node->name << " already computed for packet " << packet_id << endl;
#endif
        return;
    }

    // Recurse through children first (post-order)
    for (Node *child: node->children) {
        traverseAndComputeBounded(child, start, end, packet_id);
    }

    // Compute based on node type
    if (node->isLeaf()) {
        computeTipLikelihoodBounded(node, start, end, packet_id);
    } else {
        computeInternalLikelihoodBounded(node, packet_id);
    }
}

/**
 * Compute the overall log-likelihood of the tree
 * @return logL
 */
double LikelihoodCalculator::computeLogLikelihood() {
    // Traverse and compute partial likelihoods for all nodes
    double logL = 0.0;

    if (isBounded_) {
        vector<size_t> limits;

        computeBound(chunk_size_, num_threads_, limits);

        for (size_t packetId = 0; packetId < limits.size() - 1; ++packetId) {
            size_t start = limits[packetId];
            size_t end = limits[packetId + 1];

            double chunkLogL = computeLikelihoodFromBound(start, end, packetId);
            logL += chunkLogL;

#ifdef VERBOSE
            cout << "Computed likelihood for chunk " << packetId << " from " << start << " to " << end <<
                 "chunk log L: " << chunkLogL << endl;
#endif
        }
    } else {
        traverseAndCompute(tree_->root);
        logL = computeSiteLikelihoodFromRoot(tree_->root->partialLikelihood);
    }
    return logL;
}

/**
 * Divide the alignment into chunks for bounded computation
 * @param chunkSize Size of each chunk
 * @param numThreads Number of threads to use
 * @param limits Vector to store the start and end indices of each chunk
 */
void LikelihoodCalculator::computeBound(int chunkSize, int numThreads, vector<size_t> &limits) {
    size_t totalPatterns = aln_->patterns.size();
    size_t numChunks = (totalPatterns + chunkSize - 1) / chunkSize;
    limits.clear();
    for (size_t i = 0; i <= numChunks; ++i) {
        limits.push_back(std::min(i * chunkSize, totalPatterns));
    }
}

/**
 * Compute the likelihood for a given range of sites (bounded computation)
 * @param start Start index of the site range
 * @param end End index of the site range
 * @param packet_id Packet ID for the current computation
 * @return The computed likelihood value
 */
double LikelihoodCalculator::computeLikelihoodFromBound(size_t start, size_t end, int packet_id) {
#ifdef VERBOSE
    cout << "Processing chunk in computeLikelihoodFromBound() " << packet_id << " from " << start << " to " << end
         << endl;
#endif
    traverseAndComputeBounded(tree_->root, start, end, packet_id);


    const Matrix &rootL = tree_->root->partialLikelihood;

    return computeSiteLikelihoodFromRoot(rootL);

}

/**
 * Build the one-hot likelihood matrix for a tip (leaf) node for bounded computation
 * @param taxonName
 * @param start
 * @param end
 * @param packet_id
 * @return one-hot likelihood matrix for the given range of sites
 */
Matrix
LikelihoodCalculator::buildTipLikelihoodBounded(const std::string &taxonName, size_t start, size_t end, int packet_id) {
    int numStates = 4;
    int numPatterns = end - start;

#ifdef VERBOSE
    cout << "Processing packet " << packet_id << " from " << start << " to " << end << endl;
#endif
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
    for (size_t p = start; p < end; ++p) {
        int state = aln_->patterns[p][taxonIndex] - '0';  // Assumes pattern is a string of chars
        for (int s = 0; s < numStates; ++s) {
            L(s, p - start) = (s == state) ? 1.0 : 0.0;
        }
    }

#ifdef VERBOSE
    cout << "Tip likelihood for " << taxonName << ":\n";

    cout << L << endl;
#endif
    return L;
}

/**
 * Compute the partial likelihood for a tip (leaf) node for bounded computation
 * @param node
 * @param start
 * @param end
 * @param packet_id
 */
void LikelihoodCalculator::computeTipLikelihoodBounded(Node *node, size_t start, size_t end, int packet_id) {
    if (!node->isLeaf()) return;

    node->partialLikelihood = buildTipLikelihoodBounded(node->name, start, end, packet_id);
//    node->isPartialLikelihoodCalculated = true;

    node->completedPackets.insert(packet_id);  // Mark this packet as completed for the leaf node
}

void LikelihoodCalculator::setIsBounded(bool isBounded) {
    isBounded_ = isBounded;
}

double LikelihoodCalculator::computeSiteLikelihoodFromRoot(const Matrix &rootL) {
    double logL = 0.0;

#ifdef VERBOSE
    cout << "Root partial likelihood matrix:\n";
    cout << rootL << endl;
#endif

    int numPatterns = rootL.cols();

    Matrix baseFrequencies = model_->getBaseFrequencies();
    Matrix siteLikelihoods = baseFrequencies * rootL;

#ifdef USE_OPENACC


    // Extract a flat pointer to siteLikelihoods (assuming row-major)
    double* sl = siteLikelihoods.data();

    //  Precompute frequencies to avoid GPU accessing aln_
    int *freqData = new int[numPatterns];
    for (int j = 0; j < numPatterns; ++j) {
        freqData[j] = aln_->patterns[j].frequency;
    }

    #pragma acc enter data copyin(sl[0:numPatterns], freqData[0:numPatterns])

    #pragma acc parallel loop reduction(+:logL) present(sl[0:numPatterns], freqData[0:numPatterns])
    for (int j = 0; j < numPatterns; ++j) {
        double siteLikelihood = sl[j];  // 1-row matrix, first row
        logL += freqData[j] * std::log(siteLikelihood);
    }

    #pragma acc exit data delete(sl[0:numPatterns], freqData[0:numPatterns])

#else
    for (int j = 0; j < numPatterns; ++j) {
        double siteLikelihood = siteLikelihoods(0, j);  // Assuming siteLikelihoods is a 1-row matrix

        // Multiply by the pattern frequency
        int freq = aln_->patterns[j].frequency;
        logL += freq * std::log(siteLikelihood);

    }
#endif
    return logL;
}

void LikelihoodCalculator::setChunkSize(int chunkSize) {
    chunk_size_ = chunkSize;
}

void LikelihoodCalculator::setNumThreads(int numThreads) {
    num_threads_ = numThreads;
}

