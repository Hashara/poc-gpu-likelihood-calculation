//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "LikelihoodCalculator.h"
#include "../Params.h"
#include <cmath>
#if defined(USE_OPENACC) || defined(USE_CUBLAS)
#include <nvtx3/nvToolsExt.h>
#include <vector>
#endif
#ifdef USE_CUBLAS
#include "../helper/MatrixOpDispatcher.h"
#include "../helper/TipLikelihoodKernel.cuh"
#include "../helper/MatrixKernels.cuh"
#include <cuda_runtime.h>
#endif

using namespace std;

/**
 * Constructor
 * @param tree
 * @param aln
 * @param model
 */
LikelihoodCalculator::LikelihoodCalculator(Tree *tree, Alignment *aln,
                                           Model *model) {
  tree_ = tree;
  aln_ = aln;
  model_ = model;
}

LikelihoodCalculator::~LikelihoodCalculator() {
#ifdef USE_CUBLAS
  if (h_tip8_pinned_) {
    cudaFreeHost(h_tip8_pinned_);
    h_tip8_pinned_ = nullptr;
  }
  if (h_freq_pinned_) {
    cudaFreeHost(h_freq_pinned_);
    h_freq_pinned_ = nullptr;
  }
#endif
}

/**
 * Build the one-hot likelihood matrix for a tip (leaf) node
 * @param node
 * @return
 */
void LikelihoodCalculator::buildTipLikelihood(Node *node) {
  int numStates = Params::instance().numStates;
  int numPatterns = aln_->patterns.size();

  int taxonIndex = -1;
  for (size_t i = 0; i < aln_->seq_names.size(); ++i) {
    if (aln_->seq_names[i] == node->name) {
      taxonIndex = static_cast<int>(i);
      break;
    }
  }

  if (taxonIndex == -1) {
    throw std::runtime_error("Taxon name not found in alignment: " +
                             node->name);
  }

  node->partialLikelihood.resize(numStates, numPatterns);

#ifdef USE_OPENACC
  nvtxRangePushA("buildTipLikelihood");
  Matrix &L = node->partialLikelihood;
  double *__restrict__ l = L.data();
  const size_t sz = (size_t)numStates * (size_t)numPatterns;

#pragma acc enter data create(l[0 : sz]) async(3)

  std::vector<uint8_t> tip8(numPatterns);

  for (int p = 0; p < numPatterns; ++p) {
    int s = aln_->patterns[p].states[taxonIndex];
    tip8[p] = (s < numStates) ? s : (uint8_t)0xFF;
  }
  uint8_t *tip8_ptr = tip8.data();

#pragma acc enter data copyin(tip8_ptr[0 : numPatterns]) async(5)

// One-pass write: fewer stores than zero+scatter
#pragma acc parallel loop collapse(2) gang vector present(l, tip8_ptr)         \
    async(3) wait(5)
  for (int p = 0; p < numPatterns; ++p) {
    for (int s = 0; s < numStates; ++s) {
      uint8_t ss = tip8_ptr[p];
      // ambiguous => all-ones; else one-hot
      l[(size_t)p * numStates + s] =
          (ss == 0xFF) ? 1.0 : (s == (int)ss ? 1.0 : 0.0);
    }
  }

// Drop device copy of the temporary tip; wait before host frees tip8 memory
#pragma acc exit data delete (tip8_ptr[0 : numPatterns])async(7)
#pragma acc wait(7)

  nvtxRangePop();
#elif defined(USE_CUBLAS)
  nvtxRangePushA("buildTipLikelihood using CUDA");
  Matrix &L = node->partialLikelihood;
  L.allocDevice();              // allocate device memory
  double *d_l = L.deviceData(); // device pointer

  if (!h_tip8_pinned_ || h_tip8_pinned_size_ < (size_t)numPatterns) {
    if (h_tip8_pinned_) {
      cudaFreeHost(h_tip8_pinned_);
      h_tip8_pinned_ = nullptr;
    }
    if (cudaMallocHost(&h_tip8_pinned_, numPatterns * sizeof(uint8_t)) !=
        cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate pinned host buffer for tip states.");
    }
    h_tip8_pinned_size_ = (size_t)numPatterns;
  }
  for (int p = 0; p < numPatterns; ++p) {
    int s = aln_->patterns[p].states[taxonIndex];
    h_tip8_pinned_[p] = (s < numStates) ? s : (uint8_t)0xFF;
  }

  // OPTIMIZATION: Use cached tip8 buffer to avoid repeated cudaMalloc/cudaFree
  static uint8_t *d_tip8 = nullptr;
  static size_t d_tip8_size = 0;
  if (!d_tip8 || d_tip8_size < (size_t)numPatterns) {
    if (d_tip8)
      cudaFree(d_tip8);
    cudaMalloc(&d_tip8, numPatterns * sizeof(uint8_t));
    d_tip8_size = numPatterns;
  }

  // OPTIMIZATION: Reuse stream from MatrixOpCuBLAS singleton instead of
  // creating per node
  cudaStream_t stream = getCuBLASStream();

  cudaMemcpyAsync(d_tip8, h_tip8_pinned_, numPatterns * sizeof(uint8_t),
                  cudaMemcpyHostToDevice, stream);

  int blockSize = 256; // OPTIMIZATION: Match kernel thread count

  launchBuildTipPartials(d_tip8, d_l, numStates, numPatterns, blockSize,
                         stream);

  // OPTIMIZATION: Don't synchronize here! Let operations queue asynchronously.
  // The tip likelihood will be synchronized when needed by compositehadamard
  // through the Matrix::waitHtoD mechanism or stream ordering.

  nvtxRangePop();
#else
  node->partialLikelihood.resize(numStates, numPatterns);

  Matrix &L = node->partialLikelihood;

  double *l = L.data();
  std::fill(l, l + numPatterns, 0.0);

  for (int p = 0; p < numPatterns; ++p) {
    l[p * numStates + aln_->patterns[p].states[taxonIndex]] = 1.0;
  }

#endif
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
  if (!node->isLeaf())
    return;

  buildTipLikelihood(node);
}

/**
 * Compute the partial likelihood for an internal node
 * computes L = (P1 * L1) hadamard (P2 * L2)
 * @param node
 */
void LikelihoodCalculator::computeInternalLikelihood(Node *node,
                                                     uint8_t *scale_count) {
  if (node->isLeaf())
    return;

  // Dispatch to eigenspace if enabled
  if (useEigenSpace_) {
    computeInternalLikelihoodEigen(node, scale_count);
    return;
  }

  int numStates = Params::instance().numStates;

  Node *left = node->children[0];
  Node *right = node->children[1];

  const Matrix &L1 = left->partialLikelihood;
  const Matrix &L2 = right->partialLikelihood;

  //    Matrix P1 = model_->getTransitionMatrix(left->branchLength);
  //    Matrix P2 = model_->getTransitionMatrix(right->branchLength);
  static Matrix P1(numStates, numStates);
  P1.resize(numStates, numStates); // no-op after first call
  model_->buildTransitionMatrix(left->branchLength, P1);

  static Matrix P2(numStates, numStates);
  P2.resize(numStates, numStates); // no-op after first call
  model_->buildTransitionMatrix(right->branchLength, P2);

#if !(defined(USE_OPENACC) || defined(USE_CUBLAS))
  Matrix PL1 = P1 * L1;
  Matrix PL2 = P2 * L2;

#ifdef USE_EIGEN
  node->partialLikelihood = hadamard(PL1, PL2);
#else
  node->partialLikelihood = PL1.hadamard(PL2);
#endif
#elif defined(USE_OPENACC)
  P1.compositeHadamard(L1, P2, L2, node->partialLikelihood, scale_count);
#elif defined(USE_CUBLAS)
  switch (numStates) {
  case 4:
    compositeCuBLASHadamard<4>(P1, L1, P2, L2, node->partialLikelihood,
                               scale_count);
    break;
  case 20:
    compositeCuBLASHadamard<20>(P1, L1, P2, L2, node->partialLikelihood,
                                scale_count);
    break;
  default:
    throw std::invalid_argument(
        "computeInternalLikelihood only supports numStates 4 or 20 for CUBLAS");
  }

#endif
  node->isPartialLikelihoodCalculated = true;
}

/**
 * Compute partial likelihood using eigenspace formulation (IQ-TREE style).
 *
 * Instead of computing P(t) = full transition matrix, we use:
 *   P(t) = U · diag(exp(λ·t)) · U⁻¹
 *
 * Split into:
 *   echildren(t) = U · diag(exp(λ·t))     (per branch)
 *   inv_evec = U⁻¹                         (constant)
 *
 * At each internal node:
 *   L_dad = inv_evec * ((E_left * L_left) ⊙ (E_right * L_right))
 *
 * @param node Internal node to compute
 * @param scale_count Per-site scaling counter
 */
void LikelihoodCalculator::computeInternalLikelihoodEigen(Node *node,
                                                           uint8_t *scale_count) {
  if (node->isLeaf()) return;

  int numStates = Params::instance().numStates;
  Node *left = node->children[0];
  Node *right = node->children[1];

  const Matrix &L1 = left->partialLikelihood;
  const Matrix &L2 = right->partialLikelihood;

#if defined(USE_CUBLAS)
  // ===== GPU eigenspace path =====
  MatrixOpCuBLAS *cublas = getCuBLASBackend();
  cudaStream_t stream = getCuBLASStream();

  // Upload eigendecomposition data to GPU once
  if (!eigenDataUploaded_) {
    cublas->uploadEigenData(
        model_->getEigenvectorsFlat(),
        model_->getEigenvaluesFlat(),
        model_->getInvEigenvectorsFlat(),
        numStates);
    eigenDataUploaded_ = true;
  }

  // Build echildren on GPU for each branch
  launchBuildEChildren(
      cublas->getDeviceEigenvectors(),
      cublas->getDeviceEigenvalues(),
      cublas->getDeviceEChildLeft(),
      numStates, left->branchLength, stream);

  launchBuildEChildren(
      cublas->getDeviceEigenvectors(),
      cublas->getDeviceEigenvalues(),
      cublas->getDeviceEChildRight(),
      numStates, right->branchLength, stream);

  // Launch the eigenspace composite Hadamard kernel:
  // L_dad = inv_evec * ((E_left * L_left) ⊙ (E_right * L_right))
  cublas->eigenCompositeHadamard(
      cublas->getDeviceEChildLeft(), L1,
      cublas->getDeviceEChildRight(), L2,
      node->partialLikelihood);

#else
  // ===== CPU eigenspace path =====
  int K = numStates;
  int P = (int)L1.cols();

  // Build echildren matrices on CPU
  static Matrix E_left, E_right;
  model_->buildEChildren(left->branchLength, E_left);
  model_->buildEChildren(right->branchLength, E_right);

  // E * L̃ converts eigenspace-basis partials to original basis (= P(t) * L)
  Matrix EL1 = E_left * L1;   // K × P (original basis)
  Matrix EL2 = E_right * L2;  // K × P (original basis)

  // Hadamard product in original basis (valid Felsenstein pruning step)
#ifdef USE_EIGEN
  Matrix eigen_product = hadamard(EL1, EL2);
#else
  Matrix eigen_product = EL1.hadamard(EL2);
#endif

  // Transform back to eigenspace: L̃_dad = U⁻¹ · product
  const double *inv = model_->getInvEigenvectorsFlat();
  node->partialLikelihood.resize(K, P);
  double *result = node->partialLikelihood.data();
  const double *ep = eigen_product.data();

  for (int ptn = 0; ptn < P; ++ptn) {
    for (int s = 0; s < K; ++s) {
      double sum = 0.0;
      for (int x = 0; x < K; ++x) {
        // inv_evec(s,x) = inv[x*K + s] (column-major)
        // eigen_product(x, ptn) = ep[ptn*K + x] (column-major)
        sum += inv[x * K + s] * ep[ptn * K + x];
      }
      result[ptn * K + s] = sum;
    }
  }
#endif

  node->isPartialLikelihoodCalculated = true;
}

/**
 * Compute the partial likelihood for an internal node for bounded computation
 * computes L = (P1 * L1) hadamard (P2 * L2)
 * @param node
 * @param packet_id
 */
void LikelihoodCalculator::computeInternalLikelihoodBounded(Node *node,
                                                            int packet_id) {
  if (node->isLeaf())
    return;

  Node *left = node->children[0];
  Node *right = node->children[1];

  const Matrix &L1 = left->partialLikelihood;
  const Matrix &L2 = right->partialLikelihood;

  Matrix P1 = model_->getTransitionMatrix(left->branchLength);
  Matrix P2 = model_->getTransitionMatrix(right->branchLength);

#if !(defined(USE_OPENACC) || defined(USE_CUBLAS))
  Matrix PL1 = P1 * L1;
  Matrix PL2 = P2 * L2;

#ifdef USE_EIGEN
  node->partialLikelihood = hadamard(PL1, PL2);
#else
  node->partialLikelihood = PL1.hadamard(PL2);
#endif
#elif defined(USE_OPENACC)
  P1.compositeHadamard(L1, P2, L2, node->partialLikelihood, nullptr);
#elif defined(USE_CUBLAS)
  int numStates = Params::instance().numStates;
  switch (numStates) {
  case 4:
    compositeCuBLASHadamard<4>(P1, L1, P2, L2, node->partialLikelihood,
                               nullptr);
    break;
  case 20:
    compositeCuBLASHadamard<20>(P1, L1, P2, L2, node->partialLikelihood,
                                nullptr);
    break;
  default:
    throw std::invalid_argument(
        "computeInternalLikelihoodBounded only supports numStates 4 or 20 for "
        "CUBLAS");
  }
#endif
  node->completedPackets.insert(packet_id);
}

/**
 * Traverse the tree in post-order and compute partial likelihoods
 * @param node --> pass the root node here
 */
/*
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
#ifdef USE_OPENACC
        #pragma acc wait(1)
#endif
        computeInternalLikelihood(node);
    }
}
*/

void LikelihoodCalculator::traverseAndCompute(Node *root,
                                              uint8_t *scale_count) {
  std::vector<Node *> post;

  size_t scale_count_size = aln_->patterns.size();
#ifdef USE_OPENACC
#pragma acc enter data create(scale_count[0 : scale_count_size]) async(1)
#pragma acc parallel loop present(scale_count[0 : scale_count_size]) async(1)
  for (size_t i = 0; i < scale_count_size; ++i)
    scale_count[i] = 0;
#elif defined(USE_CUBLAS)
  // scale_count stays GPU-resident for entire traversal — zero it on device
  resetCuBLASScaleCount((int)scale_count_size);
#else
  std::fill(scale_count, scale_count + scale_count_size, 0);
#endif
  buildPostorder(root, post);

  // Phase 1: tips (independent)
  for (Node *n : post) {
    if (!n->isPartialLikelihoodCalculated && n->isLeaf()) {
      computeTipLikelihood(n);
      n->isPartialLikelihoodCalculated = true;
    }
  }

  // Phase 1b: If eigenspace mode, transform tip likelihoods into eigenspace
  //   L̃_tip = U⁻¹ · L_tip
  // All subsequent internal node computations expect eigenspace-basis inputs.
  if (useEigenSpace_) {
    int K = model_->getNumStates();
#if defined(USE_CUBLAS)
    MatrixOpCuBLAS *cublas = getCuBLASBackend();
    if (!eigenDataUploaded_) {
      cublas->uploadEigenData(
          model_->getEigenvectorsFlat(),
          model_->getEigenvaluesFlat(),
          model_->getInvEigenvectorsFlat(),
          K);
      eigenDataUploaded_ = true;
    }
    for (Node *n : post) {
      if (n->isLeaf()) {
        cublas->transformToEigenSpace(n->partialLikelihood);
      }
    }
#else
    const double *inv = model_->getInvEigenvectorsFlat();
    for (Node *n : post) {
      if (!n->isLeaf()) continue;
      Matrix &L = n->partialLikelihood;
      int P = (int)L.cols();
      Matrix Ltrans(K, P);
      double *dst = Ltrans.data();
      const double *src = L.data();
      for (int ptn = 0; ptn < P; ++ptn) {
        for (int s = 0; s < K; ++s) {
          double sum = 0.0;
          for (int x = 0; x < K; ++x) {
            // inv_evec(s,x) = inv[x*K + s] (column-major)
            sum += inv[x * K + s] * src[ptn * K + x];
          }
          dst[ptn * K + s] = sum;
        }
      }
      L = Ltrans;
    }
#endif
  }

#ifdef USE_OPENACC
#pragma acc wait(1)
#endif

  // Phase 2: internal nodes
  for (Node *n : post) {
    if (!n->isPartialLikelihoodCalculated && !n->isLeaf()) {
      computeInternalLikelihood(n, scale_count);
      n->isPartialLikelihoodCalculated = true;
    }
  }

  // scale_count remains GPU-resident — computeLogLikelihood reads it directly
}

void LikelihoodCalculator::buildPostorder(Node *n, std::vector<Node *> &out) {
  for (Node *c : n->children)
    buildPostorder(c, out);
  out.push_back(n);
}

/**
 * Traverse the tree in post-order and compute partial likelihoods for bounded
 * computation
 * @param node --> pass the root node here
 * @param start --> start index of the chunk
 * @param end --> end index of the chunk
 * @param packet_id --> id of the current chunk
 */
void LikelihoodCalculator::traverseAndComputeBounded(Node *node, size_t start,
                                                     size_t end,
                                                     int packet_id) {
  // Skip if already computed
  if (node->completedPackets.find(packet_id) != node->completedPackets.end()) {
#ifdef VERBOSE
    cout << "Node " << node->name << " already computed for packet "
         << packet_id << endl;
#endif
    return;
  }

  // Recurse through children first (post-order)
  for (Node *child : node->children) {
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
      cout << "Computed likelihood for chunk " << packetId << " from " << start
           << " to " << end << "chunk log L: " << chunkLogL << endl;
#endif
    }
  } else {
    size_t scale_count_size = aln_->patterns.size();
    uint8_t scale_count[scale_count_size];

    traverseAndCompute(tree_->root, scale_count);
    Matrix &rootL = tree_->root->partialLikelihood;

    // If eigenspace mode, root partial likelihoods are in eigenspace basis.
    // Transform back to original basis: L_root = U · L̃_root
    if (useEigenSpace_) {
#if defined(USE_CUBLAS)
      MatrixOpCuBLAS *cublas = getCuBLASBackend();
      cublas->transformFromEigenSpace(rootL);
#else
      int K = model_->getNumStates();
      int P = (int)rootL.cols();
      const double *U = model_->getEigenvectorsFlat();
      Matrix Loriginal(K, P);
      double *dst = Loriginal.data();
      const double *src = rootL.data();
      for (int ptn = 0; ptn < P; ++ptn) {
        for (int s = 0; s < K; ++s) {
          double sum = 0.0;
          for (int x = 0; x < K; ++x) {
            // U(s,x) = U[x*K + s] (column-major)
            sum += U[x * K + s] * src[ptn * K + x];
          }
          dst[ptn * K + s] = sum;
        }
      }
      rootL = Loriginal;
#endif
    }

    logL = computeSiteLikelihoodFromRoot(rootL, scale_count);
  }
  return logL;
}

/**
 * Divide the alignment into chunks for bounded computation
 * @param chunkSize Size of each chunk
 * @param numThreads Number of threads to use
 * @param limits Vector to store the start and end indices of each chunk
 */
void LikelihoodCalculator::computeBound(int chunkSize, int numThreads,
                                        vector<size_t> &limits) {
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
double LikelihoodCalculator::computeLikelihoodFromBound(size_t start,
                                                        size_t end,
                                                        int packet_id) {
#ifdef VERBOSE
  cout << "Processing chunk in computeLikelihoodFromBound() " << packet_id
       << " from " << start << " to " << end << endl;
#endif
  traverseAndComputeBounded(tree_->root, start, end, packet_id);

  const Matrix &rootL = tree_->root->partialLikelihood;

  return computeSiteLikelihoodFromRoot(rootL, nullptr);
}

/**
 * Build the one-hot likelihood matrix for a tip (leaf) node for bounded
 * computation
 * @param taxonName
 * @param start
 * @param end
 * @param packet_id
 * @return one-hot likelihood matrix for the given range of sites
 */
Matrix LikelihoodCalculator::buildTipLikelihoodBounded(
    const std::string &taxonName, size_t start, size_t end, int packet_id) {
  int numStates = Params::instance().numStates;
  int numPatterns = end - start;

#ifdef VERBOSE
  cout << "Processing packet " << packet_id << " from " << start << " to "
       << end << endl;
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
    int state = aln_->patterns[p].states[taxonIndex];
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
void LikelihoodCalculator::computeTipLikelihoodBounded(Node *node, size_t start,
                                                       size_t end,
                                                       int packet_id) {
  if (!node->isLeaf())
    return;

  node->partialLikelihood =
      buildTipLikelihoodBounded(node->name, start, end, packet_id);
  //    node->isPartialLikelihoodCalculated = true;

  node->completedPackets.insert(
      packet_id); // Mark this packet as completed for the leaf node
}

void LikelihoodCalculator::setIsBounded(bool isBounded) {
  isBounded_ = isBounded;
}

double
LikelihoodCalculator::computeSiteLikelihoodFromRoot(const Matrix &rootL,
                                                    uint8_t *scale_count) {
  double logL = 0.0;
#if !defined(USE_CUBLAS)
  constexpr double kMinSiteLikelihood = 1.0e-300;
#endif

#ifdef VERBOSE
  cout << "Root partial likelihood matrix:\n";
  cout << rootL << endl;
#endif

  int numPatterns = rootL.cols();

  Matrix baseFrequencies = model_->getBaseFrequencies();

#if defined(USE_OPENACC)
  //  Precompute frequencies to avoid GPU accessing aln_
  int *freqData = new int[numPatterns];

  for (int j = 0; j < numPatterns; ++j) {
    freqData[j] = aln_->patterns[j].frequency;
  }
#pragma acc enter data copyin(freqData[0 : numPatterns]) async(2)
  Matrix siteLikelihoods(1, numPatterns);
  baseFrequencies.multiplyInPlace(rootL, siteLikelihoods);

#elif defined(USE_CUBLAS)
  //  Precompute frequencies for GPU reduction kernel
  if (!h_freq_pinned_ || h_freq_pinned_size_ < (size_t)numPatterns) {
    if (h_freq_pinned_) {
      cudaFreeHost(h_freq_pinned_);
      h_freq_pinned_ = nullptr;
    }
    if (cudaMallocHost(&h_freq_pinned_, numPatterns * sizeof(int)) !=
        cudaSuccess) {
      throw std::runtime_error("Failed to allocate pinned host buffer for frequencies.");
    }
    h_freq_pinned_size_ = (size_t)numPatterns;
  }
  for (int j = 0; j < numPatterns; ++j) {
    h_freq_pinned_[j] = aln_->patterns[j].frequency;
  }

#else
  Matrix siteLikelihoods = baseFrequencies * rootL;
#endif

#ifdef USE_OPENACC

  // Extract a flat pointer to siteLikelihoods
  double *sl = siteLikelihoods.data();

#pragma acc wait(2)
#pragma acc parallel loop reduction(+ : logL)                                  \
    present(sl[0 : numPatterns], freqData[0 : numPatterns])
  for (int j = 0; j < numPatterns; ++j) {
    double siteLikelihood = sl[j]; // 1-row matrix, first row
    if (siteLikelihood < kMinSiteLikelihood)
      siteLikelihood = kMinSiteLikelihood;
    logL += freqData[j] *
            (std::log(siteLikelihood) + scale_count[j] * LOG_SCALING_THRESHOLD);
  }

#pragma acc exit data delete (sl[0 : numPatterns],                             \
                              freqData[0 : numPatterns])async

#elif defined(USE_CUBLAS)
  // GPU reduction: DGEMM + log-likelihood sum entirely on device
  logL = computeCuBLASLogLikelihood(baseFrequencies, rootL, h_freq_pinned_,
                                    numPatterns, LOG_SCALING_THRESHOLD);

#else
  // CPU fallback (no scaling)
  for (int j = 0; j < numPatterns; ++j) {
    double siteLikelihood =
        siteLikelihoods(0, j); // Assuming siteLikelihoods is a 1-row matrix
    if (siteLikelihood < kMinSiteLikelihood)
      siteLikelihood = kMinSiteLikelihood;

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
