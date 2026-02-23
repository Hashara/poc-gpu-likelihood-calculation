//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H

#include <vector>
#include <string>
#include "../helper/Matrix.h"
//#define DECOMP 1

class Model {
public:
    virtual ~Model() = default;

    virtual std::string getName() const = 0;

    virtual Matrix getBaseFrequencies() const = 0;

    // Transition probability matrix P(t) for a given branch length t
    virtual Matrix getTransitionMatrix(double t) const = 0;

    virtual void buildTransitionMatrix(double t, Matrix &P) const = 0;


    // 4x4 substitution rate matrix for DNA
    virtual Matrix getRateMatrix() const = 0;

    // =========================================================================
    // Eigendecomposition support (IQ-TREE-style eigenspace formulation)
    // =========================================================================
    // Q = U · diag(λ) · U⁻¹
    // P(t) = U · diag(exp(λ·t)) · U⁻¹
    // echildren(t) = U · diag(exp(λ·t))
    //
    // At each internal node:
    //   L_dad = U⁻¹ · ((E_left · L_left) ⊙ (E_right · L_right))

    /** Returns true if eigendecomposition has been computed */
    bool hasEigenDecomp() const { return eigenDecompReady_; }

    /** Number of states (4 for DNA, 20 for protein) */
    int getNumStates() const;

    /** Get eigenvalues λ (K doubles, flat array) */
    const double* getEigenvaluesFlat() const;

    /** Get eigenvectors U (K×K, column-major flat array) */
    const double* getEigenvectorsFlat() const;

    /** Get inverse eigenvectors U⁻¹ (K×K, column-major flat array) */
    const double* getInvEigenvectorsFlat() const;

    /**
     * Build echildren matrix E(t) = U · diag(exp(λ·t))
     * Result is K×K column-major, written into E.
     */
    void buildEChildren(double t, Matrix &E) const;

#if defined(USE_EIGEN) && defined(DECOMP)
    // Legacy Eigen library interface (kept for backward compatibility)
    const Matrix &getEigenvalues() const;
    const Matrix &getEigenvectors() const;
    const Matrix &getInvEigenvectors() const;
    virtual Matrix getExpDiagMatrix(double t) const = 0;
#endif

protected:
    Matrix rateMatrix_;
    Matrix baseFreq_;

    // Eigendecomposition storage (flat arrays, column-major for matrices)
    bool eigenDecompReady_ = false;
    int numStates_ = 0;
    std::vector<double> eigenvalues_;     // [K]
    std::vector<double> eigenvectors_;    // [K*K] column-major (U)
    std::vector<double> invEigenvectors_; // [K*K] column-major (U⁻¹)

    /** Compute eigendecomposition of rateMatrix_. Call from subclass constructor. */
    void computeEigenDecomp();

#if defined(USE_EIGEN) && defined(DECOMP)
    Eigen::VectorXd eigenvalues;
    Matrix eigenvectors ;
    Matrix inv_eigenvectors ;
#endif
};



#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MODEL_H
