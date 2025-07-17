//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "Model.h"

#if defined(USE_EIGEN) && defined(DECOMP)
const Matrix &Model::getEigenvalues() const {
    return eigenvalues;
}

const Matrix &Model::getEigenvectors() const {
    return eigenvectors;
}

const Matrix &Model::getInvEigenvectors() const {
    return inv_eigenvectors;
}

#endif