//
// Created by Hashara Kumarasinghe on 14/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_EIGENUTILS_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_EIGENUTILS_H

#include "../Matrix.h"

void computeEigenDecomposition(const ::Matrix& Q,
                               double* eigenvalues,
                               double* eigenvectors,
                               double* inv_eigenvectors,
                               double* inv_eigenvectors_transposed);


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_EIGENUTILS_H
