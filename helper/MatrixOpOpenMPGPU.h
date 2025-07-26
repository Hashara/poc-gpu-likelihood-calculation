//
// Created by Hashara Kumarasinghe on 27/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPOPENMPGPU_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPOPENMPGPU_H

#include "MatrixOp.h"
#include <omp.h>

class MatrixOpOpenMPGPU: public MatrixOp {
public:
Matrix multiply(const Matrix& A, const Matrix& B) override;
Matrix hadamard(const Matrix& A, const Matrix& B) override;
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPOPENMPGPU_H
