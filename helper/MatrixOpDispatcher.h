//
// Created by Hashara Kumarasinghe on 16/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPDISPATCHER_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPDISPATCHER_H

#ifndef USE_EIGEN

#include "MatrixOp.h"
#include "MatrixOpCPU.h"
#include "MatrixOpType.h"

#ifdef USE_OPENACC
#include "MatrixOpOpenACC.h"
#elif defined(USE_CUBLAS) && defined(USE_CUDA)
#include "MatrixOpCuBLAS.h"
#elif defined(USE_CUDA)
#include "MatrixOpCUDA.cuh"
#elif defined(USE_OPENMP_GPU)
#include "MatrixOpOpenMPGPU.h"
#endif

// Function to return a backend singleton based on MatrixOpType
inline MatrixOp *getBackend(MatrixOpType type) {
  switch (type) {
  case MatrixOpType::CPU: {
    static MatrixOpCPU op;
    return &op;
  }
#ifdef USE_OPENACC
  case MatrixOpType::OPENACC: {
    static MatrixOpOpenACC op;
    return &op;
  }
#elif defined(USE_CUBLAS) && defined(USE_CUDA)
  case MatrixOpType::CUBLAS: {
    static MatrixOpCuBLAS op;
    return &op;
  }
#elif defined(USE_CUDA)
  case MatrixOpType::CUDA_KERNEL: {
    static MatrixOpCUDA op;
    return &op;
  }
#elif defined(USE_OPENMP_GPU)
  case MatrixOpType::OPENMP_GPU: {
    static MatrixOpOpenMPGPU op;
    return &op;
  }
#endif
  default:
    throw std::invalid_argument("Unsupported MatrixOpType in getBackend");
  }
}

#if defined(USE_CUBLAS) && defined(USE_CUDA)
// Helper to get CUBLAS stream for reuse in LikelihoodCalculator
inline cudaStream_t getCuBLASStream() {
  static MatrixOpCuBLAS *op =
      dynamic_cast<MatrixOpCuBLAS *>(getBackend(MatrixOpType::CUBLAS));
  return op ? op->getStream() : nullptr;
}

// Helper to zero GPU-resident scale_count at start of traversal
inline void resetCuBLASScaleCount(int P) {
  static MatrixOpCuBLAS *op =
      dynamic_cast<MatrixOpCuBLAS *>(getBackend(MatrixOpType::CUBLAS));
  if (op)
    op->resetScaleCount(P);
}

// Helper to copy GPU-resident scale_count back to host after traversal
inline void syncCuBLASScaleCount(uint8_t *host_ptr, int P) {
  static MatrixOpCuBLAS *op =
      dynamic_cast<MatrixOpCuBLAS *>(getBackend(MatrixOpType::CUBLAS));
  if (op)
    op->syncScaleCount(host_ptr, P);
}
#endif

#endif // USE_EIGEN
#endif // POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPDISPATCHER_H
