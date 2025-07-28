//
// Created by Hashara Kumarasinghe on 15/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPTYPE_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPTYPE_H

enum class MatrixOpType {
    CPU,
    EIGEN,
    CUBLAS,
    CUDA_KERNEL,
    OPENACC,
    OPENMP_GPU,
};

inline std::string toString(MatrixOpType type) {
    switch (type) {
        case MatrixOpType::CPU:         return "CPU";
        case MatrixOpType::EIGEN:       return "EIGEN";
        case MatrixOpType::CUBLAS:      return "CUBLAS";
        case MatrixOpType::CUDA_KERNEL: return "CUDA_KERNEL";
        case MatrixOpType::OPENACC:     return "OPENACC";
        case MatrixOpType::OPENMP_GPU:  return "OPENMP_GPU";
        default:                        return "UNKNOWN";
    }
}

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_MATRIXOPTYPE_H
