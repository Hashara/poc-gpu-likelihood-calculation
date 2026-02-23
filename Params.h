//
// Created by Hashara Kumarasinghe on 28/9/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_PARAMS_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_PARAMS_H

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_PARAMS_H

#pragma once
#include <string>

enum SeqType { SEQ_DNA, SEQ_PROTEIN };

class Params {
public:
    int a_row, a_col, b_row, b_col;
    int seedA = 42, seedB = 52, seedC = 62, seedD = 72;
    std::string alignment_file = "../example/alignment.phy";
    std::string tree_file = "../example/tree.nwk";
    std::string log_file = "output.log";
    SeqType seq_type = SEQ_DNA;
    int numStates = 4;
    bool useEigenSpace = false; // Use eigendecomposition (IQ-TREE formulation)

    static Params& instance() {
        static Params instance;
        return instance;
    }


private:
    Params() = default; // private constructor
};
