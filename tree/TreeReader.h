//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_TREEREADER_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_TREEREADER_H


#include "Tree.h"
#include <string>

class TreeReader {
public:
    Tree readFromFile(const std::string& filename);

private:
    size_t pos = 0;
    std::string newick;

    Node* parseSubtree(Node* parent);
    std::string parseName();
    double parseBranchLength();
    void skipWhitespace();
};

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_TREEREADER_H
