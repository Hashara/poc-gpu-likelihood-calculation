//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_TREE_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_TREE_H


#include "Node.h"
#include "../alignment/Alignment.h"
#include "../model/Model.h"

class Tree {
public:
    Node* root = nullptr;

    ~Tree() {
        clear(root);
    }

    /** * Set the matrix Op type for Matrix operations.
     * @param matrixOpType The type of matrix operation to be used.
     * * This will determine how likelihood calculations are performed
     */
    void setMatrixOpType(MatrixOpType matrixOpType);

    /** * Compute the likelihood of the tree given an alignment and model.
     * @param aln The alignment containing sequences for each taxon.
     * @param model The substitution model to use for likelihood calculations.
     * @return The computed likelihood value.
     */
    double computeLikelihood(Alignment* aln, Model* model);

private:
    void clear(Node* node) {
        if (!node) return;
        for (Node* child : node->children)
            clear(child);
        delete node;
    }

};

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_TREE_H
