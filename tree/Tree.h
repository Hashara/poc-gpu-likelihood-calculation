//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_TREE_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_TREE_H


#include "Node.h"

class Tree {
public:
    Node* root = nullptr;

    ~Tree() {
        clear(root);
    }

private:
    void clear(Node* node) {
        if (!node) return;
        for (Node* child : node->children)
            clear(child);
        delete node;
    }
};

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_TREE_H
