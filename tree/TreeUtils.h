//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_TREEUTILS_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_TREEUTILS_H

#pragma once
#include "Node.h"
#include <iostream>
#include <queue>

inline void printTree(Node* root) {
    if (!root) return;

    std::queue<Node*> q;
    q.push(root);

    while (!q.empty()) {
        Node* current = q.front();
        q.pop();

        std::cout << "Node: " << (current->name.empty() ? "[internal]" : current->name)
                  << ", Branch length: " << current->branchLength << "\n";

        for (Node* child : current->children) {
            q.push(child);
        }
    }
}

inline void collectLevels(Node* node, int depth, std::vector<std::vector<Node*>>& levels) {
    if (!node) return;
    if (levels.size() <= depth) levels.resize(depth + 1);
    levels[depth].push_back(node);

    for (Node* child : node->children) {
        collectLevels(child, depth + 1, levels);
    }
}

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_TREEUTILS_H
