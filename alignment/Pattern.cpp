//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "Pattern.h"
#include <iostream>

using namespace std;

// Compare two Pattern objects (based on their string content)
bool patternEqual(const Pattern& p1, const Pattern& p2) {
    return p1.states == p2.states;
}

// Print a Pattern object for debugging
void printPattern(const Pattern& p) {
    for (size_t i = 0; i < p.states.size(); ++i) {
        std::cout << int(p.states[i]);
        if (i + 1 < p.states.size()) std::cout << ' ';
    }
    std::cout << "  x" << p.frequency << '\n';
}
