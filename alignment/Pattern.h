//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_PATTERN_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_PATTERN_H

#include <vector>
#include <cstdint>

using namespace std;


class Pattern {
public:
    int frequency;  // How many times this pattern occurs in the alignment
    std::vector<std::uint8_t> states;

    // Default constructor
    Pattern() : frequency(0) {}

    explicit Pattern(const std::vector<uint8_t>& v, int freq = 1)
            : states(v.begin(), v.end()), frequency(freq) {}


};

bool patternEqual(const Pattern& p1, const Pattern& p2);
void printPattern(const Pattern& p);

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_PATTERN_H
