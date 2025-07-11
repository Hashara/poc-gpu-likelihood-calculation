//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_PATTERN_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_PATTERN_H

#include <vector>
#include <cstdint>
#include <string>

using namespace std;


class Pattern : public string {
public:
    int frequency;  // How many times this pattern occurs in the alignment

    // Default constructor
    Pattern() : string(), frequency(0) {}

    // Constructor with pattern and optional frequency
    Pattern(const string& pattern, int freq = 1)
            : string(pattern), frequency(freq) {}

};

bool patternEqual(const Pattern& p1, const Pattern& p2);
void printPattern(const Pattern& p);

#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_PATTERN_H
