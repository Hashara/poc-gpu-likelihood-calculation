//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "pattern.h"
#include <iostream>

using namespace std;

// Compare two Pattern objects (based on their string content)
bool patternEqual(const Pattern& p1, const Pattern& p2) {
    return static_cast<string>(p1) == static_cast<string>(p2);
}

// Print a Pattern object for debugging
void printPattern(const Pattern& p) {
    cout << "Pattern: " << static_cast<string>(p)
         << "  Frequency: " << p.frequency << endl;
}
