//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#ifndef POC_GPU_LIKELIHOOD_CALCULATIONS_ALIGNMENT_H
#define POC_GPU_LIKELIHOOD_CALCULATIONS_ALIGNMENT_H

#include <iostream>
#include <string>
#include <vector>

#include "Pattern.h"

using namespace std;

class Alignment {

public:
    vector<string> seq_names;
    size_t num_sites;
    vector<Pattern> patterns;
    size_t num_taxa = 0;

    inline size_t getNSeq()  {
        return seq_names.size();
    }

    inline size_t getNSites() {
        return num_sites;
    }
/**
     * @brief Add a taxon name to the alignment.
     *
     * @param name Taxon/sequence name
     */
    void addTaxonName(const string& name) {
        seq_names.push_back(name);
        num_taxa = seq_names.size();
    }

    /**
     * @brief Add a unique site pattern.
     *
     * @param pattern_str Encoded sequence at a column (e.g., "0123")
     * @param freq How many times this pattern appears
     */
    void addPattern(const string& pattern_str, int freq = 1) {
        patterns.emplace_back(pattern_str, freq);
    }

    /**
     * @brief Print alignment metadata and patterns (for debugging).
     */
    void printAlignment() const {
        cout << "Alignment: " << num_taxa << " taxa, "
             << num_sites << " sites, "
             << patterns.size() << " unique patterns" << endl;

        for (const auto& p : patterns) {
            printPattern(p);
        }
    }
};


#endif //POC_GPU_LIKELIHOOD_CALCULATIONS_ALIGNMENT_H
