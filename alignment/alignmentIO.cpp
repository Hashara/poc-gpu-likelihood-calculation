//
// Created by Hashara Kumarasinghe on 11/7/2025.
//


#include <fstream>
#include <sstream>
#include <unordered_map>
#include "alignmentIO.h"

using namespace std;

// Map DNA bases to 0–3 (A, C, G, T)
char encodeDNA(char c) {
    switch (toupper(c)) {
        case 'A': return '0';
        case 'C': return '1';
        case 'G': return '2';
        case 'T': return '3';
        default:  return '4'; // unknown / N
    }
}

void readPhylipFile(const string& filename, Alignment& aln) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    size_t num_taxa, num_sites;
    infile >> num_taxa >> num_sites;
    aln.num_sites = 0;

    string name, sequence;
    vector<string> raw_seqs;

    // Read sequence lines
    for (size_t i = 0; i < num_taxa; ++i) {
        infile >> name >> sequence;
        aln.addTaxonName(name);
        raw_seqs.push_back(sequence);
    }

    unordered_map<string, int> pattern_map;

    for (size_t site = 0; site < num_sites; ++site) {
        string encoded_col;
        for (size_t t = 0; t < num_taxa; ++t) {
            encoded_col += encodeDNA(raw_seqs[t][site]);
        }
        pattern_map[encoded_col]++;
        aln.num_sites++;  // Every column contributes to total site count
    }

    // Convert map to patterns
    for (const auto& [pattern_str, freq] : pattern_map) {
        aln.addPattern(pattern_str, freq);
    }

    infile.close();
}