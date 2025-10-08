//
// Created by Hashara Kumarasinghe on 11/7/2025.
//


#include <fstream>
#include <sstream>
#include <unordered_map>
#include "alignmentIO.h"
#include "../Params.h"
#include <cctype>

using namespace std;

unordered_map<char, int> char_map;

void initCharMap() {
    char_map.clear();

    if (Params::instance().seq_type == SEQ_DNA) {
        // DNA bases: 0..3 + 4 for unknown
        char_map['A'] = char_map['a'] = 0;
        char_map['C'] = char_map['c'] = 1;
        char_map['G'] = char_map['g'] = 2;
        char_map['T'] = char_map['t'] = 3;
        char_map['N'] = char_map['n'] = 4;
        char_map['-'] = 4;
    }
    else if (Params::instance().seq_type == SEQ_PROTEIN) {
        // Amino acids: 0..19 + 20 for unknown
        const string upper = "ARNDCQEGHILKMFPSTWYV";
        for (size_t i = 0; i < upper.size(); ++i) {
            char_map[upper[i]] = static_cast<int>(i);
            char_map[tolower(upper[i])] = static_cast<int>(i);
        }

        // Unknowns (char_map, B, Z, J, U, O, -, etc.)
        const std::string unknowns = "char_mapBZJUO-";
        for (char c : unknowns)
            char_map[c] = char_map[std::tolower(c)] = 20;
    }
}

struct VecIntHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v)
            h ^= std::hash<int>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct VecIntEq {
    bool operator()(const std::vector<int>& a, const std::vector<int>& b) const noexcept {
        return a == b;
    }
};



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

    initCharMap();
    unordered_map<vector<int>, int, VecIntHash, VecIntEq> pattern_map;

    for (size_t site = 0; site < num_sites; ++site) {
        vector<int> encoded_col(num_taxa);
        for (size_t t = 0; t < num_taxa; ++t) {
            encoded_col[t] = char_map[raw_seqs[t][site]];
        }
        pattern_map[encoded_col]++;
        aln.num_sites++;  // Every column contributes to total site count
    }

    // Convert map to patterns
    for (const auto& [pattern_vec, freq] : pattern_map) {
        aln.addPattern(pattern_vec, freq);
    }

    infile.close();
}