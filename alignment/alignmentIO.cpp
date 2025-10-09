//
// Created by Hashara Kumarasinghe on 11/7/2025.
//


#include <fstream>
#include <sstream>
#include <unordered_map>
#include "alignmentIO.h"
#include "../Params.h"
#include <cctype>
#include <array>

using namespace std;

static std::array<std::uint8_t, 256> char_map; // use uint8_t to save space

void initCharMap() {
    char_map.fill(0xFF); // 0xFF for unknowns

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
            char_map[(unsigned char)upper[i]] = (uint8_t)i;
            char_map[(unsigned char)tolower(upper[i])] = (uint8_t)i;
        }

        // Unknowns (char_map, B, Z, J, U, O, -, etc.)
        const std::string unknowns = "char_mapBZJUO-";
        for (unsigned char c : unknowns)
            char_map[c] = char_map[std::tolower(c)] = 20;
    }
}

struct VecU8Hash {
    std::size_t operator()(const std::vector<std::uint8_t>& v) const noexcept {
        std::size_t h = 1469598103934665603ull; // FNV-1a 64-bit
        for (auto b : v) { h ^= b; h *= 1099511628211ull; }
        return h;
    }
};
struct VecU8Eq {
    bool operator()(const std::vector<std::uint8_t>& a,
                    const std::vector<std::uint8_t>& b) const noexcept { return a == b; }
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
//    unordered_map<vector<int>, int, VecIntHash, VecIntEq> pattern_map;

    unordered_map<std::vector<std::uint8_t>, int, VecU8Hash, VecU8Eq> pattern_map;
    pattern_map.reserve(num_sites * 2);

    std::vector<std::uint8_t> encoded_col; encoded_col.resize(num_taxa);

    for (std::size_t site = 0; site < num_sites; ++site) {
        for (std::size_t t = 0; t < num_taxa; ++t) {
            unsigned char ch = static_cast<unsigned char>(raw_seqs[t][site]);
            std::uint8_t code = char_map[ch];
            if (code == 0xFF) code = (Params::instance().seq_type == SEQ_DNA) ? 4 : 20;
            encoded_col[t] = code;
        }
        pattern_map[encoded_col]++;   // hashes bytes
        ++aln.num_sites;
    }
    for (const auto& [pat, freq] : pattern_map) {
        aln.addPattern(pat, freq);
    }

    infile.close();
}