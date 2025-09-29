//
// Created by Hashara Kumarasinghe on 11/7/2025.
//


#include <fstream>
#include <sstream>
#include <unordered_map>
#include "alignmentIO.h"
#include "../Params.h"

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

// Map AA based to 0-19
char encodeAA(char c) {
    switch (toupper(c)) {
        case 'A': return '0';  // Alanine
        case 'R': return '1';  // Arginine
        case 'N': return '2';  // Asparagine
        case 'D': return '3';  // Aspartic acid
        case 'C': return '4';  // Cysteine
        case 'Q': return '5';  // Glutamine
        case 'E': return '6';  // Glutamic acid
        case 'G': return '7';  // Glycine
        case 'H': return '8';  // Histidine
        case 'I': return '9';  // Isoleucine
        case 'L': return 'A';  // Leucine
        case 'K': return 'B';  // Lysine
        case 'M': return 'C';  // Methionine
        case 'F': return 'D';  // Phenylalanine
        case 'P': return 'E';  // Proline
        case 'S': return 'F';  // Serine
        case 'T': return 'G';  // Threonine
        case 'W': return 'H';  // Tryptophan
        case 'Y': return 'I';  // Tyrosine
        case 'V': return 'J';  // Valine
        default:  return 'Z';  // Unknown (e.g., X, B, Z, gaps)
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

    if (Params::instance().seq_type == SEQ_PROTEIN) {
        for (size_t site = 0; site < num_sites; ++site) {
            string encoded_col;
            for (size_t t = 0; t < num_taxa; ++t) {
                encoded_col += encodeAA(raw_seqs[t][site]);
            }
            pattern_map[encoded_col]++;
            aln.num_sites++;  // Every column contributes to total site count
        }
    } else {
        for (size_t site = 0; site < num_sites; ++site) {
            string encoded_col;
            for (size_t t = 0; t < num_taxa; ++t) {
                encoded_col += encodeDNA(raw_seqs[t][site]);
            }
            pattern_map[encoded_col]++;
            aln.num_sites++;  // Every column contributes to total site count
        }
    }


    // Convert map to patterns
    for (const auto& [pattern_str, freq] : pattern_map) {
        aln.addPattern(pattern_str, freq);
    }

    infile.close();
}