//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "TreeReader.h"
#include <fstream>
#include <sstream>
#include <cctype>
#include <stdexcept>

Tree TreeReader::readFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open tree file: " + filename);

    std::ostringstream ss;
    ss << file.rdbuf();
    newick = ss.str();
    pos = 0;

    Tree tree;
    tree.root = parseSubtree(nullptr);

    skipWhitespace();
    if (newick[pos] != ';') {
        throw std::runtime_error("Expected ';' at end of Newick string");
    }

    return tree;
}

Node* TreeReader::parseSubtree(Node* parent) {
    skipWhitespace();
    Node* node = new Node;
    node->parent = parent;

    if (newick[pos] == '(') {
        ++pos;  // skip '('
        while (true) {
            Node* child = parseSubtree(node);
            node->children.push_back(child);
            skipWhitespace();
            if (newick[pos] == ',') {
                ++pos;
            } else if (newick[pos] == ')') {
                ++pos;
                break;
            } else {
                throw std::runtime_error("Expected ',' or ')' in Newick tree");
            }
        }
    }

    skipWhitespace();
    node->name = parseName();
    skipWhitespace();

    if (newick[pos] == ':') {
        ++pos;
        node->branchLength = parseBranchLength();
    }

    return node;
}

std::string TreeReader::parseName() {
    skipWhitespace();
    std::string name;
    while (pos < newick.size() && newick[pos] != ':' && newick[pos] != ',' && newick[pos] != ')' && newick[pos] != ';') {
        name += newick[pos++];
    }
    return name;
}

double TreeReader::parseBranchLength() {
    skipWhitespace();
    std::string num;
    while (pos < newick.size() && (isdigit(newick[pos]) || newick[pos] == '.' || newick[pos] == 'e' || newick[pos] == '-' || newick[pos] == '+')) {
        num += newick[pos++];
    }
    return std::stod(num);
}

void TreeReader::skipWhitespace() {
    while (pos < newick.size() && isspace(newick[pos])) ++pos;
}
