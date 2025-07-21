#include <iostream>
#include "helper/Matrix.h"
#include "alignment/alignmentIO.h"
#include "tree/TreeReader.h"
#include "tree/TreeUtils.h"
#include "model/ModelJC.h"
#include "tree/LikelihoodCalculator.h"
#include <chrono>
#include <string>
#include <iomanip>  // for std::setprecision


//#define VERBOSE

struct Params {
    int a_row, a_col, b_row, b_col;
    int seedA = 42, seedB = 52, seedC = 62, seedD = 72;
    std::string alignment_file = "../example/alignment.phy";
    std::string tree_file = "../example/tree.nwk";
};

Params parseArgs(int argc, char *argv[]) {
    Params params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" && i + 1 < argc) params.alignment_file = argv[++i];
        else if (arg == "-te" && i + 1 < argc) params.tree_file = argv[++i];
    }
    return params;
}

void printUsage(const char *progName) {
    std::cerr << "Usage: " << progName
              << "-s alignment -te tree.\n";
}


int main(int argc, char *argv[]) {
    try {
        auto params = parseArgs(argc, argv);

        auto start = std::chrono::high_resolution_clock::now();
        Alignment aln;
        readPhylipFile(params.alignment_file, aln);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Alignment loaded successfully.\n";
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Time taken to read alignment: " << elapsed.count() << " seconds\n";

        start = std::chrono::high_resolution_clock::now();
        TreeReader reader;
        Tree tree = reader.readFromFile(params.tree_file);
        end = std::chrono::high_resolution_clock::now();

        std::cout << "Tree loaded successfully.\n";
        elapsed = end - start;
        std::cout << "Time taken to read tree: " << elapsed.count() << " seconds\n";



#ifdef VERBOSE
        aln.printAlignment();

        printTree(tree.root);
#endif
        ModelJC jc;


#ifdef USE_OPENACC
        //Tree tree2(MatrixOpType::OPENACC);
        tree.setMatrixOpType(MatrixOpType::OPENACC);
        std::cout << "Using OpenACC for matrix operations." << std::endl;
#elif defined USE_EIGEN
        cout << "Using Eigen for matrix operations." << std::endl;
#else
        //Tree tree2(MatrixOpType::CPU);
        tree.setMatrixOpType(MatrixOpType::CPU);
        std::cout << "Using CPU for matrix operations." << std::endl;

#endif
        cout << "Starting likelihood calculation..." << endl;
        start = std::chrono::high_resolution_clock::now();

        double logLikelihood = tree.computeLikelihood(&aln, &jc);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        std::cout << std::setprecision(18) << "Log-likelihood: " << logLikelihood << std::endl;
        std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
//        // Cleanup
//        delete jc;
//        delete tree;
//        delete aln;

    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}