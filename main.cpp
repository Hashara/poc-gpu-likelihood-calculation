#include <iostream>
#include "helper/Matrix.h"
#include "alignment/alignmentIO.h"
#include "tree/TreeReader.h"
#include "model/ModelJC.h"
#include <string>
#include <iomanip>  // for std::setprecision
#include <chrono>
#include "helper/logger/Logger.h"


//#define VERBOSE

struct Params {
    int a_row, a_col, b_row, b_col;
    int seedA = 42, seedB = 52, seedC = 62, seedD = 72;
    std::string alignment_file = "../example/alignment.phy";
    std::string tree_file = "../example/tree.nwk";
    std::string log_file = "output.log";
};

Params parseArgs(int argc, char *argv[]) {
    Params params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" && i + 1 < argc) params.alignment_file = argv[++i];
        else if (arg == "-te" && i + 1 < argc) params.tree_file = argv[++i];
        else if (arg == "-prefix" && i + 1 < argc) params.log_file = argv[++i];
        }
    return params;
}

void printUsage(const char *progName) {
    std::cerr << "U sage: " << progName
              << "-s alignment -te tree.\n";
}


int main(int argc, char *argv[]) {
    try {
        auto params = parseArgs(argc, argv);

        initLogger(params.log_file);
        auto start = std::chrono::high_resolution_clock::now();
        Alignment aln;
        readPhylipFile(params.alignment_file, aln);
        auto end = std::chrono::high_resolution_clock::now();

        logInfo("Alignment loaded successfully.");
        std::chrono::duration<double> elapsed = end - start;
        logInfo("Time taken to read alignment: " + std::to_string(elapsed.count()) + " seconds");

        start = std::chrono::high_resolution_clock::now();
        TreeReader reader;
        Tree tree = reader.readFromFile(params.tree_file);
        end = std::chrono::high_resolution_clock::now();

        logInfo("Tree loaded successfully.");
        elapsed = end - start;
        logInfo("Time taken to read tree: " + std::to_string(elapsed.count()) + " seconds");

#ifndef USE_EIGEN
        MatrixOpType opType;
#endif
#ifdef VERBOSE
        aln.printAlignment();
        printTree(tree.root);
#endif
        ModelJC jc;
        logInfo("ModelJC initialized successfully.");


#ifdef USE_OPENACC
        MatrixOpType opType = MatrixOpType::OPENACC;
        logInfo("Using OpenACC for matrix operations.");
        std::cout << "Using OpenACC for matrix operations." << std::endl;
#elif defined USE_EIGEN
        logInfo("Using Eigen for matrix operations.");
        cout << "Using Eigen for matrix operations." << std::endl;
#else
        opType = MatrixOpType::CPU;
        logInfo("Using CPU for matrix operations.");
        std::cout << "Using CPU for matrix operations." << std::endl;

#endif

#ifndef USE_EIGEN
        tree.setMatrixOpType(opType);
#endif
        logInfo("Starting likelihood calculation...");
        cout << "Starting likelihood calculation..." << endl;
        start = std::chrono::high_resolution_clock::now();

        double logLikelihood = tree.computeLikelihood(&aln, &jc);

        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        logResult("CPU", aln.getNSeq(), aln.getNSites(), aln.patterns.size(), elapsed.count(), logLikelihood);
        std::cout << std::setprecision(18) << "Log-likelihood: " << logLikelihood << std::endl;
        std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
//        // Cleanup
//        delete jc;
//        delete tree;
//        delete aln;

        closeLogger();

    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}