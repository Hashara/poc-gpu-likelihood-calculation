#include <iostream>
#include "helper/Matrix.h"
#include "alignment/alignmentIO.h"
#include "tree/TreeReader.h"
#include "tree/TreeUtils.h"
#include "model/ModelJC.h"
#include "tree/LikelihoodCalculator.h"
#include <chrono>
#include <string>

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
        if (arg == "-s" && i + 1 < argc) params.alignment_file = std::stoi(argv[++i]);
        else if (arg == "-te" && i + 1 < argc) params.tree_file = std::stoi(argv[++i]);
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

        Alignment aln;
        readPhylipFile(params.alignment_file, aln);
        aln.printAlignment();

        TreeReader reader;
        Tree tree = reader.readFromFile(params.tree_file);
        std::cout << "Tree loaded successfully.\n";
        printTree(tree.root);

        ModelJC jc;


#ifdef USE_OPENACC
//        auto op = MatrixOpFactory::create(MatrixOpType::OPENACC);
        //Tree tree2(MatrixOpType::OPENACC);
        tree.setMatrixOpType(MatrixOpType::OPENACC);
#else
//        auto op = MatrixOpFactory::create(MatrixOpType::CPU);
        //Tree tree2(MatrixOpType::CPU);
        tree.setMatrixOpType(MatrixOpType::CPU);

#endif
        //tree2.readFromFile(params.tree_file);
//        tree.computeLikelihood(&aln, &jc);
        // double loglk = tree2.computeLogLikelihood(aln, jc);
        double logLikelihood = tree.computeLikelihood(&aln, &jc);

        std::cout << "Log-likelihood: " << logLikelihood << std::endl;

//        // Cleanup
//        delete jc;
//        delete tree;
//        delete aln;






        // commenting out the matrix operations for now
       /* Matrix A(params.a_row, params.a_col);
        Matrix C(params.a_row, params.a_col); // create C with A dimensions
        Matrix B(params.b_row, params.b_col);
        Matrix D(params.b_row, params.b_col); // create D with B dimensions

        A.fillRandom(params.seedA);
        B.fillRandom(params.seedB);
        C.fillRandom(params.seedC);
        D.fillRandom(params.seedD);

#ifdef USE_OPENACC
        auto op = MatrixOpFactory::create(MatrixOpType::OPENACC);
#else
        auto op = MatrixOpFactory::create(MatrixOpType::CPU);
#endif

        // start timer
        auto t0 = std::chrono::high_resolution_clock::now();

        Matrix AB = op->multiply(A, B);
        Matrix CD = op->multiply(C, D);
        Matrix H = op->hadamard(AB, CD);
//        Matrix H = op->hadamard(A, B);

        // stop timer
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::milli> ms = t1 - t0;

        // output timing
        std::cout << "Time taken: " << ms.count() << " ms\n";

*/
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}