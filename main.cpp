#include <iostream>
#include "helper/Matrix.h"
#include "helper/MatrixOpFactory.h"
#include "alignment/AlignmentIO.h"
#include <chrono>
#include <string>

struct Params {
    int a_row, a_col, b_row, b_col;
    int seedA = 42, seedB = 52, seedC = 62, seedD = 72;
    std::string alignment_file = "../example/alignment.phy";
};

Params parseArgs(int argc, char *argv[]) {
    Params params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--a_row" && i + 1 < argc) params.a_row = std::stoi(argv[++i]);
        else if (arg == "--a_col" && i + 1 < argc) params.a_col = std::stoi(argv[++i]);
        else if (arg == "--b_row" && i + 1 < argc) params.b_row = std::stoi(argv[++i]);
        else if (arg == "--b_col" && i + 1 < argc) params.b_col = std::stoi(argv[++i]);
        else if (arg == "--seedA" && i + 1 < argc) params.seedA = std::stoi(argv[++i]);
        else if (arg == "--seedB" && i + 1 < argc) params.seedB = std::stoi(argv[++i]);
        else {
            throw std::invalid_argument("Unknown or malformed flag: " + arg);
        }
    }
    if (params.a_row < 1 || params.a_col < 1 || params.b_row < 1 || params.b_col < 1)
        throw std::invalid_argument("Must specify positive dimensions for both A and B.");
    if (params.a_col != params.b_row)
        throw std::invalid_argument(
                "A’s columns (" + std::to_string(params.a_col) +
                ") must equal B’s rows (" + std::to_string(params.b_row) + ").");
    return params;
}

void printUsage(const char *progName) {
    std::cerr << "Usage: " << progName
              << " --a_row N --a_col M --b_row P --b_col Q"
                 " [--seedA S] [--seedB T]\n"
                 "  Requires M == P.\n";
}


int main(int argc, char *argv[]) {
    try {
        auto params = parseArgs(argc, argv);

        Alignment aln;
        readPhylipFile(params.alignment_file, aln);
        aln.printAlignment();
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