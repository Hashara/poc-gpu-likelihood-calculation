#include <iostream>
#include "helper/Matrix.h"
#include "helper/MatrixOpFactory.h"

int main() {
    Matrix A(3, 3);
    A.fillRandom(42);  // Same values every run

    Matrix B(3, 3);
    B.fillRandom(52);  // Same values every run

    auto op = MatrixOpFactory::create(MatrixOpType::CPU);
    Matrix C = op->multiply(A, B);
    Matrix H = op->hadamard(A, B);

    std::cout << "Matrix A:\n";
    A.print();
    std::cout << "Matrix B:\n";
    B.print();
    std::cout << "Matrix C (A * B):\n";
    C.print();
    std::cout << "Matrix H (Hadamard product of A and B):\n";
    H.print();

}
