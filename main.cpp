#include <iostream>
#include "helper/Matrix.h"

int main() {
    Matrix A(4, 3);
    A.fillRandom(42);  // Same values every run

    std::cout << "Seeded Random Matrix A:\n";
    A.print();
}
