#include <gtest/gtest.h>
#include "MatrixUtils.h"
#include "MatrixOpCPU.h"

#ifdef USE_OPENACC
#include "MatrixOpOpenACC.h"
#endif

#define VERBOSE 1

template <typename T>
class MatrixOpTest : public ::testing::Test {};

using Implementations = ::testing::Types<MatrixOpCPU
#ifdef USE_OPENACC
        , MatrixOpOpenACC
#endif
>;

TYPED_TEST_SUITE(MatrixOpTest, Implementations);

TYPED_TEST(MatrixOpTest, MultiplyBasic) {
    Matrix A(2, 3), B(3, 2);
    A.fill(1.0);
    B.fill(2.0);

    TypeParam op;
    Matrix C = op.multiply(A, B);

#if VERBOSE
    std::cout << "Matrix A:\n"; A.print();
    std::cout << "Matrix B:\n"; B.print();
    std::cout << "Matrix C (A * B):\n"; C.print();
#endif

    for (size_t i = 0; i < C.rows(); ++i)
        for (size_t j = 0; j < C.cols(); ++j)
            EXPECT_DOUBLE_EQ(C.at(i, j), 6.0);
}

TYPED_TEST(MatrixOpTest, HadamardBasic) {
    Matrix A(2, 2), B(2, 2);
    A.fill(3.0); B.fill(4.0);

    TypeParam op;
    Matrix C = op.hadamard(A, B);

#if VERBOSE
    std::cout << "Matrix C (Hadamard):\n"; C.print();
#endif

    for (size_t i = 0; i < C.rows(); ++i)
        for (size_t j = 0; j < C.cols(); ++j)
            EXPECT_DOUBLE_EQ(C.at(i, j), 12.0);
}

TYPED_TEST(MatrixOpTest, MultiplyInvalidDimsThrows) {
    Matrix A(2, 3), B(4, 2);
    TypeParam op;
    EXPECT_THROW(op.multiply(A, B), std::invalid_argument);
}

TYPED_TEST(MatrixOpTest, HadamardInvalidDimsThrows) {
    Matrix A(2, 2), B(3, 2);
    TypeParam op;
    EXPECT_THROW(op.hadamard(A, B), std::invalid_argument);
}

TYPED_TEST(MatrixOpTest, CompositeHadamard) {
    Matrix A(2, 2), B(2, 2), C(2, 2), D(2, 2);
    A.fill(1.0); B.fill(2.0); C.fill(3.0); D.fill(4.0);

    TypeParam op;
    Matrix result = compositeHadamard(op, A, B, C, D);

#if VERBOSE
    std::cout << "Result (compositeHadamard):\n"; result.print();
#endif

    for (size_t i = 0; i < result.rows(); ++i)
        for (size_t j = 0; j < result.cols(); ++j)
            EXPECT_DOUBLE_EQ(result.at(i, j), 96.0);
}
