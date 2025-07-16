//
// Created by Hashara Kumarasinghe on 11/7/2025.
//

#include "Tree.h"
#include "LikelihoodCalculator.h"

double Tree::computeLikelihood(Alignment *aln, Model *model) {
    LikelihoodCalculator calc(this, aln, model);
    return calc.computeLogLikelihood();
}

void Tree::setMatrixOpType(MatrixOpType matrixOpType) {
    Matrix::setMOpType(matrixOpType);
}
