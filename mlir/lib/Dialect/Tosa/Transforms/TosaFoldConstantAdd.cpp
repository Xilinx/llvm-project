//===- TosaFoldConstantAdd.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA Add operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantAdd : public OpRewritePattern<AddOp> {

  using OpRewritePattern::OpRewritePattern;

  static APInt computeIntAdd(const APInt &first, const APInt &second) {
    return first.sadd_sat(second);
  }

  static APFloat computeFloatAdd(const APFloat &first, const APFloat &second) {
    return first + second;
  }

  LogicalResult matchAndRewrite(AddOp addOp,
                                PatternRewriter &rewriter) const override {
    auto leftOp = addOp.getInput1();
    auto rightOp = addOp.getInput2();

    auto resultType = addOp.getType();
    auto lhsElemType = leftOp.getType().getElementType();
    auto rhsElemType = rightOp.getType().getElementType();
    if (lhsElemType != rhsElemType) {
      return rewriter.notifyMatchFailure(
          addOp, "Expected type of add arguments to match.");
    }

    // Check if both tensors are constant
    auto rhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(leftOp, addOp, rewriter);
    if (failed(rhsIsConstantCheck)) {
      return rhsIsConstantCheck;
    }
    auto lhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(rightOp, addOp, rewriter);
    if (failed(lhsIsConstantCheck)) {
      return lhsIsConstantCheck;
    }

    // Extract the tensor values
    DenseElementsAttr lhsValues;
    matchPattern(leftOp, m_Constant(&lhsValues));

    DenseElementsAttr rhsValues;
    matchPattern(rightOp, m_Constant(&rhsValues));

    if (!constantBinaryOpShouldBeFolded(addOp, lhsValues, rhsValues)) {
      return rewriter.notifyMatchFailure(
          addOp, "Currently, adds will only be folded if this requires only "
                 "little additional memory usage.");
    }

    DenseElementsAttr newTensor;
    if (isa<IntegerType>(lhsElemType)) {
      assert(isa<IntegerType>(rhsElemType) &&
             isa<IntegerType>(resultType.getElementType()));
      newTensor = applyElementWise<APInt, APInt, APInt>(
          lhsValues, rhsValues, resultType, &computeIntAdd);
    } else {
      assert(isa<FloatType>(lhsElemType) && isa<FloatType>(rhsElemType) &&
             isa<FloatType>(resultType.getElementType()));
      newTensor = applyElementWise<APFloat, APFloat, APFloat>(
          lhsValues, rhsValues, resultType, &computeFloatAdd);
    }
    rewriter.replaceOpWithNewOp<ConstOp>(addOp, newTensor.getType(), newTensor);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantAddPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantAdd>(ctx);
}
