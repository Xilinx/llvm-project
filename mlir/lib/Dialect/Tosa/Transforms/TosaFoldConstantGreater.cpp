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

struct TosaFoldConstantGreater : public OpRewritePattern<GreaterOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GreaterOp op,
                                PatternRewriter &rewriter) const override {
    auto leftOp = op.getInput1();
    auto rightOp = op.getInput2();

    auto resultType = op.getType();
    auto lhsElemType = leftOp.getType().getElementType();
    auto rhsElemType = rightOp.getType().getElementType();
    if (lhsElemType != rhsElemType) {
      return rewriter.notifyMatchFailure(
          op, "Expected type of add arguments to match.");
    }

    // Check if both tensors are constant
    auto rhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(leftOp, op, rewriter);
    if (failed(rhsIsConstantCheck)) {
      return rhsIsConstantCheck;
    }
    auto lhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(rightOp, op, rewriter);
    if (failed(lhsIsConstantCheck)) {
      return lhsIsConstantCheck;
    }

    // Extract the tensor values
    DenseElementsAttr lhsValues;
    matchPattern(leftOp, m_Constant(&lhsValues));

    DenseElementsAttr rhsValues;
    matchPattern(rightOp, m_Constant(&rhsValues));

    if (!constantBinaryOpShouldBeFolded(op, lhsValues, rhsValues)) {
      return rewriter.notifyMatchFailure(
          op, "Currently, adds will only be folded if this requires only "
                 "little additional memory usage.");
    }

    DenseElementsAttr newTensor;
    if (isa<IntegerType>(lhsElemType)) {
      assert(isa<IntegerType>(rhsElemType) &&
             isa<IntegerType>(resultType.getElementType()));
      auto intAdd = [](const APInt &first, const APInt &second) {
        auto res = APInt(1, first.sgt(second));
        return res;
      };
      newTensor = applyElementWise<APInt, APInt>(lhsValues, rhsValues,
                                                 resultType, intAdd);
    } else {
      assert(isa<FloatType>(lhsElemType) && isa<FloatType>(rhsElemType) &&
             isa<IntegerType>(resultType.getElementType()));
      auto floatAdd = [](const APFloat &first, const APFloat &second) {
        return APInt(1, first > second);
      };
      newTensor = applyElementWise<APFloat, APInt>(lhsValues, rhsValues,
                                                     resultType, floatAdd);
    }
    rewriter.replaceOpWithNewOp<ConstOp>(op, newTensor.getType(), newTensor);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantGreaterPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantGreater>(ctx);
}
