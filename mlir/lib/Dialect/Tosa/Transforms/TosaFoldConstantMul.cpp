//===- TosaFoldConstantMul.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA Mul operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <llvm/ADT/APInt.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantMul : public OpRewritePattern<MulOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (mulOp.getShift() > 0) {
      return rewriter.notifyMatchFailure(
          mulOp, "Non-zero shift folding is currently not implemented.");
    }

    auto leftOp = mulOp.getInput1();
    auto rightOp = mulOp.getInput2();

    // Check if both tensors are constant
    auto rhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(leftOp, mulOp, rewriter);
    if (failed(rhsIsConstantCheck)) {
      return rhsIsConstantCheck;
    }
    auto lhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(rightOp, mulOp, rewriter);
    if (failed(lhsIsConstantCheck)) {
      return lhsIsConstantCheck;
    }

    // Extract the tensor values
    DenseElementsAttr lhsValues;
    matchPattern(leftOp, m_Constant(&lhsValues));

    DenseElementsAttr rhsValues;
    matchPattern(rightOp, m_Constant(&rhsValues));

    if (!constantBinaryOpShouldBeFolded(mulOp, lhsValues, rhsValues)) {
      return rewriter.notifyMatchFailure(
          mulOp, "Currently, muls will only be folded if this requires only "
                 "little additional memory usage.");
    }

    DenseElementsAttr newTensor;

    auto lhsElemType = leftOp.getType().getElementType();
    auto rhsElemType = rightOp.getType().getElementType();
    assert(lhsElemType == rhsElemType);

    auto resultType = mulOp.getType();
    auto resultElementType = resultType.getElementType();
    if (isa<IntegerType>(lhsElemType)) {
      assert(isa<IntegerType>(rhsElemType) &&
             isa<IntegerType>(resultElementType));
      auto resultElementWidth = resultElementType.getIntOrFloatBitWidth();
      assert(resultElementWidth >= lhsElemType.getIntOrFloatBitWidth() &&
             "The multiplication is expected to have an at least as big output "
             "as input type");

      // Compute the multiplication and track if an overflow occurred to enable
      // emitting a warning
      bool mulOverflowed = false;
      auto intMulFun = [&resultElementWidth, &mulOverflowed](
                           const APInt &first, const APInt &second) {
        bool didOverflow;
        auto res = first.sext(resultElementWidth)
                       .smul_ov(second.sext(resultElementWidth), didOverflow);
        mulOverflowed |= didOverflow;
        return res;
      };
      newTensor = applyElementWise<APInt, APInt>(lhsValues, rhsValues,
                                                 resultType, intMulFun);
      if (mulOverflowed) {
        mulOp.emitWarning(
            "Multiplication did overflow. The results are unspecified.");
      }
    } else {
      assert(isa<FloatType>(lhsElemType) && isa<FloatType>(rhsElemType) &&
             isa<FloatType>(resultType.getElementType()));
      auto mulFun = [](const APFloat &first, const APFloat &second) {
        return first * second;
      };
      newTensor = applyElementWise<APFloat, APFloat>(lhsValues, rhsValues,
                                                     resultType, mulFun);
    }
    rewriter.replaceOpWithNewOp<ConstOp>(mulOp, newTensor.getType(), newTensor);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantMulPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantMul>(ctx);
}
