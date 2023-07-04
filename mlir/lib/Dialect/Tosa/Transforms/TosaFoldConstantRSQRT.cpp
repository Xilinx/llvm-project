//===- TosaFoldConstantRSQRT.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA RSQRT (reciprocal square root) operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/IR/Constants.h>
#include <cmath>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantRSQRT : public OpRewritePattern<RsqrtOp> {

  using OpRewritePattern::OpRewritePattern;

  static APFloat computeRSQRT(const APFloat &apFloatVal, FloatType floatTy) {
    // The result for negative values (apart from zero) is always NaN
    if (apFloatVal.isNegative() && !apFloatVal.isNegZero()) {
      return APFloat::getNaN(apFloatVal.getSemantics());
    }

    // Compute the square root (APFloat unfortunately does not provide this
    // function, such that we need to unpack here)
    auto floatVal = apFloatVal.convertToFloat();
    auto sqrtVal = std::sqrt(floatVal);
    APFloat apSqrtVal(sqrtVal);

    // Compute the reciprocal
    return computeReciprocal(apSqrtVal, floatTy);
  }

  LogicalResult matchAndRewrite(RsqrtOp rsqrt,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = rsqrt.getInput1();

    // Reject non-float or non-dense tensors
    auto foldable =
        notifyIfNotConstantFloatTosaTensor(inputTensor, rsqrt, rewriter);
    if (failed(foldable)) {
      return foldable;
    }

    // Extract the tensor values
    DenseElementsAttr inputValues;
    matchPattern(inputTensor, m_Constant(&inputValues));

    // Check whether this should be folded.
    if (!constantUnaryOpShouldBeFolded(rsqrt, inputValues)) {
      return rewriter.notifyMatchFailure(
          rsqrt, "Currently, reciprocals will only be folded if the input "
                 "tensor has a single user");
    }

    // Create a new tensor with the updated values
    auto newTensor = applyElementWise<APFloat, APFloat, FloatType>(
        inputValues, &computeRSQRT,
        cast<FloatType>(inputValues.getElementType()));

    // Replace the use of the reciprocal with the transformed tensor
    rewriter.replaceOpWithNewOp<ConstOp>(rsqrt, newTensor.getType(), newTensor);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantRSQRTPatterns(

    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantRSQRT>(ctx);
}
