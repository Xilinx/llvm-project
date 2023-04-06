//===- TosaFoldConstantPow.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA Pow operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <cmath>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantPow : public OpRewritePattern<PowOp> {

  using OpRewritePattern::OpRewritePattern;

  static APFloat computePower(const APFloat &base, const APFloat &exp) {
    // Propagate NaN
    if (base.isNaN() || exp.isNaN()) {
      return APFloat::getNaN(base.getSemantics());
    }
    // TOSA defines 0.0**0.0 as NaN
    if (base.isZero() && exp.isZero()) {
      return APFloat::getNaN(base.getSemantics());
    }
    // In case the value is negative, the exponent needs to be an integer
    if (base.isNegative() && !base.isZero()) {
      if (!exp.isInteger()) {
        return APFloat::getNaN(base.getSemantics());
      }
    }

    // Actually compute base**exp. Special cases for [-]infinity and [-]0 are
    // already handled in accordance with the TOSA spec.
    auto powFloat = std::pow(base.convertToFloat(), exp.convertToFloat());
    auto res = APFloat(powFloat);

    bool lostPrecision;
    res.convert(base.getSemantics(), APFloat::rmNearestTiesToEven,
                &lostPrecision);
    return res;
  }

  LogicalResult matchAndRewrite(PowOp powOp,
                                PatternRewriter &rewriter) const override {
    auto baseOp = powOp.getInput1();
    auto expOp = powOp.getInput2();

    if (baseOp.getType().getElementType() != expOp.getType().getElementType()) {
      return rewriter.notifyMatchFailure(
          powOp, "Expected type of pow arguments to match.");
    }

    // Check if both tensors are constant
    auto baseIsConstCheck =
        notifyIfNotConstantFloatTosaTensor(baseOp, powOp, rewriter);
    if (failed(baseIsConstCheck)) {
      return baseIsConstCheck;
    }
    auto expIsConstCheck =
        notifyIfNotConstantFloatTosaTensor(expOp, powOp, rewriter);
    if (failed(expIsConstCheck)) {
      return expIsConstCheck;
    }

    // Extract the tensor values
    DenseElementsAttr baseValues;
    matchPattern(baseOp, m_Constant(&baseValues));

    DenseElementsAttr expValues;
    matchPattern(expOp, m_Constant(&expValues));

    if (!constantBinaryOpShouldBeFolded(powOp, baseValues, expValues)) {
      return rewriter.notifyMatchFailure(
          powOp, "Currently, pows will only be folded if this requires only "
                 "little additional memory usage.");
    }

    auto newTensor = applyElementWise<APFloat, APFloat>(
        baseValues, expValues, powOp.getType(), &computePower);
    rewriter.replaceOpWithNewOp<ConstOp>(powOp, newTensor.getType(), newTensor);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantPowPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantPow>(ctx);
}
