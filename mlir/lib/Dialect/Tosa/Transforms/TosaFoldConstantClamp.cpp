//===- TosaFoldConstantClamp.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA Clamp operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantClamp : public OpRewritePattern<ClampOp> {

  using OpRewritePattern::OpRewritePattern;

  static void
  changeSemanticsLossless(APFloat &floatVal,
                          const llvm::fltSemantics *floatSemantics) {
    bool losesInfo;
    floatVal.convert(*floatSemantics, tosaRoundingMode, &losesInfo);
    assert(!losesInfo);
  }

  DenseElementsAttr applyClamp(DenseElementsAttr inputValues,
                               const APInt &lowerBound, const APInt &upperBound,
                               TensorType resultType) const {

    // Determine the width for the APInt comparison
    auto comparisonWidth =
        std::max(inputValues.getElementType().getIntOrFloatBitWidth(),
                 lowerBound.getBitWidth());
    // Sign-extend the upper and lower bound
    auto extUpperBound = upperBound.sext(comparisonWidth);
    auto extLowerBound = lowerBound.sext(comparisonWidth);

    // Determine the result type
    auto resultingIntType = cast<IntegerType>(resultType.getElementType());

    // Lambda to perform the clamp
    auto clampFun = [&extLowerBound, &extUpperBound,
                     &comparisonWidth](const APInt &val, IntegerType type) {
      auto clampedUpper =
          llvm::APIntOps::smin(val.sext(comparisonWidth), extUpperBound);
      auto fullyClamped = llvm::APIntOps::smax(clampedUpper, extLowerBound);
      assert(type.getWidth() >= fullyClamped.getSignificantBits());
      return fullyClamped.trunc(type.getWidth());
    };
    auto newTensor = applyElementWise<APInt, APInt, IntegerType>(
        inputValues, clampFun, resultingIntType);

    return newTensor;
  }

  DenseElementsAttr applyClamp(DenseElementsAttr inputValues,
                               APFloat lowerBound, APFloat upperBound,
                               TensorType resultType) const {
    auto inputValType = cast<FloatType>(inputValues.getElementType());
    auto inputWidth = inputValType.getWidth();
    auto bWidth = APFloat::semanticsSizeInBits(lowerBound.getSemantics());
    auto *comparisonSem = inputWidth < bWidth
                              ? &lowerBound.getSemantics()
                              : &inputValType.getFloatSemantics();

    changeSemanticsLossless(lowerBound, comparisonSem);
    changeSemanticsLossless(upperBound, comparisonSem);

    auto resultingFloatType = cast<FloatType>(resultType.getElementType());

    // Ensure that the value is larger than the lower bound and smaller than the
    // upper bound
    auto clampFun = [&lowerBound, &upperBound, &comparisonSem](APFloat val,
                                                               FloatType type) {
      if (val.isNaN()) {
        return APFloat::getNaN(type.getFloatSemantics());
      }
      changeSemanticsLossless(val, comparisonSem);
      auto clampedUpper = val < upperBound ? val : upperBound;
      auto fullyClamped = clampedUpper < lowerBound ? lowerBound : clampedUpper;
      changeSemanticsLossless(fullyClamped, &type.getFloatSemantics());
      return fullyClamped;
    };
    auto newTensor = applyElementWise<APFloat, APFloat, FloatType>(
        inputValues, clampFun, resultingFloatType);

    return newTensor;
  }

  LogicalResult matchAndRewrite(ClampOp clampOp,
                                PatternRewriter &rewriter) const override {
    auto valsToClamp = clampOp.getInput();
    auto inputElementType = valsToClamp.getType().getElementType();

    // Check if the input is constant
    if (failed(notifyIfNoTosaDenseConstantTensor(valsToClamp, clampOp,
                                                 rewriter))) {
      return failure();
    }

    if (isa<IntegerType>(inputElementType) &&
        cast<IntegerType>(inputElementType).isUnsigned()) {
      return rewriter.notifyMatchFailure(
          clampOp, "Currently, unsigned integer clamps are unsupported.");
    }

    // Extract the tensor values
    DenseElementsAttr inputValues;
    matchPattern(valsToClamp, m_Constant(&inputValues));

    if (!constantUnaryOpShouldBeFolded(clampOp, inputValues)) {
      return rewriter.notifyMatchFailure(
          clampOp,
          "Currently, clamps will only be folded if this requires only "
          "little additional memory usage.");
    }

    // Apply the clamp to all values of the int/float tensor
    auto resultType = clampOp.getType();
    DenseElementsAttr newTensor;
    if (isa<IntegerType>(inputElementType)) {
      auto lowerBoundVal = clampOp.getMinIntAttr().getValue();
      auto upperBoundVal = clampOp.getMaxIntAttr().getValue();
      assert(lowerBoundVal.getBitWidth() == upperBoundVal.getBitWidth());

      newTensor =
          applyClamp(inputValues, lowerBoundVal, upperBoundVal, resultType);
    } else {
      assert(isa<FloatType>(inputElementType));
      auto lowerBoundVal = clampOp.getMinFp();
      auto upperBoundVal = clampOp.getMaxFp();
      assert(APFloat::getSizeInBits(lowerBoundVal.getSemantics()) ==
             APFloat::getSizeInBits(upperBoundVal.getSemantics()));

      newTensor =
          applyClamp(inputValues, lowerBoundVal, upperBoundVal, resultType);
    }

    rewriter.replaceOpWithNewOp<ConstOp>(clampOp, newTensor.getType(),
                                         newTensor);

    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantClampPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantClamp>(ctx);
}
