//===- TosaFoldConstantCast.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA cast operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantCast : public OpRewritePattern<CastOp> {

  using OpRewritePattern::OpRewritePattern;

  static APFloat convertIntToFloat(const APInt &toConvert,
                                   FloatType targetType) {
    APFloat res(targetType.getFloatSemantics());
    res.convertFromAPInt(toConvert, true /* isSigned */, tosaRoundingMode);
    return res;
  }

  static APFloat convertFloatToFloat(const APFloat &toConvert,
                                     FloatType targetType) {
    APFloat res(toConvert);
    bool didLosePrecision;
    res.convert(targetType.getFloatSemantics(), tosaRoundingMode,
                &didLosePrecision);
    return res;
  }

  static APInt convertFloatToInt(const APFloat &toConvert,
                                 IntegerType targetType) {
    auto targetWidth = targetType.getIntOrFloatBitWidth();
    // Converting NaN to an integer results in an unpredictable value. Pick 0.
    if (toConvert.isNaN()) {
      return APInt::getZero(targetWidth);
    }

    // Make sure to properly translate booleans
    if (targetWidth == 1) {
      return toConvert.isZero() ? APInt::getZero(1) : APInt::getAllOnes(1);
    }

    // Use the built-in functionality of APFloats to convert to integers.
    // The result of this conversion should be an integer which might still be
    // outside of the target integer range.
    auto floatSize = APFloat::getSizeInBits(toConvert.getSemantics());
    APSInt converted(std::max(floatSize, targetWidth), targetType.isUnsigned());
    bool ignored = false;
    toConvert.convertToInteger(converted, APFloat::rmNearestTiesToEven,
                               &ignored);
    // Clip to allowed range.
    if (targetWidth < floatSize) {
      if (targetType.isUnsigned()) {
        return converted.truncUSat(targetWidth);
      }
      return converted.truncSSat(targetWidth);
    }
    return converted;
  }

  static APInt convertIntToInt(const APInt &toConvert, IntegerType targetType) {
    // Make sure to properly translate booleans
    if (targetType.getWidth() == 1) {
      return toConvert.isZero() ? APInt::getZero(1) : APInt::getAllOnes(1);
    }
    if (targetType.isUnsigned()) {
      return toConvert.zextOrTrunc(targetType.getIntOrFloatBitWidth());
    }
    return toConvert.sextOrTrunc(targetType.getIntOrFloatBitWidth());
  }

  static void warnAboutNaNToIntCast(DenseElementsAttr elements, CastOp location,
                                    PatternRewriter &rewriter) {
    // This is only relevant if the input values are float
    if (!isa<FloatType>(elements.getElementType())) {
      return;
    }
    // Check if it is an float to integer conversion
    auto resultType = location.getOutput().getType();
    if (!isa<IntegerType>(cast<TensorType>(resultType).getElementType())) {
      return;
    }

    // Report encountered NaNs
    auto checkNan = [](const APFloat &val) { return val.isNaN(); };
    if (any_of(elements.getValues<APFloat>(), checkNan)) {
      location->emitWarning(
          "Float tensor is casted to integer and it contains NaN values. The "
          "cast results in an unspecified value.");
    }
  }

  LogicalResult matchAndRewrite(CastOp tosaCast,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = tosaCast.getInput();

    // If the input tensor is not constant, we cannot fold it.
    if (failed(notifyIfNoTosaDenseConstantTensor(inputTensor, tosaCast,
                                                 rewriter))) {
      return failure();
    }

    auto fromType = inputTensor.getType().getElementType();
    auto toType = tosaCast.getOutput().getType().getElementType();

    DenseElementsAttr elements;
    matchPattern(inputTensor, m_Constant(&elements));

    // Issue a warning if we convert float -> int and NaNs are present; the
    // result value is unspecified in that case
    warnAboutNaNToIntCast(elements, tosaCast, rewriter);

    // Only fold splat tensors and those used only once to avoid duplicating
    // them.
    if (!inputTensor.hasOneUse() && !isa<SplatElementsAttr>(elements)) {
      return rewriter.notifyMatchFailure(tosaCast,
                                         "Currently, casts will only be folded "
                                         "if its input only has a single user");
    }

    // Report a match failure for unexpected types
    if (!toType.isIntOrFloat() || !fromType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          tosaCast, "Only casts from/to int/float are supported.");
    }

    auto isUnsigned = [](Type toCheck) {
      return isa<IntegerType>(toCheck) &&
             cast<IntegerType>(toCheck).isUnsigned();
    };
    auto typesToCheck = {toType, fromType};
    if (llvm::any_of(typesToCheck, isUnsigned)) {
      // TOSA casts currently don't support unsigned integers.
      // To support them by here, one could use APSInt instead of APInts,
      // however, this causes trouble with `getValues` which does not support
      // APSInts currently.
      return rewriter.notifyMatchFailure(
          tosaCast, "Cast folding from/to unsigned integers is not supported.");
    }

    DenseElementsAttr res;
    if (auto intOutTy = dyn_cast<IntegerType>(toType)) {
      if (isa<FloatType>(fromType)) {
        res = applyElementWise<APFloat, APInt, IntegerType>(
            elements, &convertFloatToInt, intOutTy);
      } else {
        assert(isa<IntegerType>(fromType));
        res = applyElementWise<APInt, APInt, IntegerType>(
            elements, &convertIntToInt, intOutTy);
      }
    } else {
      assert(isa<FloatType>(toType));
      auto floatOutTy = cast<FloatType>(toType);
      if (isa<FloatType>(fromType)) {
        res = applyElementWise<APFloat, APFloat, FloatType>(
            elements, &convertFloatToFloat, floatOutTy);
      } else {
        assert(isa<IntegerType>(fromType));
        res = applyElementWise<APInt, APFloat, FloatType>(
            elements, &convertIntToFloat, floatOutTy);
      }
    }

    rewriter.replaceOpWithNewOp<ConstOp>(tosaCast, res.getType(), res);
    return success();
  }
};

struct TosaFoldConstantFloatCasts : TosaFoldConstantCast {

  TosaFoldConstantFloatCasts(MLIRContext *ctx) : TosaFoldConstantCast(ctx) {}

  LogicalResult matchAndRewrite(CastOp tosaCast,
                                PatternRewriter &rewriter) const override {
    if (isa<IntegerType>(tosaCast.getInput().getType().getElementType())) {
      return rewriter.notifyMatchFailure(
          tosaCast, "Folding casts from int is currently disabled.");
    }

    return TosaFoldConstantCast::matchAndRewrite(tosaCast, rewriter);
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantCastPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, bool enableIntCastFolding) {
  if (enableIntCastFolding) {
    patterns.add<TosaFoldConstantCast>(ctx);
  } else {
    patterns.add<TosaFoldConstantFloatCasts>(ctx);
  }
}
