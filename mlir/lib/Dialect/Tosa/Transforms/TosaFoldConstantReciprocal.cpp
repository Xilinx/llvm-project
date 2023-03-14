//===- TosaFoldConstantReciprocal.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA Reciprocal operation on constant data
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/FloatingPointMode.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {

struct TosaFoldConstantReciprocal : public OpRewritePattern<ReciprocalOp> {

  using OpRewritePattern::OpRewritePattern;
  static constexpr llvm::RoundingMode reciprocalRoundingMode =
      APFloat::rmNearestTiesToEven;

  APFloat computeReciprocal(const APFloat &floatVal, Type floatTy) const {
    auto recipAttr = FloatAttr::get(floatTy, 1.0);
    APFloat recip = recipAttr.getValue();
    recip.divide(floatVal, reciprocalRoundingMode);

    return recip;
  }

  DenseElementsAttr
  replaceTensorWithReciprocal(ConstOp tensorToReplace,
                              const DenseElementsAttr &inputValues) const {
    // TODO it would be nicer to do this in-place

    // Compute the reciprocal for each tensor element
    llvm::SmallVector<APFloat, 1> transformedValues;
    // We already know the amount of values we will insert, reserve space for
    // all of them to avoid dynamic resizing
    transformedValues.reserve(inputValues.getNumElements());
    for (auto val : inputValues.getValues<APFloat>()) {
      auto recipVal = computeReciprocal(val, inputValues.getElementType());
      transformedValues.push_back(recipVal);
    }

    // Replace the current tensor with one containing the computed reciprocals
    auto newTensor =
        DenseElementsAttr::get(inputValues.getType(), transformedValues);
    return newTensor;
  }

  LogicalResult matchAndRewrite(ReciprocalOp recip,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = recip.getInput1();
    auto elemType = inputTensor.getType().getElementType();
    // TOSA only allows for floats as inputs to the reciprocal operation, so
    // bail if anything else is contained
    if (!isa<FloatType>(elemType)) {
      return rewriter.notifyMatchFailure(recip,
                                         "Unexpected input tensor type: the "
                                         "TOSA spec only allows floats");
    }

    // Check whether the tensor is constant and dense
    DenseElementsAttr inputValues;
    if (!matchPattern(inputTensor, m_Constant(&inputValues))) {
      return rewriter.notifyMatchFailure(
          recip, "Non-const or non-dense input to reciprocal");
    }

    // In case we have a splat, we only need to calculate the reciprocal once
    // and update the tensor to the transformed splat value.
    if (auto splatAttrs = dyn_cast<SplatElementsAttr>(inputValues)) {
      // Transform the splat value
      auto splatVal = splatAttrs.getSplatValue<APFloat>();
      auto newSplatRecipAttr = computeReciprocal(splatVal, elemType);

      // Create a tensor with the transformed splat value
      auto newSplatTensor =
          DenseElementsAttr::get(splatAttrs.getType(), newSplatRecipAttr);

      // Replace the reciprocal op with the newly constructed tensor
      rewriter.replaceOpWithNewOp<ConstOp>(recip, newSplatTensor.getType(),
                                           newSplatTensor);
      return success();
    }

    if (!isa<ConstOp>(inputTensor.getDefiningOp())) {
      return rewriter.notifyMatchFailure(recip,
                                         "The reciprocal can only be folded if "
                                         "it operates on a TOSA constant");
    }
    auto definingConstOp = cast<ConstOp>(inputTensor.getDefiningOp());

    // Our transformation replaces the input tensor with the transformed tensor.
    // If the input has several users we need to keep the input. This can
    // result in a significantly increased memory usage, such that we currently
    // refrain from applying the transformation in that case.
    if (!definingConstOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(
          recip, "Currently, reciprocals will only be folded if the input "
                 "tensor has a single user");
    }

    // Create a new tensor with the updated values
    auto newTensor = replaceTensorWithReciprocal(definingConstOp, inputValues);

    // Replace the use of the reciprocal with the transformed tensor
    rewriter.replaceOpWithNewOp<ConstOp>(recip, newTensor.getType(), newTensor);
    return success();
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantReciprocalPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantReciprocal>(ctx);
}
