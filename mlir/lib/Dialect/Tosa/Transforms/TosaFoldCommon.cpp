//===- TosaFoldCommon.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions useful for various different TOSA constant folds.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include <llvm/ADT/APFloat.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

namespace {
static constexpr llvm::RoundingMode reciprocalRoundingMode =
    APFloat::rmNearestTiesToEven;
} // namespace

DenseElementsAttr mlir::tosa::applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<llvm::APFloat(const llvm::APFloat &, Type)> &toApply) {
  llvm::SmallVector<llvm::APFloat, 1> transformedValues;
  // We already know the amount of values we will insert, reserve space for
  // all of them to avoid dynamic resizing
  transformedValues.reserve(toTransform.getNumElements());
  for (auto val : toTransform.getValues<llvm::APFloat>()) {
    auto transformedVal = toApply(val, toTransform.getElementType());
    transformedValues.push_back(transformedVal);
  }

  // Replace the current tensor with one containing the computed values
  auto newTensor =
      DenseElementsAttr::get(toTransform.getType(), transformedValues);
  return newTensor;
}

LogicalResult
mlir::tosa::notifyIfNotConstantFloatTosaTensor(TypedValue<TensorType> toCheck,
                                               TosaOp location,
                                               PatternRewriter &rewriter) {
  auto floatCheck = notifyIfNotFloat(toCheck, location, rewriter);
  if (failed(floatCheck)) {
    return floatCheck;
  }
  return notifyIfNoTosaDenseConstantTensor(toCheck, location, rewriter);
}

LogicalResult
mlir::tosa::notifyIfNoTosaDenseConstantTensor(TypedValue<TensorType> toCheck,
                                              TosaOp location,
                                              PatternRewriter &rewriter) {
  // Check whether the tensor is constant and dense
  // TODO We currently ensure the tensor is dense by using the correct type for
  // the bind_value, however we do not actually need this value. It would be
  // nicer to only have a check here.
  DenseElementsAttr tmp;
  if (!matchPattern(toCheck, m_Constant(&tmp))) {
    return rewriter.notifyMatchFailure(location,
                                       "Non-const or non-dense input tensor");
  }

  // Make sure it actually is a TOSA constant (the match allows for other
  // constants as well)
  if (isa<ConstOp>(toCheck.getDefiningOp())) {
    return success();
  }

  return rewriter.notifyMatchFailure(location,
                                     "The reciprocal can only be folded if "
                                     "it operates on a TOSA constant");
}

LogicalResult mlir::tosa::notifyIfNotFloat(TypedValue<TensorType> toCheck,
                                           TosaOp location,
                                           PatternRewriter &rewriter) {
  if (isa<FloatType>(toCheck.getType().getElementType())) {
    return success();
  }
  return rewriter.notifyMatchFailure(location,
                                     "Unexpected input tensor type: the "
                                     "TOSA spec only allows floats");
}

APFloat mlir::tosa::computeReciprocal(const APFloat &floatVal, Type floatTy) {
  auto recipAttr = FloatAttr::get(floatTy, 1.0);
  APFloat recip = recipAttr.getValue();
  recip.divide(floatVal, reciprocalRoundingMode);

  return recip;
}
