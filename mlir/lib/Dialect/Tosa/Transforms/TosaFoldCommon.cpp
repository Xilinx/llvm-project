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
#include <llvm/ADT/SmallVector.h>
#include <algorithm>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

template <class SrcValType, class TargetValType, class TargetType>
DenseElementsAttr mlir::tosa::applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<TargetValType(const SrcValType &, TargetType)> &toApply,
    TargetType targetType) {
  SmallVector<TargetValType> transformedValues;
  // We already know the amount of values we will insert, reserve space for
  // all of them to avoid dynamic resizing
  transformedValues.reserve(toTransform.getNumElements());
  for (auto val : toTransform.getValues<SrcValType>()) {
    auto transformedVal = toApply(val, targetType);
    transformedValues.push_back(transformedVal);
  }

  auto inShape = toTransform.getType();
  auto outTy = inShape.cloneWith({}, targetType);

  // Replace the current tensor with one containing the computed values
  auto newTensor = DenseElementsAttr::get(outTy, transformedValues);
  return newTensor;
}

template DenseElementsAttr
mlir::tosa::applyElementWise<APFloat, APFloat, FloatType>(
    const DenseElementsAttr &toTransform,
    const std::function<APFloat(const APFloat &, FloatType)> &toApply,
    FloatType targetType);

template DenseElementsAttr
mlir::tosa::applyElementWise<APInt, APFloat, FloatType>(
    const DenseElementsAttr &toTransform,
    const std::function<APFloat(const APInt &, FloatType)> &toApply,
    FloatType targetType);

template DenseElementsAttr
mlir::tosa::applyElementWise<APFloat, APInt, IntegerType>(
    const DenseElementsAttr &toTransform,
    const std::function<APInt(const APFloat &, IntegerType)> &toApply,
    IntegerType targetType);

template DenseElementsAttr
mlir::tosa::applyElementWise<APInt, APInt, IntegerType>(
    const DenseElementsAttr &toTransform,
    const std::function<APInt(const APInt &, IntegerType)> &toApply,
    IntegerType targetType);

template <class ElementType, class ResultType>
DenseElementsAttr mlir::tosa::applyElementWise(
    const DenseElementsAttr &first, const DenseElementsAttr &second,
    TensorType targetType,
    const std::function<ResultType(const ElementType &, const ElementType &)>
        &toApply) {
  // Make sure to use the correct values in case broadcasting is required
  SmallVector<ResultType> transformedValues;
  // We already know the amount of values we will insert, reserve space for
  // all of them to avoid dynamic resizing
  auto targetSize = 1;
  auto targetShape = targetType.getShape();
  for (const auto &dimSize : targetShape) {
    targetSize *= dimSize;
  }
  transformedValues.reserve(targetSize);

  // Apply the given function to each pair of values from the input tensors.
  // Make sure to broadcast the offsets properly.
  auto firstIt = first.getValues<ElementType>();
  auto firstShape = first.getType().getShape();
  auto secondIt = second.getValues<ElementType>();
  auto secondShape = second.getType().getShape();
  for (auto offset = 0; offset < targetSize; offset++) {
    OffsetType offsetInTargetFirst =
        getBroadcastedOffset(targetShape, firstShape, offset);
    OffsetType offsetInTargetSecond =
        getBroadcastedOffset(targetShape, secondShape, offset);
    auto res =
        toApply(firstIt[offsetInTargetFirst], secondIt[offsetInTargetSecond]);
    transformedValues.push_back(res);
  }

  // Generate a tensor with the computed values.
  auto newTensor = DenseElementsAttr::get(targetType, transformedValues);
  return newTensor;
}

template DenseElementsAttr mlir::tosa::applyElementWise<APFloat, APFloat>(
    const DenseElementsAttr &first, const DenseElementsAttr &second,
    TensorType targetType,
    const std::function<APFloat(const APFloat &, const APFloat &)> &toApply);

template DenseElementsAttr mlir::tosa::applyElementWise<APInt, APInt>(
    const DenseElementsAttr &first, const DenseElementsAttr &second,
    TensorType targetType,
    const std::function<APInt(const APInt &, const APInt &)> &toApply);

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

OffsetType mlir::tosa::indexToOffset(DimensionType shape, DimensionType index) {
  OffsetType offset = 0;
  for (size_t i = 0; i < shape.size(); i++) {
    offset = offset * shape[i] + index[i];
  }
  return offset;
}

SmallVector<int64_t> mlir::tosa::offsetToIndex(DimensionType shape,
                                               OffsetType offset) {
  auto rank = shape.size();
  // The rank of the index will be equal to the rank of the shape
  SmallVector<int64_t> resultIndex;
  resultIndex.reserve(rank);
  // Compute all the index values from the last to the first one, reverse the
  // vector afterwards as there is no convenient push_front.
  for (int32_t i = rank - 1; i >= 0; i--) {
    resultIndex.push_back(offset % shape[i]);
    offset /= shape[i];
  }
  std::reverse(resultIndex.begin(), resultIndex.end());
  return resultIndex;
}

SmallVector<int64_t>
mlir::tosa::getBroadcastedIndex(DimensionType desiredShape,
                                DimensionType toBeBroadcastedShape,
                                DimensionType index) {
  SmallVector<int64_t> broadCasted;
  broadCasted.reserve(desiredShape.size());
  for (size_t i = 0; i < desiredShape.size(); i++) {
    auto toInsert = 0;
    if (toBeBroadcastedShape[i] == desiredShape[i]) {
      toInsert = index[i];
    }
    broadCasted.push_back(toInsert);
  }
  return broadCasted;
}

OffsetType mlir::tosa::getBroadcastedOffset(DimensionType desiredShape,
                                            DimensionType toBeBroadcastedShape,
                                            OffsetType offset) {
  // Simply return the offset if the shapes are equal.
  if (desiredShape.equals(toBeBroadcastedShape)) {
    return offset;
  }
  auto indexInTarget = offsetToIndex(desiredShape, offset);
  auto indexBroadcasted =
      getBroadcastedIndex(desiredShape, toBeBroadcastedShape, indexInTarget);
  return indexToOffset(toBeBroadcastedShape, indexBroadcasted);
}

bool mlir::tosa::constantBinaryOpShouldBeFolded(
    TosaOp binaryOp, DenseElementsAttr valuesFirst,
    DenseElementsAttr valuesSecond) {
  assert(binaryOp->getNumOperands() == 2);
  auto firstOp = binaryOp->getOperand(0);
  auto secondOp = binaryOp->getOperand(1);

  // If both tensors are splat, we don't care for the number of users
  if (isa<SplatElementsAttr>(valuesFirst) &&
      isa<SplatElementsAttr>(valuesSecond)) {
    return true;
  }

  // If this is the only use of one of the tensors, it will be replaced an no
  // additional memory is required.
  if (firstOp.hasOneUse() || secondOp.hasOneUse()) {
    return true;
  }

  // Fold it both inputs are equal and those are the only uses. Don't fold
  // otherwise.
  auto numUsers =
      std::distance(firstOp.getUses().begin(), firstOp.getUses().end());
  return firstOp == secondOp && numUsers == 2;
}

bool mlir::tosa::constantUnaryOpShouldBeFolded(TosaOp unaryOp,
                                               DenseElementsAttr values) {
  assert(unaryOp->getNumOperands() == 1);
  auto inputOp = unaryOp->getOperand(0);

  // If the input is a splat, we don't care for the number of users
  if (isa<SplatElementsAttr>(values)) {
    return true;
  }

  // If this is the only use of the tensors it will be replaced an no
  // additional memory is required.
  return inputOp.hasOneUse();
}

APFloat mlir::tosa::computeReciprocal(const APFloat &floatVal,
                                      FloatType floatTy) {
  auto recipAttr = FloatAttr::get(floatTy, 1.0);
  APFloat recip = recipAttr.getValue();
  recip.divide(floatVal, tosaRoundingMode);

  return recip;
}
