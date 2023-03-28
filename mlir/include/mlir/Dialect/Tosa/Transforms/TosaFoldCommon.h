//===- TosaFoldCommon.h - Helper Functions for Folds ------------*- C++ -*-===//
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
#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_TOSA_FOLD_COMMON_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_TOSA_FOLD_COMMON_H

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <functional>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
namespace tosa {

/// Type that represents tensor dimensions.
using DimensionType = ArrayRef<int64_t>;

/// Type for tensor offsets.
using OffsetType = size_t;

/// Transform a tensor with the given transformation function.
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<llvm::APFloat(const llvm::APFloat &, Type)> &toApply);

/// Apply the given transformation function on the elements of the given
/// tensors. If the input tensors do not match \p targetType, broadcasting is
/// applied.
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &first, const DenseElementsAttr &second,
    TensorType targetType,
    const std::function<APFloat(const APFloat &, const APFloat &)> &toApply);

/// Function that checks if \p toCheck is a dense TOSA constant float tensor.
LogicalResult notifyIfNotConstantFloatTosaTensor(TypedValue<TensorType> toCheck,
                                                 TosaOp location,
                                                 PatternRewriter &rewriter);

/// Function that checks if \p toCheck is a dense TOSA constant tensor.
LogicalResult notifyIfNoTosaDenseConstantTensor(TypedValue<TensorType> toCheck,
                                                TosaOp location,
                                                PatternRewriter &rewriter);

/// Function that checks if the type contained in \p toCheck is float.
LogicalResult notifyIfNotFloat(TypedValue<TensorType> toCheck, TosaOp location,
                               PatternRewriter &rewriter);

/// Compute the offset in \p shape which corresponds to the given \p index.
OffsetType indexToOffset(DimensionType shape, DimensionType index);

/// Compute the index into \p shape which corresponds to the given \p offset.
SmallVector<int64_t> offsetToIndex(DimensionType shape, OffsetType offset);

/// Given an \p index into \p desiredShape, compute the corresponding index into
/// \p toBeBroadcastedShape.
/// \returns broadcasted index into \p toBeBroadcastedShape.
SmallVector<int64_t> getBroadcastedIndex(DimensionType desiredShape,
                                         DimensionType toBeBroadcastedShape,
                                         DimensionType index);
/// Given an \p offset into \p desiredShape, compute the corresponding offset
/// into \p toBeBroadcastedShape.
/// \returns broadcasted offset into \p toBeBroadcastedShape.
OffsetType getBroadcastedOffset(DimensionType desiredShape,
                                DimensionType toBeBroadcastedShape,
                                OffsetType offset);

/// Function to compute the reciprocal.
APFloat computeReciprocal(const APFloat &floatVal, Type floatTy);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_TOSA_FOLD_COMMON_H
