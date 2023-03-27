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
#include <functional>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
namespace tosa {

/// Transform a tensor with the given transformation function.
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<llvm::APFloat(const llvm::APFloat &, Type)> &toApply);

/// Function that checks if \p toCheck is a dense TOSA constant float tensor.
LogicalResult notifyIfNotConstantFloatTosaTensor(TypedValue<TensorType> toCheck,
                                                 TosaOp location,
                                                 PatternRewriter &);

/// Function that checks if \p toCheck is a dense TOSA constant tensor.
LogicalResult notifyIfNoTosaDenseConstantTensor(TypedValue<TensorType> toCheck,
                                                TosaOp location,
                                                PatternRewriter &);

/// Function that checks if the type contained in \p toCheck is float.
LogicalResult notifyIfNotFloat(TypedValue<TensorType> toCheck, TosaOp location,
                               PatternRewriter &);

/// Function to compute the reciprocal.
APFloat computeReciprocal(const APFloat &, Type);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_TOSA_FOLD_COMMON_H
