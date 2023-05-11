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
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
namespace tosa {

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

/// Heuristic to decide when to replace a unary operation on a constant with the
/// folded value.
/// Folding operations on constants can lead to an increased memory usage
/// whenever the input cannot be replaced but a new constant is inserted. Hence,
/// this will currently only suggest folding when the memory impact is
/// negligible.
/// Takes the \p unaryOp and the constant input \p values.
/// \returns Whether folding should be applied.
bool constantUnaryOpShouldBeFolded(TosaOp unaryOp, DenseElementsAttr values);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_TOSA_FOLD_COMMON_H
