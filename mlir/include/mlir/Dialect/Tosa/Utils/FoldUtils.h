//===- FoldUtils.h - Helper Functions for Folds -----------------*- C++ -*-===//
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
#ifndef MLIR_DIALECT_TOSA_UTILS_FOLD_UTILS_H
#define MLIR_DIALECT_TOSA_UTILS_FOLD_UTILS_H

#include <functional>
#include <mlir/IR/BuiltinAttributes.h>

namespace mlir {
namespace tosa {

/// Rounding mode to be used on floating point operations that require rounding.
static constexpr llvm::RoundingMode tosaRoundingMode =
    llvm::APFloat::rmNearestTiesToEven;

/// Apply the given transformation \p toApply to every element of the tensor to
/// be transformed \p toTransform.
///
/// Elements of \p toTransform are extracted as \p SrcValueType.
///
/// \returns A tensor with the same size as \p toTransform, containing
/// \p TargetValueType values of type \p TargetType.
template <class SrcValType, class TargetValType, class TargetType>
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<TargetValType(const SrcValType &, TargetType)> &toApply,
    TargetType targetType);

} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_UTILS_FOLD_UTILS_H
