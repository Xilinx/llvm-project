//===- FoldUtils.cpp ------------------------------------------------------===//
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

#include "mlir/Dialect/Tosa/Utils/FoldUtils.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

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

  // Make sure that the output tensor has the expected output type
  auto inShape = toTransform.getType();
  auto outTy = inShape.cloneWith({}, targetType);

  return DenseElementsAttr::get(outTy, transformedValues);
}

template DenseElementsAttr
mlir::tosa::applyElementWise<APFloat, APFloat, FloatType>(
    const DenseElementsAttr &toTransform,
    const std::function<APFloat(const APFloat &, FloatType)> &toApply,
    FloatType targetType);
