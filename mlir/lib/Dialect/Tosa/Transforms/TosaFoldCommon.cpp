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

#include "TosaFoldCommon.h"
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::tosa;

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

bool mlir::tosa::constantUnaryOpShouldBeFolded(TosaOp unaryOp,
                                               DenseElementsAttr values) {
  assert(unaryOp->getNumOperands() == 1);
  auto inputOp = unaryOp->getOperand(0);

  // If the input is a splat, we don't care for the number of users
  if (isa<SplatElementsAttr>(values)) {
    return true;
  }

  // If this is the only use of the tensor it should be replaced as no
  // additional memory is required
  return inputOp.hasOneUse();
}
