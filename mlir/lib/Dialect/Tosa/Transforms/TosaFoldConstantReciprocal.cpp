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
#include "mlir/Dialect/Tosa/Transforms/TosaFoldCommon.h"
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

  LogicalResult matchAndRewrite(ReciprocalOp recip,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = recip.getInput1();

    // Check that we can apply folding
    auto preCondCheck =
        notifyIfNotConstantFloatTosaTensor(inputTensor, recip, rewriter);
    if (failed(preCondCheck)) {
      return preCondCheck;
    }

    // Extract the tensor values
    DenseElementsAttr inputValues;
    matchPattern(inputTensor, m_Constant(&inputValues));

    // Our transformation replaces the input tensor with the transformed tensor.
    // If the input has several users we need to keep the input. This can
    // result in a significantly increased memory usage, such that we currently
    // refrain from applying the transformation in that case.
    // Allow this only for splat values, because the amount of data is small.
    if (!inputTensor.hasOneUse() && !isa<SplatElementsAttr>(inputValues)) {
      return rewriter.notifyMatchFailure(
          recip, "Currently, reciprocals will only be folded if the input "
                 "tensor has a single user");
    }

    // Create a new tensor with the updated values
    auto newTensor = applyElementWise(inputValues, &computeReciprocal);

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
