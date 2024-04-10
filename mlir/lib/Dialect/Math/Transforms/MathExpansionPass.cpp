//===- MathExpansionPass.cpp - Pass to expand math ops --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass to call the expansion patterns defined in
// ExpandPatterns.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::math {
#define GEN_PASS_DEF_MATHEXPANSION
#include "mlir/Dialect/Math/Transforms/Passes.h.inc"
} // namespace mlir::math

using namespace mlir;

namespace {

struct MathExpansion final
    : public math::impl::MathExpansionBase<MathExpansion> {
  using MathExpansionBase::MathExpansionBase;

  void runOnOperation() override {
    auto *op = getOperation();

    RewritePatternSet patterns(op->getContext());
    if (expandCtlz)
      populateExpandCtlzPattern(patterns);
    if (expandTan)
      populateExpandTanPattern(patterns);
    if (expandTanh)
      populateExpandTanhPattern(patterns);
    if (expandFmaF)
      populateExpandFmaFPattern(patterns);
    if (expandFloorF)
      populateExpandFloorFPattern(patterns);
    if (expandCeilF)
      populateExpandCeilFPattern(patterns);
    if (expandExp2F)
      populateExpandExp2FPattern(patterns);
    if (expandPowF)
      populateExpandPowFPattern(patterns);
    if (expandRoundF)
      populateExpandRoundFPattern(patterns);
    if (expandRoundEven)
      populateExpandRoundEvenPattern(patterns);
    if (expandRsqrt)
      populateExpandRsqrtPattern(patterns);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
