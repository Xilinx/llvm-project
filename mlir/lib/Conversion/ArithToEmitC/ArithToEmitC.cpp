//===- ArithToEmitC.cpp - Arith to EmitC conversion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert arith ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

static bool isConvertibleToEmitC(Type type) {
  Type baseType = type;
  if (auto tensorType = dyn_cast<TensorType>(type)) {
    if (!tensorType.hasRank() || !tensorType.hasStaticShape()) {
      return false;
    }
    baseType = tensorType.getElementType();
  }

  if (isa<IndexType>(baseType)) {
    return true;
  }

  if (auto intType = dyn_cast<IntegerType>(baseType)) {
    switch (intType.getWidth()) {
    case 1:
    case 8:
    case 16:
    case 32:
    case 64:
      return true;
    }
    return false;
  }

  if (auto floatType = dyn_cast<FloatType>(baseType)) {
    return floatType.isF32() || floatType.isF64();
  }

  return false;
}

class ArithConstantOpConversionPattern
    : public OpRewritePattern<arith::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp arithConst,
                                PatternRewriter &rewriter) const override {

    auto constantType = arithConst.getType();
    if (!isConvertibleToEmitC(constantType)) {
      return rewriter.notifyMatchFailure(arithConst.getLoc(),
                                         "Type cannot be converted to emitc");
    }

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(arithConst, constantType,
                                                   arithConst.getValue());
    return success();
  }
};

struct ConvertArithToEmitCPass
    : public impl::ArithToEmitCConversionPassBase<ConvertArithToEmitCPass> {
public:
  void runOnOperation() override {

    ConversionTarget target(getContext());
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    RewritePatternSet patterns(&getContext());
    populateArithToEmitCConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateArithToEmitCConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ArithConstantOpConversionPattern>(patterns.getContext());
}
