//===- FuncToEmitCPass.cpp - Func to EmitC Pass -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Func dialect to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToEmitC/FuncToEmitCPass.h"

#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_CONVERTFUNCTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertFuncToEmitC
    : public impl::ConvertFuncToEmitCBase<ConvertFuncToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertFuncToEmitC::runOnOperation() {
  // Convert function interface types within the func dialect first to supported
  // EmitC types
  ConversionTarget interfaceConversionTarget(getContext());
  interfaceConversionTarget.addDynamicallyLegalOp<func::CallOp>(
      [](func::CallOp op) {
        auto operandTypes = op->getOperandTypes();
        if (std::any_of(operandTypes.begin(), operandTypes.end(),
                        [](Type t) { return isa<IndexType>(t); }))
          return false;
        auto resultTypes = op.getResultTypes();
        return !(std::any_of(resultTypes.begin(), resultTypes.end(),
                             [](Type t) { return isa<IndexType>(t); }));
      });
  interfaceConversionTarget.addDynamicallyLegalOp<func::FuncOp>(
      [](func::FuncOp op) {
        auto operandTypes = op->getOperandTypes();
        if (std::any_of(operandTypes.begin(), operandTypes.end(),
                        [](Type t) { return isa<IndexType>(t); }))
          return false;
        auto argumentTypes = op.getArgumentTypes();
        if (std::any_of(argumentTypes.begin(), argumentTypes.end(),
                        [](Type t) { return isa<IndexType>(t); }))
          return false;
        auto resultTypes = op.getResultTypes();
        return !(std::any_of(resultTypes.begin(), resultTypes.end(),
                             [](Type t) { return isa<IndexType>(t); }));
      });
  interfaceConversionTarget.addDynamicallyLegalOp<func::ReturnOp>(
      [](func::ReturnOp op) {
        auto operandTypes = op->getOperandTypes();
        return !(std::any_of(operandTypes.begin(), operandTypes.end(),
                             [](Type t) { return isa<IndexType>(t); }));
      });

  RewritePatternSet interfaceRewritePatterns(&getContext());
  TypeConverter typeConverter;
  populateEmitCSizeTypeConversions(typeConverter);
  populateEmitCDefaultTypeConversions(typeConverter);
  populateReturnOpTypeConversionPattern(interfaceRewritePatterns,
                                        typeConverter);
  populateCallOpTypeConversionPattern(interfaceRewritePatterns, typeConverter);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
      interfaceRewritePatterns, typeConverter);

  if (failed(applyPartialConversion(getOperation(), interfaceConversionTarget,
                                    std::move(interfaceRewritePatterns))))
    signalPassFailure();

  // Then convert the func ops themselves to EmitC
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalOp<func::CallOp, func::FuncOp, func::ReturnOp>();

  RewritePatternSet patterns(&getContext());
  populateFuncToEmitCPatterns(patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
