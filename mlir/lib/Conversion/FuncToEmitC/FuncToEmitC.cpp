//===- FuncToEmitC.cpp - Func to EmitC Patterns -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Func dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
class CallOpConversion final : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Multiple results func cannot be converted to `emitc.func`.
    if (callOp.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          callOp, "only functions with zero or one result can be converted");

    // Convert the original function results.
    SmallVector<Type> types;
    if (failed(typeConverter->convertTypes(callOp.getResultTypes(), types))) {
      return rewriter.notifyMatchFailure(
          callOp, "function return type conversion failed");
    }
    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        callOp, types, adaptor.getOperands(), callOp->getAttrs());

    return success();
  }
};

class FuncOpConversion final : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    FunctionType type = funcOp.getFunctionType();
    if (!type)
      return failure();

    if (type.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          funcOp, "only functions with zero or one result can be converted");

    const TypeConverter *converter = getTypeConverter();

    // Convert function signature
    TypeConverter::SignatureConversion signatureConversion(type.getNumInputs());
    SmallVector<Type, 1> convertedResults;
    if (failed(converter->convertSignatureArgs(type.getInputs(),
                                               signatureConversion)) ||
        failed(converter->convertTypes(type.getResults(), convertedResults)) ||
        failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                           *converter, &signatureConversion)))
      return rewriter.notifyMatchFailure(funcOp, "signature conversion failed");

    // Convert the function type
    auto convertedFunctionType = FunctionType::get(
        rewriter.getContext(), signatureConversion.getConvertedTypes(),
        convertedResults);

    // Create the converted `emitc.func` op.
    emitc::FuncOp newFuncOp = rewriter.create<emitc::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), convertedFunctionType);

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Add `extern` to specifiers if `func.func` is declaration only.
    if (funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"extern"});
      newFuncOp.setSpecifiersAttr(specifiers);
    }

    // Add `static` to specifiers if `func.func` is private but not a
    // declaration.
    if (funcOp.isPrivate() && !funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"static"});
      newFuncOp.setSpecifiersAttr(specifiers);
    }

    if (!funcOp.isDeclaration())
      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());
    rewriter.eraseOp(funcOp);

    return success();
  }
};

class ReturnOpConversion final : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (returnOp.getNumOperands() > 1)
      return rewriter.notifyMatchFailure(
          returnOp, "only zero or one operand is supported");

    rewriter.replaceOpWithNewOp<emitc::ReturnOp>(
        returnOp,
        returnOp.getNumOperands() ? adaptor.getOperands()[0] : nullptr);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateFuncToEmitCPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<CallOpConversion, FuncOpConversion, ReturnOpConversion>(
      typeConverter, ctx);
}
