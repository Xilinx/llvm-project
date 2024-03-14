//===- ArithToEmitC.cpp - Arith to EmitC Patterns ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Arith dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
class ArithConstantOpConversionPattern
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp arithConst,
                  arith::ConstantOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        arithConst, arithConst.getType(), adaptor.getValue());
    return success();
  }
};

/// Return an operation that returns true (in i1) when operand is NaN.
emitc::CmpOp isNan(ConversionPatternRewriter &rewriter, Location loc,
                   Value operand) {
  // A value is NaN exactly when it compares unequal to itself.
  return rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::ne, operand, operand);
}

class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    emitc::CmpPredicate predicate;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::UGT:
      // unordered or greater than
      predicate = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::ULT:
      // unordered or less than
      predicate = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::UNO: {
      // unordered, i.e. either operand is nan
      auto lhsIsNan = isNan(rewriter, op.getLoc(), adaptor.getLhs());
      if (adaptor.getLhs() == adaptor.getRhs()) {
        rewriter.replaceOp(op, lhsIsNan);
        return success();
      }
      auto rhsIsNan = isNan(rewriter, op.getLoc(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<arith::OrIOp>(op, op.getType(), lhsIsNan,
                                                rhsIsNan);
      return success();
    }
    default:
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot match predicate ");
    }

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, op.getType(), predicate, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

template <typename ArithOp, typename EmitCOp>
class ArithOpConversion final : public OpConversionPattern<ArithOp> {
public:
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp arithOp, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.template replaceOpWithNewOp<EmitCOp>(arithOp, arithOp.getType(),
                                                  adaptor.getOperands());

    return success();
  }
};

class SelectOpConversion : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp selectOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type dstType = getTypeConverter()->convertType(selectOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(selectOp, "type conversion failed");

    if (!adaptor.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          selectOp,
          "can only be converted if condition is a scalar of type i1");

    rewriter.replaceOpWithNewOp<emitc::ConditionalOp>(selectOp, dstType,
                                                      adaptor.getOperands());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateArithToEmitCPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    ArithConstantOpConversionPattern,
    ArithOpConversion<arith::AddFOp, emitc::AddOp>,
    ArithOpConversion<arith::DivFOp, emitc::DivOp>,
    ArithOpConversion<arith::MulFOp, emitc::MulOp>,
    ArithOpConversion<arith::SubFOp, emitc::SubOp>,
    ArithOpConversion<arith::AddIOp, emitc::AddOp>,
    ArithOpConversion<arith::MulIOp, emitc::MulOp>,
    ArithOpConversion<arith::SubIOp, emitc::SubOp>,
    CmpFOpConversion,
    SelectOpConversion
  >(typeConverter, ctx);
  // clang-format on
}
