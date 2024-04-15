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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
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

/// Return an operation that returns true (in i1) when operand is not NaN.
emitc::CmpOp isNotNan(ConversionPatternRewriter &rewriter, Location loc,
                      Value operand) {
  // A value is not NaN exactly when it compares equal to itself.
  return rewriter.create<emitc::CmpOp>(
      loc, rewriter.getI1Type(), emitc::CmpPredicate::eq, operand, operand);
}

/// Return an op that return true (in i1) if the operands \p first and \p second
/// are unordered (i.e., at least one of them is NaN).
emitc::LogicalOrOp createCheckIsUnordered(ConversionPatternRewriter &rewriter,
                                          Location loc, Value first,
                                          Value second) {
  auto firstIsNaN = isNan(rewriter, loc, first);
  auto secondIsNaN = isNan(rewriter, loc, second);
  return rewriter.create<emitc::LogicalOrOp>(loc, rewriter.getI1Type(),
                                             firstIsNaN, secondIsNaN);
}

/// Return an op that return true (in i1) if the operands \p first and \p second
/// are both ordered (i.e., none one of them is NaN).
Value createCheckIsOrdered(ConversionPatternRewriter &rewriter, Location loc,
                           Value first, Value second) {
  auto firstIsNaN = isNotNan(rewriter, loc, first);
  auto secondIsNaN = isNotNan(rewriter, loc, second);
  return rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                              firstIsNaN, secondIsNaN);
}

class CmpFOpConversion : public OpConversionPattern<arith::CmpFOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (!isa<FloatType>(adaptor.getRhs().getType())) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cmpf currently only supported on "
                                         "floats, not tensors/vectors thereof");
    }

    bool unordered = false;
    emitc::CmpPredicate predicate;
    switch (op.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse: {
      auto constant = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(),
          rewriter.getBoolAttr(/*value=*/false));
      rewriter.replaceOp(op, constant);
      return success();
    }
    case arith::CmpFPredicate::OEQ:
      unordered = false;
      predicate = emitc::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::OGT:
      // ordered and greater than
      unordered = false;
      predicate = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::OGE:
      unordered = false;
      predicate = emitc::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::OLT:
      unordered = false;
      predicate = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::ONE:
      unordered = false;
      predicate = emitc::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::ORD: {
      // ordered, i.e. none of the operands is NaN
      auto cmp = createCheckIsOrdered(rewriter, op.getLoc(), adaptor.getLhs(),
                                      adaptor.getRhs());
      rewriter.replaceOp(op, cmp);
      return success();
    }
    case arith::CmpFPredicate::UEQ:
      // unordered or equal
      unordered = true;
      predicate = emitc::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::UGT:
      // unordered or greater than
      unordered = true;
      predicate = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::UGE:
      // unordered or greater equal
      unordered = true;
      predicate = emitc::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::ULT:
      // unordered or less than
      unordered = true;
      predicate = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::ULE:
      // unordered or less than
      unordered = true;
      predicate = emitc::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::UNE:
      unordered = true;
      predicate = emitc::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::UNO: {
      // unordered, i.e. either operand is nan
      auto cmp = createCheckIsUnordered(rewriter, op.getLoc(), adaptor.getLhs(),
                                        adaptor.getRhs());
      rewriter.replaceOp(op, cmp);
      return success();
    }
    case arith::CmpFPredicate::AlwaysTrue: {
      auto constant = rewriter.create<emitc::ConstantOp>(
          op.getLoc(), rewriter.getI1Type(),
          rewriter.getBoolAttr(/*value=*/true));
      rewriter.replaceOp(op, constant);
      return success();
    }
    default:
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot match predicate ");
    }

    // Compare the values naively
    auto cmpResult =
        rewriter.create<emitc::CmpOp>(op.getLoc(), op.getType(), predicate,
                                      adaptor.getLhs(), adaptor.getRhs());

    // Adjust the results for unordered/ordered semantics
    if (unordered) {
      auto isUnordered = createCheckIsUnordered(
          rewriter, op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
      rewriter.replaceOpWithNewOp<emitc::LogicalOrOp>(
          op, op.getType(), isUnordered.getResult(), cmpResult);
      return success();
    }

    auto isOrdered = createCheckIsOrdered(rewriter, op.getLoc(),
                                          adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOpWithNewOp<emitc::LogicalAndOp>(op, op.getType(),
                                                     isOrdered, cmpResult);
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

// Floating-point to integer conversions.
template <typename CastOp>
class FtoICastOpConversion : public OpConversionPattern<CastOp> {
private:
  bool floatToIntTruncates;

public:
  FtoICastOpConversion(const TypeConverter &typeConverter, MLIRContext *context,
                       bool optionFloatToIntTruncates)
      : OpConversionPattern<CastOp>(typeConverter, context),
        floatToIntTruncates(optionFloatToIntTruncates) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type operandType = adaptor.getIn().getType();
    if (!emitc::isSupportedFloatType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    if (!floatToIntTruncates)
      return rewriter.notifyMatchFailure(
          castOp, "conversion currently requires EmitC casts to use truncation "
                  "as rounding mode");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    if (!emitc::isSupportedIntegerType(dstType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    rewriter.replaceOpWithNewOp<emitc::CastOp>(castOp, dstType,
                                               adaptor.getOperands());

    return success();
  }
};

// Integer to floating-point conversions.
template <typename CastOp>
class ItoFCastOpConversion : public OpConversionPattern<CastOp> {
public:
  ItoFCastOpConversion(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<CastOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(CastOp castOp, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type operandType = adaptor.getIn().getType();
    if (!emitc::isSupportedIntegerType(operandType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast source type");

    Type dstType = this->getTypeConverter()->convertType(castOp.getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(castOp, "type conversion failed");

    if (!emitc::isSupportedFloatType(dstType))
      return rewriter.notifyMatchFailure(castOp,
                                         "unsupported cast destination type");

    rewriter.replaceOpWithNewOp<emitc::CastOp>(castOp, dstType,
                                               adaptor.getOperands());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateArithToEmitCPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        bool optionFloatToIntTruncates) {
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
    SelectOpConversion,
    ItoFCastOpConversion<arith::SIToFPOp>,
    ItoFCastOpConversion<arith::UIToFPOp>
  >(typeConverter, ctx)
  .add<
    FtoICastOpConversion<arith::FPToSIOp>,
    FtoICastOpConversion<arith::FPToUIOp>
  >(typeConverter, ctx, optionFloatToIntTruncates);
  // clang-format on
}
