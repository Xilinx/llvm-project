//===- SCFToEmitC.cpp - SCF to EmitC conversion ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.if ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SCFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {

struct SCFToEmitCPass : public impl::SCFToEmitCBase<SCFToEmitCPass> {
  void runOnOperation() override;
};

// Lower scf::for to emitc::for, implementing result values using
// emitc::variable's updated within the loop body.
struct ForLowering : public OpConversionPattern<ForOp> {
  using OpConversionPattern<ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ForOp forOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Create an uninitialized emitc::variable op for each result of the given op.
template <typename T>
static LogicalResult
createVariablesForResults(T op, const TypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter,
                          SmallVector<Value> &resultVariables) {
  if (!op.getNumResults())
    return success();

  Location loc = op->getLoc();
  MLIRContext *context = op.getContext();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpResult result : op.getResults()) {
    Type resultType = typeConverter->convertType(result.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result type conversion failed");
    emitc::OpaqueAttr noInit = emitc::OpaqueAttr::get(context, "");
    emitc::VariableOp var =
        rewriter.create<emitc::VariableOp>(loc, resultType, noInit);
    resultVariables.push_back(var);
  }

  return success();
}

// Create a series of assign ops assigning given values to given variables at
// the current insertion point of given rewriter.
static void assignValues(ValueRange values, ValueRange variables,
                         ConversionPatternRewriter &rewriter, Location loc) {
  for (auto [value, var] : llvm::zip(values, variables))
    rewriter.create<emitc::AssignOp>(loc, var, value);
}

static LogicalResult lowerYield(Operation *op, ValueRange resultVariables,
                                ConversionPatternRewriter &rewriter,
                                scf::YieldOp yield) {
  Location loc = yield.getLoc();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield);

  SmallVector<Value> yieldOperands;
  if (failed(rewriter.getRemappedValues(yield.getOperands(), yieldOperands))) {
    return rewriter.notifyMatchFailure(op, "failed to lower yield operands");
  }

  assignValues(yieldOperands, resultVariables, rewriter, loc);

  rewriter.create<emitc::YieldOp>(loc);
  rewriter.eraseOp(yield);

  return success();
}

LogicalResult
ForLowering::matchAndRewrite(ForOp forOp, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the loop body.
  SmallVector<Value> resultVariables;
  if (failed(createVariablesForResults(forOp, getTypeConverter(), rewriter,
                                       resultVariables)))
    return rewriter.notifyMatchFailure(forOp,
                                       "create variables for results failed");

  assignValues(adaptor.getInitArgs(), resultVariables, rewriter, loc);

  emitc::ForOp loweredFor = rewriter.create<emitc::ForOp>(
      loc, adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep());

  Block *loweredBody = loweredFor.getBody();

  // Erase the auto-generated terminator for the lowered for op.
  rewriter.eraseOp(loweredBody->getTerminator());

  // Convert the original region types into the new types by adding unrealized
  // casts in the beginning of the loop. This performs the conversion in place.
  if (failed(rewriter.convertRegionTypes(&forOp.getRegion(),
                                         *getTypeConverter(), nullptr))) {
    return rewriter.notifyMatchFailure(forOp, "region types conversion failed");
  }

  // Register the replacements for the block arguments and inline the body of
  // the scf.for loop into the body of the emitc::for loop.
  Block *scfBody = &(forOp.getRegion().front());
  SmallVector<Value> replacingValues;
  replacingValues.push_back(loweredFor.getInductionVar());
  replacingValues.append(resultVariables.begin(), resultVariables.end());
  rewriter.mergeBlocks(scfBody, loweredBody, replacingValues);

  auto result = lowerYield(forOp, resultVariables, rewriter,
                           cast<scf::YieldOp>(loweredBody->getTerminator()));

  if (failed(result)) {
    return result;
  }

  rewriter.replaceOp(forOp, resultVariables);
  return success();
}

// Lower scf::if to emitc::if, implementing result values as emitc::variable's
// updated within the then and else regions.
struct IfLowering : public OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

LogicalResult
IfLowering::matchAndRewrite(IfOp ifOp, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const {
  Location loc = ifOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the then & else regions.
  SmallVector<Value> resultVariables;
  if (failed(createVariablesForResults(ifOp, getTypeConverter(), rewriter,
                                       resultVariables)))
    return rewriter.notifyMatchFailure(ifOp,
                                       "create variables for results failed");

  // Utility function to lower the contents of an scf::if region to an emitc::if
  // region. The contents of the scf::if regions is moved into the respective
  // emitc::if regions, but the scf::yield is replaced not only with an
  // emitc::yield, but also with a sequence of emitc::assign ops that set the
  // yielded values into the result variables.
  auto lowerRegion = [&resultVariables, &rewriter,
                      &ifOp](Region &region, Region &loweredRegion) {
    rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
    Operation *terminator = loweredRegion.back().getTerminator();
    auto result = lowerYield(ifOp, resultVariables, rewriter,
                             cast<scf::YieldOp>(terminator));
    if (failed(result)) {
      return result;
    }
    return success();
  };

  Region &thenRegion = adaptor.getThenRegion();
  Region &elseRegion = adaptor.getElseRegion();

  bool hasElseBlock = !elseRegion.empty();

  auto loweredIf =
      rewriter.create<emitc::IfOp>(loc, adaptor.getCondition(), false, false);

  Region &loweredThenRegion = loweredIf.getThenRegion();
  auto result = lowerRegion(thenRegion, loweredThenRegion);
  if (failed(result)) {
    return result;
  }

  if (hasElseBlock) {
    Region &loweredElseRegion = loweredIf.getElseRegion();
    auto result = lowerRegion(elseRegion, loweredElseRegion);
    if (failed(result)) {
      return result;
    }
  }

  rewriter.replaceOp(ifOp, resultVariables);
  return success();
}

void mlir::populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                TypeConverter &typeConverter) {
  patterns.add<ForLowering>(typeConverter, patterns.getContext());
  patterns.add<IfLowering>(typeConverter, patterns.getContext());
}

void SCFToEmitCPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  TypeConverter typeConverter;
  // Fallback converter
  // See note https://mlir.llvm.org/docs/DialectConversion/#type-converter
  // Type converters are called most to least recently inserted
  typeConverter.addConversion([](Type t) { return t; });
  populateEmitCSizeTTypeConversions(typeConverter);
  populateSCFToEmitCConversionPatterns(patterns, typeConverter);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForOp, scf::IfOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
