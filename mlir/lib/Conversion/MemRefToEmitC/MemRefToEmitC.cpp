//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMEMREFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

/// Disallow all memrefs even though we only have conversions
/// for memrefs with static shape right now to have good diagnostics.
bool isLegal(Type t) { return !isa<BaseMemRefType>(t); }

template <typename RangeT>
std::enable_if_t<!std::is_convertible<RangeT, Type>::value &&
                     !std::is_convertible<RangeT, Operation *>::value,
                 bool>
isLegal(RangeT &&range) {
  return llvm::all_of(range, [](Type type) { return isLegal(type); });
}

bool isLegal(Operation *op) {
  return isLegal(op->getOperandTypes()) && isLegal(op->getResultTypes());
}

bool isSignatureLegal(FunctionType ty) {
  return isLegal(llvm::concat<const Type>(ty.getInputs(), ty.getResults()));
}

struct ConvertMemRefToEmitCPass
    : public impl::ConvertMemRefToEmitCBase<ConvertMemRefToEmitCPass> {
  void runOnOperation() override {
    TypeConverter converter;
    // Pass through for all other types.
    converter.addConversion([](Type type) { return type; });

    converter.addConversion([](MemRefType memRefType) -> std::optional<Type> {
      if (memRefType.hasStaticShape()) {
        return emitc::ArrayType::get(memRefType.getShape(),
                                     memRefType.getElementType());
      }
      return {};
    });

    converter.addConversion(
        [&converter](FunctionType ty) -> std::optional<Type> {
          SmallVector<Type> inputs;
          if (failed(converter.convertTypes(ty.getInputs(), inputs)))
            return std::nullopt;

          SmallVector<Type> results;
          if (failed(converter.convertTypes(ty.getResults(), results)))
            return std::nullopt;

          return FunctionType::get(ty.getContext(), inputs, results);
        });

    RewritePatternSet patterns(&getContext());
    populateMemRefToEmitCConversionPatterns(patterns, converter);

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp op) { return isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalDialect<func::FuncDialect>(
        [](Operation *op) { return isLegal(op); });
    target.addIllegalDialect<
    memref::MemRefDialect
    >(
      
    );

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void mlir::populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &converter) {

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
}

std::unique_ptr<OperationPass<>> mlir::createConvertMemRefToEmitCPass() {
  return std::make_unique<ConvertMemRefToEmitCPass>();
}
