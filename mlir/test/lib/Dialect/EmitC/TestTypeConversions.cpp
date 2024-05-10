//===- TestTypeConversions.cpp - Test EmitC type conversions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a test pass to check EmitC type conversions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct TestEmitCTypeConversions
    : public PassWrapper<TestEmitCTypeConversions, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestEmitCTypeConversions)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-emitc-type-conversions"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect>();
  }
  StringRef getDescription() const final {
    return "Test EmitC type conversions";
  }
};

struct ConstantConverter : public OpConversionPattern<emitc::ConstantOp> {
public:
  using OpConversionPattern<emitc::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(emitc::ConstantOp constantOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get value attribute
    auto valueAttr = ::cast<TypedAttr>(constantOp.getValue());
    valueAttr.dump();

    Type dstType = getTypeConverter()->convertType(valueAttr.getType());

    if (!dstType)
      return rewriter.notifyMatchFailure(constantOp, "type conversion failed");

    // Invalid op since value should be an attribute but we'll get here at least
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(constantOp, dstType,
                                                   adaptor.getOperands());

    return success();
  }
};

} // namespace

void TestEmitCTypeConversions::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  TypeConverter typeConverter;
  populateEmitCSizeTypeConversionPatterns(typeConverter);
  // Add a default converter
  // typeConverter.addConversion([](Type t) { t.dump(); return t; });
  ConversionTarget target(getContext());
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addDynamicallyLegalOp<emitc::ConstantOp>([](emitc::ConstantOp op) {
    return isa<emitc::SignedSizeType, emitc::UnsignedSizeType>(op.getType());
  });
  patterns.insert<ConstantConverter>(typeConverter, context);
  (void)applyPartialConversion(getOperation(), target, std::move(patterns));
}

namespace mlir {
namespace test {
void registerTestEmitCTypeConversions() {
  PassRegistration<TestEmitCTypeConversions>();
}
} // namespace test
} // namespace mlir
