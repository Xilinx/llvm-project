//===- MemRefToEmitC.cpp - MemRef to EmitC conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert memref ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct ConvertAlloca final : public OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Allow alignment if it is not more than the natural alignment
      // of the C array.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform alloca with alignment requirement");
    }

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }
    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, resultTy, noInit);
    return success();
  }
};

struct ConvertGlobal final : public OpConversionPattern<memref::GlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    if (!op.getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op.getLoc(), "cannot transform global with dynamic shape");
    }

    if (op.getAlignment().value_or(1) > 1) {
      // TODO: Extend GlobalOp to specify alignment via the `alignas` specifier.
      return rewriter.notifyMatchFailure(
          op.getLoc(), "global variable with alignment requirement is "
                       "currently not supported");
    }
    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }

    SymbolTable::Visibility visibility = SymbolTable::getSymbolVisibility(op);
    if (visibility != SymbolTable::Visibility::Public &&
        visibility != SymbolTable::Visibility::Private) {
      return rewriter.notifyMatchFailure(
          op.getLoc(),
          "only public and private visibility is currently supported");
    }
    // We are explicit in specifing the linkage because the default linkage
    // for constants is different in C and C++.
    bool staticSpecifier = visibility == SymbolTable::Visibility::Private;
    bool externSpecifier = !staticSpecifier;

    Attribute initialValue = operands.getInitialValueAttr();
    if (isa_and_present<UnitAttr>(initialValue))
      initialValue = {};

    rewriter.replaceOpWithNewOp<emitc::GlobalOp>(
        op, operands.getSymName(), resultTy, initialValue, externSpecifier,
        staticSpecifier, operands.getConstant());
    return success();
  }
};

struct ConvertGetGlobal final
    : public OpConversionPattern<memref::GetGlobalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
    rewriter.replaceOpWithNewOp<emitc::GetGlobalOp>(op, resultTy,
                                                    operands.getNameAttr());
    return success();
  }
};

struct ConvertLoad final : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(), "cannot convert type");
    }

    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    auto pointerValue =
        dyn_cast<TypedValue<emitc::PointerType>>(operands.getMemref());
    if (!arrayValue && !pointerValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    Value subscript;
    if (arrayValue) {
      subscript = rewriter.create<emitc::SubscriptOp>(op.getLoc(), arrayValue,
                                                      operands.getIndices());
    } else {
      // !! This is completely broken !! Just to see if it generates.
      // The indices need to be properly calculated
      subscript = rewriter.create<emitc::SubscriptOp>(op.getLoc(), pointerValue,
                                                      operands.getIndices()[0]);
    }

    auto noInit = emitc::OpaqueAttr::get(getContext(), "");
    auto var =
        rewriter.create<emitc::VariableOp>(op.getLoc(), resultTy, noInit);

    auto assign = rewriter.create<emitc::AssignOp>(op.getLoc(), var, subscript);
    rewriter.replaceOp(op, var);
    return success();
  }
};

struct ConvertStore final : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getMemref());
    auto pointerValue =
        dyn_cast<TypedValue<emitc::PointerType>>(operands.getMemref());
    if (!arrayValue && !pointerValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    Value subscript;
    if (arrayValue) {
      subscript = rewriter.create<emitc::SubscriptOp>(op.getLoc(), arrayValue,
                                                      operands.getIndices());
    } else {
      // !! This is completely broken !! Just to see if it generates.
      // The indices need to be properly calculated
      subscript = rewriter.create<emitc::SubscriptOp>(op.getLoc(), pointerValue,
                                                      operands.getIndices()[0]);
    }
    auto store = rewriter.replaceOpWithNewOp<emitc::AssignOp>(
        op, subscript, operands.getValue());
    if (isa<emitc::PointerType>(store.getVar().getType())) {
      llvm::errs() << "Store: ";
      store.dump();
    }
    return success();
  }
};

struct ConvertCollapseShape final
    : public OpConversionPattern<memref::CollapseShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayValue = dyn_cast<TypedValue<emitc::ArrayType>>(operands.getSrc());
    if (!arrayValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, resultTy, operands.getSrc());
    return success();
  }
};

struct ConvertExpandShape final
    : public OpConversionPattern<memref::ExpandShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayValue = dyn_cast<TypedValue<emitc::ArrayType>>(operands.getSrc());
    auto pointerValue =
        dyn_cast<TypedValue<emitc::PointerType>>(operands.getSrc());
    if (!arrayValue && !pointerValue) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "expected array or pointer type");
    }

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, resultTy, operands.getSrc());
    return success();
  }
};

struct ConvertCast final : public OpConversionPattern<memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getSource());
    auto pointerValue =
        dyn_cast<TypedValue<emitc::PointerType>>(operands.getSource());
    if (!arrayValue && !pointerValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    auto resultTy = getTypeConverter()->convertType(op.getType());
    if (!resultTy) {
      return rewriter.notifyMatchFailure(op.getLoc(),
                                         "cannot convert result type");
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, resultTy,
                                               operands.getSource());
    return success();
  }
};

struct ConvertSubView final : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Check that the memory is sequential (incomplete)
    auto arrayValue =
        dyn_cast<TypedValue<emitc::ArrayType>>(operands.getSource());
    if (!arrayValue) {
      return rewriter.notifyMatchFailure(op.getLoc(), "expected array type");
    }

    if (!operands.getOffsets().empty()) {
      return op->emitOpError("Expected the offset to be a integer constant!");
    }

    for (auto offset : operands.getStaticOffsets()) {
      if (offset != 0) {
        return op->emitOpError("Expected the offset to be zero!");
      }
    }

    if (!operands.getStrides().empty()) {
      return op->emitOpError("Expected the offset to be a integer constant!");
    }

    for (auto stride : operands.getStaticStrides()) {
      if (stride != 1) {
        return op->emitOpError("Expected the offset to be zero!");
      }
    }

    // If memory is sequential, blindly convert it to a pointer
    auto resultTy = emitc::PointerType::get(op.getType().getElementType());
    auto sub = rewriter.replaceOpWithNewOp<emitc::CastOp>(op, resultTy,
                                                          operands.getSource());
    llvm::errs() << "SubView: ";
    sub.dump();

    return success();
  }
};

struct ConvertDim final : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    // !! Also broken !!
    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(
        op, op.getType(), rewriter.getI64IntegerAttr(845));
    return success();
  }
};

} // namespace

void mlir::populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        if (/*!memRefType.hasStaticShape() ||
            memref<3x?xi64, strided<[845, 1]>> -> isIdentity() is false
            !memRefType.getLayout().isIdentity() ||*/
            memRefType.getRank() == 0) {
          return {};
        }
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType)
          return {};

        // C does not support arrays with arbitrary dynamic sizes. Model them
        // with a pointer for now. This makes it such that all downstream ops
        // need to implement the strides to walk over the outer dimensions.
        //
        // One simplication that can be made - in some cases - is to move the
        // dynamic dimension to the right-most dimension of the array. This way
        // we can iterate over the array using array subscription.
        if (memRefType.hasStaticShape()) {
          return emitc::ArrayType::get(memRefType.getShape(),
                                       convertedElementType);
        }

        return emitc::PointerType::get(convertedElementType);
      });
}

void mlir::populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                                   TypeConverter &converter) {
  patterns.add<ConvertAlloca, ConvertCollapseShape, ConvertExpandShape,
               ConvertGlobal, ConvertGetGlobal, ConvertLoad, ConvertStore,
               ConvertCast, ConvertSubView, ConvertDim>(converter,
                                                        patterns.getContext());
}
