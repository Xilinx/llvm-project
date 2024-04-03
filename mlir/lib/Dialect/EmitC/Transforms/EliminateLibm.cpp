//===- EliminateLibm.cpp - Replace Libm by Math.h Inclusion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that replaces C Libm standard library
// prototypes (e.g., inserted by the MathToLibm pass) by an inclusion of the
// <math.h> header.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_ELIMINATELIBM
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"
} // namespace emitc
} // namespace mlir

using namespace mlir;
using namespace emitc;

namespace {

/// Replace all Libm calls (where callee has `libm` attribute + no definition)
/// by opaque calls
struct OpacifyLibmCall : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {

    auto *st = SymbolTable::getNearestSymbolTable(callOp);
    auto *calleeSym = SymbolTable::lookupSymbolIn(st, callOp.getCallee());
    assert(isa<FunctionOpInterface>(calleeSym));
    FunctionOpInterface callee = dyn_cast<FunctionOpInterface>(calleeSym);

    if (!(callee->hasAttr("libm") && callee.isDeclaration()))
      return failure();

    auto opaqueCall = rewriter.create<emitc::CallOpaqueOp>(
        callOp->getLoc(), callOp.getResultTypes(), callOp.getCallee(),
        callOp.getArgOperands());

    rewriter.replaceOp(callOp, opaqueCall);
    return success();
  }
};

struct EliminateLibmPass
    : public emitc::impl::EliminateLibmBase<EliminateLibmPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    MLIRContext *context = rootOp->getContext();

    if (!llvm::isa<ModuleOp>(rootOp))
      return;

    ModuleOp module = dyn_cast<ModuleOp>(rootOp);

    // Find the first math.h inclusion
    SmallVector<func::FuncOp> libmPrototypes;
    module.walk([&libmPrototypes](func::FuncOp funcOp) {
      if (funcOp->hasAttr("libm") && funcOp.isDeclaration())
        libmPrototypes.push_back(funcOp);
    });

    if (libmPrototypes.empty())
      return;

    // Replace with a rewrite pattern
    RewritePatternSet opacifyLibmCallPatterns(context);
    opacifyLibmCallPatterns.add<OpacifyLibmCall>(
        opacifyLibmCallPatterns.getContext());

    if (failed(applyPatternsAndFoldGreedily(
            rootOp, std::move(opacifyLibmCallPatterns))))
      return signalPassFailure();

    // Check that none of the prototypes have users
    for (auto &proto : libmPrototypes) {
      assert(proto->getUsers().empty());
    }

    // This builder has insertion point at the first
    // occurrence of a libm function found
    auto firstLibmDecl = libmPrototypes.front();
    OpBuilder builder(firstLibmDecl);

    auto includeOp =
        builder.create<emitc::IncludeOp>(builder.getUnknownLoc(), "math.h");
    includeOp->setAttr("is_standard_include", UnitAttr::get(context));

    for (auto &proto : libmPrototypes) {
      proto->erase();
    }
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect>();
    registry.insert<func::FuncDialect>();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::emitc::createEliminateLibmPass() {
  return std::make_unique<EliminateLibmPass>();
}
