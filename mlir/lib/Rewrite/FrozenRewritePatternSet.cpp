//===- FrozenRewritePatternSet.cpp - Frozen Pattern List -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "ByteCode.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <mlir/Dialect/PDL/IR/Builtins.h>
#include <optional>

using namespace mlir;

// Include the PDL rewrite support.
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
#include "mlir/Conversion/PDLToPDLInterp/PDLToPDLInterp.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"

static LogicalResult
convertPDLToPDLInterp(ModuleOp pdlModule,
                      DenseMap<Operation *, PDLPatternConfigSet *> &configMap) {
  // Skip the conversion if the module doesn't contain pdl.
  if (pdlModule.getOps<pdl::PatternOp>().empty())
    return success();

  // Simplify the provided PDL module. Note that we can't use the canonicalizer
  // here because it would create a cyclic dependency.
  auto simplifyFn = [](Operation *op) {
    // TODO: Add folding here if ever necessary.
    if (isOpTriviallyDead(op))
      op->erase();
  };
  pdlModule.getBody()->walk(simplifyFn);

  /// Lower the PDL pattern module to the interpreter dialect.
  PassManager pdlPipeline(pdlModule->getName());
#ifdef NDEBUG
  // We don't want to incur the hit of running the verifier when in release
  // mode.
  pdlPipeline.enableVerifier(false);
#endif
  pdlPipeline.addPass(createPDLToPDLInterpPass(configMap));
  if (failed(pdlPipeline.run(pdlModule)))
    return failure();

  // Simplify again after running the lowering pipeline.
  pdlModule.getBody()->walk(simplifyFn);
  return success();
}
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

class PdllRewritePattern : public RewritePattern {
  mutable PDLPatternModule pdllModule;
  class State {
  public:
    // using Payload = llvm::PointerUnion<Operation *, Attribute, Value>;
    using Payload = PDLValue;
    DenseMap<Value, PDLValue> s;
    void add(Value constraint, PDLValue target) {
      Payload p = target;
      s.try_emplace(constraint, p);
    }
    void add(ResultRange results, ArrayRef<PDLValue> targets) {
      for (auto it : llvm::zip(results, targets)) {
        add(std::get<0>(it), std::get<1>(it));
      }
    }
    PDLValue get(Value constraint) { return s.at(constraint); }
  };

public:
  class MyPDLResultList : public PDLResultList {
  public:
    MyPDLResultList(unsigned maxNumResults) : PDLResultList(maxNumResults) {}

    /// Return the list of PDL results.
    MutableArrayRef<PDLValue> getResults() { return results; }
  };

  PdllRewritePattern(PDLPatternModule pdllModule)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1,
                       pdllModule.getModule().getContext()),
        pdllModule(std::move(pdllModule)) {}
  FailureOr<State> tryMatch(PatternRewriter &rewriter, pdl::PatternOp pattern,
                            Operation *target) const {

    State state;
    for (Operation &constraint :
         pattern.getBodyRegion().front().without_terminator()) {
      assert(!isa<pdl::RewriteOp>(constraint));
      if (auto operation = dyn_cast<pdl::OperationOp>(constraint)) {
        if (auto name = operation.getOpName()) {
          llvm::errs() << "name: " << target->getName().getStringRef() << " "
                       << *name << "\n";
          if (target->getName().getStringRef() != *operation.getOpName())
            return rewriter.notifyMatchFailure(target, "name doesn't match");
        }
        for (auto it : llvm::zip(operation.getAttributeValueNames(),
                                 operation.getAttributeValues())) {
          auto name = cast<StringAttr>(std::get<0>(it)).getValue();
          auto val = state.get(std::get<1>(it)).cast<Attribute>();
          if (target->getAttr(name) != val) {
            return rewriter.notifyMatchFailure(
                target, Twine("attr doesn't match: ") + name);
          }
        }
        state.add(operation.getResult(), target);
      } else if (auto operands = dyn_cast<pdl::OperandsOp>(constraint)) {
        state.add(operands.getResult(), nullptr);
      } else if (auto type = dyn_cast<pdl::TypeOp>(constraint)) {
        state.add(type.getResult(), nullptr);
      } else if (auto attr = dyn_cast<pdl::AttributeOp>(constraint)) {
        if (auto val = attr.getValue()) {
          state.add(attr.getResult(), *val);
        } else {
          state.add(attr.getResult(), nullptr);
        }
        // TODO: use attribute value type
      } else if (auto result = dyn_cast<pdl::ResultOp>(constraint)) {
        state.add(result.getResult(), state.get(result.getParent())
                                          .cast<Operation *>()
                                          ->getResult(result.getIndex()));
      } else if (auto applyNativeConstraint =
                     dyn_cast<pdl::ApplyNativeConstraintOp>(constraint)) {
        // MyPDLResultList results(applyNativeConstraint->getNumResults());
        auto fn = pdllModule.getConstraintFunctions().at(
            applyNativeConstraint.getName());
        SmallVector<PDLValue> args;
        for (auto arg : applyNativeConstraint->getOperands()) {
          args.push_back(state.get(arg));
        }
        if (failed(
                fn(rewriter, args)) /*!= applyNativeConstraint.getIsNegated()*/)
          return rewriter.notifyMatchFailure(
              target, Twine("Native constraint failed: ") +
                          applyNativeConstraint.getName());

        // state.add(applyNativeConstraint.getResults(), results.getResults());
      } else {
        constraint.dump();
        assert(false);
      }
    }

    return state;
  }

  LogicalResult applyRewrite(Operation *op, PatternRewriter &rewriter,
                             pdl::RewriteOp rewrite, State &state) const {
    for (Operation &child :
         rewrite.getBodyRegion().front().without_terminator()) {
      if (auto applyNativeRewrite =
              dyn_cast<pdl::ApplyNativeRewriteOp>(child)) {
        MyPDLResultList results(applyNativeRewrite->getNumResults());
        auto fn =
            pdllModule.getRewriteFunctions().at(applyNativeRewrite.getName());
        SmallVector<PDLValue> args;
        for (auto arg : applyNativeRewrite->getOperands()) {
          args.push_back(state.get(arg));
        }
        if (failed(fn(rewriter, results, args)))
          assert(false && "Native rewrite failed");

        state.add(applyNativeRewrite.getResults(), results.getResults());
      } else if (auto attr = dyn_cast<pdl::AttributeOp>(child)) {
        assert(attr.getValue());
        state.add(attr.getResult(), *attr.getValue());
      } else {
        child.dump();
        assert(false);
      }
    }
    return success();
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    pdl::RewriteOp rewrite;
    State bestState;
    int32_t maxBenefit = -1;
    pdllModule.getModule().walk([&](pdl::PatternOp pattern) {
      FailureOr<State> state = tryMatch(rewriter, pattern, op);
      if (failed(state))
        return;
      if ((int32_t)pattern.getBenefit() > maxBenefit) {
        maxBenefit = (int32_t)pattern.getBenefit();
        rewrite = pattern.getRewriter();
        bestState = *state;
      }
    });
    if (rewrite) {
      return applyRewrite(op, rewriter, rewrite, bestState);
    }
    return rewriter.notifyMatchFailure(op, "No pattern matched");
  }
};
//===----------------------------------------------------------------------===//
// FrozenRewritePatternSet
//===----------------------------------------------------------------------===//

FrozenRewritePatternSet::FrozenRewritePatternSet()
    : impl(std::make_shared<Impl>()) {}

FrozenRewritePatternSet::FrozenRewritePatternSet(
    RewritePatternSet &&patterns, ArrayRef<std::string> disabledPatternLabels,
    ArrayRef<std::string> enabledPatternLabels)
    : impl(std::make_shared<Impl>()) {
  DenseSet<StringRef> disabledPatterns, enabledPatterns;
  disabledPatterns.insert(disabledPatternLabels.begin(),
                          disabledPatternLabels.end());
  enabledPatterns.insert(enabledPatternLabels.begin(),
                         enabledPatternLabels.end());

  // Functor used to walk all of the operations registered in the context. This
  // is useful for patterns that get applied to multiple operations, such as
  // interface and trait based patterns.
  std::vector<RegisteredOperationName> opInfos;
  auto addToOpsWhen =
      [&](std::unique_ptr<RewritePattern> &pattern,
          function_ref<bool(RegisteredOperationName)> callbackFn) {
        if (opInfos.empty())
          opInfos = pattern->getContext()->getRegisteredOperations();
        for (RegisteredOperationName info : opInfos)
          if (callbackFn(info))
            impl->nativeOpSpecificPatternMap[info].push_back(pattern.get());
        impl->nativeOpSpecificPatternList.push_back(std::move(pattern));
      };

  for (std::unique_ptr<RewritePattern> &pat : patterns.getNativePatterns()) {
    // Don't add patterns that haven't been enabled by the user.
    if (!enabledPatterns.empty()) {
      auto isEnabledFn = [&](StringRef label) {
        return enabledPatterns.count(label);
      };
      if (!isEnabledFn(pat->getDebugName()) &&
          llvm::none_of(pat->getDebugLabels(), isEnabledFn))
        continue;
    }
    // Don't add patterns that have been disabled by the user.
    if (!disabledPatterns.empty()) {
      auto isDisabledFn = [&](StringRef label) {
        return disabledPatterns.count(label);
      };
      if (isDisabledFn(pat->getDebugName()) ||
          llvm::any_of(pat->getDebugLabels(), isDisabledFn))
        continue;
    }

    if (std::optional<OperationName> rootName = pat->getRootKind()) {
      impl->nativeOpSpecificPatternMap[*rootName].push_back(pat.get());
      impl->nativeOpSpecificPatternList.push_back(std::move(pat));
      continue;
    }
    if (std::optional<TypeID> interfaceID = pat->getRootInterfaceID()) {
      addToOpsWhen(pat, [&](RegisteredOperationName info) {
        return info.hasInterface(*interfaceID);
      });
      continue;
    }
    if (std::optional<TypeID> traitID = pat->getRootTraitID()) {
      addToOpsWhen(pat, [&](RegisteredOperationName info) {
        return info.hasTrait(*traitID);
      });
      continue;
    }
    impl->nativeAnyOpPatterns.push_back(std::move(pat));
  }

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  // Generate the bytecode for the PDL patterns if any were provided.
  PDLPatternModule &pdlPatterns = patterns.getPDLPatterns();
  ModuleOp pdlModule = pdlPatterns.getModule();
  if (!pdlModule)
    return;
  /*if (failed(convertPDLToPDLInterp(pdlModule, configMap)))
    llvm::report_fatal_error(
        "failed to lower PDL pattern module to the PDL Interpreter");
*/
  pdl::registerBuiltins(pdlPatterns);

  std::unique_ptr<PdllRewritePattern> pattern =
      RewritePattern::create<PdllRewritePattern>(std::move(pdlPatterns));
  impl->nativeAnyOpPatterns.push_back(std::move(pattern));

  // Generate the pdl bytecode.
  /*impl->pdlByteCode = std::make_unique<detail::PDLByteCode>(
      pdlModule, pdlPatterns.takeConfigs(), configMap,
      pdlPatterns.takeConstraintFunctions(),
      pdlPatterns.takeRewriteFunctions());*/
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
}

FrozenRewritePatternSet::~FrozenRewritePatternSet() = default;
