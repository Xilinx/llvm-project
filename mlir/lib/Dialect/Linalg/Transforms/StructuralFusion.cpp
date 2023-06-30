//===- StructuralFusion.cpp - linalg fused ops generation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg structural fusion pass. It creates
// linalg.subgraph ops based on external fusion recipes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"
#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_LINALGSTRUCTURALFUSION
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-structural-fusion"

using namespace mlir;
using namespace mlir::linalg;

/// Determines whether @p op matches @p filter .
static bool matches(Operation *op, OperatorClass filter) {
  using llvm::operator&;
  return (linalg::classifyOperator(op) & filter) != OperatorClass::None;
}

/// Determines whether @p target can be wrapped in a `linalg.subgraph` operation.
static bool canWrapInSubgraph(Operation *target) {
  if (!target)
    return false;

  // - target must have exactly 0 or 1 result
  switch (target->getNumResults()) {
  case 0:
    return true;

  case 1:
    // - the result must be any ranked tensor type
    return target->getResult(0).getType().isa<RankedTensorType>();

  default:
    return false;
  }
}

static void collectUses(Operation *root, Operation *op,
                        SmallVectorImpl<Value> &uses) {
  const auto isLocal = [&](Value value) {
    auto source = value.getDefiningOp();
    if (auto barg = value.dyn_cast<BlockArgument>())
      source = barg.getOwner()->getParentOp();
    while (source) {
      if (source == root)
        return true;
      source = source->getParentOp();
    }
    return false;
  };
  const auto unite = [&](Value operand) {
    if (isLocal(operand))
      return;
    if (llvm::find(uses, operand) != uses.end())
      return;
    uses.push_back(operand);
  };

  llvm::for_each(op->getOperands(), unite);

  for (auto &region : op->getRegions()) {
    for (auto &op : region.getOps()) {
      collectUses(root, &op, uses);
    }
  }
}

/// Wraps @p target in a `linalg.subgraph` operation.
///
/// See canWrapInSubgraph() on what scenarios this transformation is allowed in.
///
/// @pre      `canWrapInSubgraph(target)`
///
/// Given IR of the form:
/// ```mlir
/// %r = target %o_0, ...
/// ```
///
/// This produces:
/// ```mlir
/// %r = linalg.subgraph (%c_0 = %o_0, ...) {
///   %w = target %c_0, ...
///   linalg.yield %w
/// }
/// ```
static SubgraphOp wrapInSubgraph(Operation *target) {
  assert(canWrapInSubgraph(target));

  SmallVector<Value> captures;
  collectUses(target, target, captures);

  // Create the Subgraph.
  OpBuilder builder(target);
  auto subgraph = builder.create<SubgraphOp>(
      target->getLoc(),
      /*resulType=*/target->getNumResults() ? target->getResult(0).getType()
                                            : Type{},
      /*captures=*/captures,
      [=](OpBuilder &builder, Location loc, IRMapping &captures) {
        // Clone the target op into the Subgraph.
        auto clonedOp = builder.insert(target->clone(captures));
        // Yield the result of the cloned op (which may have none).
        builder.create<linalg::YieldOp>(target->getLoc(),
                                        clonedOp->getResults());
      });

  // Remove the target operation.
  target->replaceAllUsesWith(subgraph);
  target->erase();
  return subgraph;
}

/// Determines whether @p target contains wrapped op that can be unwrapped.
static bool canUnwrapSubgraph(SubgraphOp target) {
  auto body = &target.getBody().front();

  // - exactly one wrapped op must be contained (plus terminator)
  if (body->getOperations().size() != 2)
    return false;

  // - the terminator must yield the result of that op.
  auto terminator = body->getTerminator();
  auto wrapped = &body->getOperations().front();
  return llvm::equal(terminator->getOperands(), wrapped->getResults());
}

/// Unwraps the single op inside the `linalg.subgraph` @p target op.
///
/// See canUnwrapSubgraph() on what scenarios this transformation is allowed in.
///
/// @pre      `canUnwrapSubgraph(target)`
///
/// Essentially exactly undoes what wrapInSubgraph() does.
static Operation *unwrapSubgraph(SubgraphOp target) {
  assert(canUnwrapSubgraph(target));

  // Compute the mapping from captured arguments to operands.
  auto unCaptureMapping = target.getUnCaptureMapping();

  // Clone the wrapped op out of the Subgraph.
  OpBuilder builder(target);
  auto unwrapped = builder.insert(
      target.getBody().front().getOperations().front().clone(unCaptureMapping));

  // Remove the target operation
  target.replaceAllUsesWith(unwrapped);
  target.erase();
  return unwrapped;
}

/// Determines whether @p producer is duplicated when prepended to @p target .
static bool mustDuplicateProducer(SubgraphOp target, Operation *producer) {
  assert(producer);

  // Duplicates if there is one user of a result of producer that is not target.
  return llvm::any_of(producer->getResults(), [=](Value value) {
    return llvm::any_of(value.getUsers(),
                        [=](auto user) { return user != target; });
  });
}

/// Determines whether @p producer can be prepended to @p target .
static bool canFuseProducer(SubgraphOp target, Operation *producer) {
  if (!producer)
    return false;

  // - producer may not be a user of target
  if (llvm::find(target->getUsers(), producer) != target->getUsers().end())
    return false;

  // - if producer will be duplicated, it must not have any side-effects
  return !mustDuplicateProducer(target, producer) ||
         isMemoryEffectFree(producer);
}

/// Gets the index of the result that @p value has at the defining op.
static unsigned resultIndex(Value value) {
  auto parent = value.getDefiningOp();
  for (unsigned idx = 0; idx < parent->getNumResults(); ++idx) {
    if (parent->getResult(idx) == value)
      return idx;
  }

  llvm_unreachable("value not produced by defining op");
}

/// Fuses @p producer into @p target .
///
/// See canFuseProducer() on what scenarios this transformation is allowed in.
///
/// @pre      `canFuseProducer(target, producer)`
///
/// Given IR of the form:
/// ```mlir
/// %r_0, ..., %r_N = producer %o_0, ...
/// %z = linalg.subgraph (%x_0 = %y_0, ..., %x_i = %r_i, ...) {
///   ...
/// }
/// ```
///
/// This produces:
/// ```mlir
/// %z = linalg.subgraph (%x_0 = %y_0, ..., %o_0, ...) {
///   %x_i, ... = producer %o_0, ...
///   ...
/// }
/// ```
///
/// If @p producer had any users beside @p target , a duplicate of the operation
/// will remain. canFuseProducer() therefore checks if this does not cause
/// added side-effects.
static SubgraphOp fuseProducer(SubgraphOp target, Operation *producer) {
  assert(canFuseProducer(target, producer));

  // Determine which captures will still be needed and what is dropped.
  SmallVector<Value> newCaptures;
  DenseMap<BlockArgument, unsigned> argToResultIdx;
  for (unsigned idx = 0; idx < target.getCaptures().size(); ++idx) {
    auto capture = target.getCaptures()[idx];
    if (capture.getDefiningOp() == producer) {
      // Capture will be replaced by op result.
      argToResultIdx.try_emplace(target.getCaptureArgs()[idx],
                                 resultIndex(capture));
      continue;
    }

    // Capture is still needed.
    newCaptures.push_back(capture);
  }

  // Ensure all operands to producer are captured.
  collectUses(producer, producer, newCaptures);

  // Create the new Subgraph.
  OpBuilder builder(target);
  auto result = builder.create<SubgraphOp>(
      target.getLoc(),
      /*resultType=*/target.getResult() ? target.getResult().getType() : Type{},
      newCaptures,
      [&](OpBuilder &builder, Location loc, IRMapping &captures) {
        // Clone the op to be prepended into the new block.
        auto prepended = builder.insert(producer->clone(captures));

        // Remap the capture arguments of the old Subgraph.
        for (unsigned idx = 0; idx < target.getCaptures().size(); ++idx) {
          captures.map(target.getCaptureArgs()[idx],
                       captures.lookup(target.getCaptures()[idx]));
        }

        // Update the capture mapping with results of the prepended op.
        for (auto &pair : argToResultIdx) {
          captures.map(pair.getFirst(), prepended->getResult(pair.getSecond()));
        }

        // Clone the contents of the old body.
        for (auto &op : target.getBody().front()) {
          builder.insert(op.clone(captures));
        }
      });

  // Remove the target operation.
  target.replaceAllUsesWith(result);
  target.erase();

  // If now unused, remove the producer operation.
  if (producer->getUsers().empty())
    producer->erase();
  return result;
}

/// Determines whether @p consumer can be appended to @p target .
static bool canFuseConsumer(SubgraphOp target, Operation *consumer) {
  if (!consumer)
    return false;

  // - target may not be a user of consumer
  if (llvm::find(consumer->getUsers(), target) != consumer->getUsers().end())
    return false;

  // - consumer must be the only user of target (if any).
  if (llvm::any_of(target->getUsers(),
                   [&](auto user) { return user != consumer; }))
    return false;

  // - consumer must be wrappable
  return canWrapInSubgraph(consumer);
}

/// Computes the length of the longest path from \p op to the function entry
static size_t computeAncestorLength(Operation *op) {
  if (!op)
    return 0;

  // Cache the results to avoid re-computation
  static DenseMap<Operation *, size_t> lengthCache;

  // Check cache before starting the computation
  auto it = lengthCache.find(op);
  if (it != lengthCache.end()) {
    return it->second;
  }

  size_t currentLength = 0;
  for (unsigned idx = 0; idx < op->getNumOperands(); ++idx) {
    Operation *ancestor = op->getOperand(idx).getDefiningOp();
    currentLength =
        std::max(currentLength, 1 + computeAncestorLength(ancestor));
  }

  // Update cache with computed value
  lengthCache.insert({op, currentLength});
  return currentLength;
}

/// Fused @p consumer into @p target .
///
/// See canFuseConsumer() on what scenarios this transformation is allowed in.
///
/// @pre      `canFuseConsumer(target, consumer)`
///
/// Given IR of the form:
/// ```mlir
/// %z = linalg.subgraph (%x_0 = %y_0, ...) {
///   ...
/// }
/// %r = consumer %o_0, ..., %z, ...
/// ```
///
/// This produces:
/// ```mlir
/// %r = linalg.subgraph (%x_0 = %y_0, ..., %o_0, ...) {
///   ...
///   %r = consumer %o_0, ..., %z, ...
///   linalg.yield %r
/// }
/// ```
static SubgraphOp fuseConsumer(SubgraphOp target, Operation *consumer) {
  assert(canFuseConsumer(target, consumer));

  // Ensure that all operands of consumer are captured.
  auto newCaptures = llvm::to_vector(target.getCaptures());
  collectUses(consumer, consumer, newCaptures);
  llvm::erase_if(newCaptures,
                 [&](Value capture) { return capture == target.getResult(); });

  // Create the new Subgraph.
  OpBuilder builder(consumer);
  auto result = builder.create<SubgraphOp>(
      target.getLoc(),
      /*resultType=*/consumer->getNumResults()
          ? consumer->getResult(0).getType()
          : Type{},
      newCaptures,
      [&](OpBuilder &builder, Location loc, IRMapping &captures) {
        // Remap the capture arguments of the old Subgraph.
        for (unsigned idx = 0; idx < target.getCaptures().size(); ++idx) {
          captures.map(target.getCaptureArgs()[idx],
                       captures.lookup(target.getCaptures()[idx]));
        }

        // Clone the contents of the old body (without the terminator).
        for (auto &op : target.getBody().front().without_terminator()) {
          builder.insert(op.clone(captures));
        }

        // If the old fused op had a result, add it to the mapping.
        auto terminator = target.getBody().front().getTerminator();
        if (terminator->getNumOperands())
          captures.map(target.getResult(),
                       captures.lookup(terminator->getOperand(0)));

        // Clone the op to be appended to the new block.
        auto appended = builder.insert(consumer->clone(captures));
        // Yield the result of the cloned op (which may have none).
        builder.create<linalg::YieldOp>(target->getLoc(),
                                        appended->getResults());
      });

  // Remove the consumer and then the target operation.
  consumer->replaceAllUsesWith(result);
  consumer->erase();
  target.erase();
  return result;
}

namespace {

struct Tactic {
  enum class Method {
    None = 0,
    Seed,
    FuseProducers,
    FuseProducersGreedy,
    FuseConsumers,
    FuseConsumersGreedy,
    FuseLongestAncestorConsumer,
    Dissolve
  };

  Method method;
  OperatorClass filter;
};

using Strategy = SmallVector<Tactic>;

static FailureOr<Tactic> parseTactic(StringRef string) {
  const auto symbolize = [](StringRef item) {
    return llvm::StringSwitch<std::optional<Tactic::Method>>(item)
        .Case("seed", Tactic::Method::Seed)
        .Case("producers", Tactic::Method::FuseProducers)
        .Case("producers*", Tactic::Method::FuseProducersGreedy)
        .Case("consumers", Tactic::Method::FuseConsumers)
        .Case("consumers*", Tactic::Method::FuseConsumersGreedy)
        .Case("longest_consumer", Tactic::Method::FuseLongestAncestorConsumer)
        .Case("dissolve", Tactic::Method::Dissolve)
        .Default(std::nullopt);
  };

  auto [methodStr, tail] = string.split('(');
  if (tail.empty())
    return failure();

  auto method = symbolize(methodStr);
  if (!method.has_value())
    return failure();

  auto [filterStr, rest] = tail.split(')');
  if (!rest.empty())
    return failure();

  FailureOr<OperatorClass> filter;
  if (filterStr.empty()) {
    filter = OperatorClass::None;
  } else {
    filter = parseOperatorClass(filterStr);
    if (failed(filter))
      return failure();
  }

  return Tactic{method.value(), filter.value()};
}

static FailureOr<Strategy> parseStrategy(ArrayRef<std::string> tactics) {
  Strategy result;
  for (auto &tactic : tactics) {
    auto parsed = parseTactic(tactic);
    if (failed(parsed))
      return failure();
    result.push_back(parsed.value());
  }
  return std::move(result);
}

struct LinalgStructuralFusionPass
    : public impl::LinalgStructuralFusionBase<LinalgStructuralFusionPass> {
  void runOnOperation() override {
    if (!strategy) {
      if (!strategyString.empty()) {
        auto parsed = parseStrategy(strategyString);
        if (failed(parsed)) {
          llvm::WithColor::error()
              << "[linalg-structural-fusion] error parsing strategy\n";
          signalPassFailure();
          return;
        }

        strategy = std::move(parsed.value());
      } else {
        static Strategy defaultStrategy;
        if (defaultStrategy.empty()) {
          // We use a double-seeded approach. For the first one we use:
          //  1. Start with all pooling operations
          //  2. Fuse them with producers: paddings, activations, convolutions,
          //     broadcasts and constants and finally paddings again.
          //
          // We fuse paddings twice to first fuse them with poolings and then
          // a second time to fuse them with convolutions.
          //
          // For the 2nd seed, we use the following tactics:
          //  1. Start with all convolutions
          //  2. Fuse them with producers: broadcasts, constant, paddings
          //  3. Fuse them with elementwise operations (e.g. tensor.add) if they
          //     are on the longest path to the function entry (longest ancestor
          //     consumer fusion)
          //  4. Fuse activation consumers
          //  5. Fuse constant producers again, since some additional constants
          //     might be fusable after the activations have been fused (e.g.
          //     constants used in LeakyRelU)
          //
          // The 2nd seed is required as we might have convolutions that are not
          // consumed by poolings. The first seed would not catch those.

          // 1st Seed
          defaultStrategy.push_back(
              Tactic{Tactic::Method::Seed, OperatorClass::Pooling});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Padding});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Activation});
          defaultStrategy.push_back(Tactic{Tactic::Method::FuseProducers,
                                           OperatorClass::Convolution});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Broadcast});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Constant});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Padding});

          // 2nd Seed
          defaultStrategy.push_back(
              Tactic{Tactic::Method::Seed, OperatorClass::Convolution});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Broadcast});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Constant});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Padding});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseLongestAncestorConsumer,
                     OperatorClass::Elementwise});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseConsumers, OperatorClass::Activation});
          defaultStrategy.push_back(
              Tactic{Tactic::Method::FuseProducers, OperatorClass::Constant});
        }

        strategy = defaultStrategy;
      }
    }

    apply(strategy.value());
  }

private:
  std::optional<Strategy> strategy;

  using WorkingSet = llvm::SmallVector<SubgraphOp>;

  // Check whether \p target is the longest ancestor of \p consumer. Ancestors
  // that are not a fused operation or are not part of \p work are ignored.
  bool isLongestAncestor(WorkingSet &work, SubgraphOp target,
                         Operation *consumer) {
    if (!consumer)
      return false;

    // Keep track of the longest path and the Subgraph linked to it
    SubgraphOp longestAncestorFuse = nullptr;
    size_t longestAncestorLength = 0;
    for (unsigned idx = 0; idx < consumer->getNumOperands(); ++idx) {
      Operation *ancestor = consumer->getOperand(idx).getDefiningOp();

      // Skip any operands that are not a fused op or not part of the worklist
      auto fusedAncestor = dyn_cast_or_null<SubgraphOp>(ancestor);
      if (!fusedAncestor || llvm::find(work, fusedAncestor) == work.end())
        continue;

      unsigned currentAncestorLength = computeAncestorLength(fusedAncestor);

      // Update if a longer path was found
      if (currentAncestorLength > longestAncestorLength) {
        longestAncestorLength = currentAncestorLength;
        longestAncestorFuse = fusedAncestor;
      }
    }

    // target is the longest ancestor if it is the same Subgraph that was found
    // to have the longest path
    assert(longestAncestorFuse && "No ancestor found!");
    return longestAncestorFuse == target;
  }

  size_t seed(WorkingSet &work, OperatorClass filter) {
    using llvm::operator&;

    size_t result = 0;
    work.clear();
    getOperation().walk([&](Operation *op, const WalkStage &stage) {
      if (isa<SubgraphOp>(op)) {
        // Do not look into fused ops.
        return WalkResult::skip();
      }

      if (matches(op, filter)) {
        // Seed this op, but do not look into it.
        if (canWrapInSubgraph(op)) {
          work.push_back(wrapInSubgraph(op));
          ++result;
        }
        return WalkResult::skip();
      }

      return WalkResult::advance();
    });

    return result;
  }

  FailureOr<SubgraphOp> fuseFirstProducer(SubgraphOp target, OperatorClass filter) {
    for (auto op : target.getOperands()) {
      auto candidate = op.getDefiningOp();
      if (!candidate || !matches(candidate, filter) ||
          !canFuseProducer(target, candidate))
        continue;

      return ::fuseProducer(target, candidate);
    }

    return failure();
  }

  FailureOr<SubgraphOp> fuseProducers(SubgraphOp target, OperatorClass filter) {
    auto result = false;
    while (true) {
      auto fused = fuseFirstProducer(target, filter);
      if (succeeded(fused)) {
        target = fused.value();
        result = true;
        continue;
      }

      if (result)
        return target;

      return failure();
    }
  }

  size_t fuseProducers(WorkingSet &work, OperatorClass filter) {
    WorkingSet results;
    llvm::erase_if(work, [&](SubgraphOp op) {
      auto fused = fuseProducers(op, filter);
      if (failed(fused))
        return false;

      results.push_back(fused.value());
      return true;
    });
    work.append(results);
    return results.size();
  }

  size_t fuseProducersGreedy(WorkingSet &work, OperatorClass filter) {
    size_t result = 0;
    while (auto add = fuseProducers(work, filter)) {
      result += add;
    }
    return result;
  }

  FailureOr<SubgraphOp> fuseConsumer(SubgraphOp target, OperatorClass filter,
                                  WorkingSet &work, bool longestAncestor) {
    if (!target.getResult())
      return failure();

    assert(!target.getResult().getUsers().empty());
    Operation *candidate = *target.getResult().getUsers().begin();
    if (!matches(candidate, filter) || !canFuseConsumer(target, candidate) ||
        (longestAncestor && !isLongestAncestor(work, target, candidate)))
      return failure();

    return ::fuseConsumer(target, candidate);
  }

  /// Fuse all consumers of a fused operation that match the given \p filter.
  /// If there is more than one consumer, no fusion is applied.
  ///
  /// If \p longestAncestor is set, the consumer is only fused to the fused
  /// operation with the longest path to a shared input or function entry. For
  /// consumers with only a single input, there is trivially only one path and
  /// setting this flag will have no effect.
  /// Consumers with multiple inputs have their inputs traversed until either a
  /// shared input or the function entry is reached. The consumer is fused with
  /// the fused operation that has the longer path. Inputs that are not fused
  /// operations or are not in \p work are ignored.
  size_t fuseConsumers(WorkingSet &work, OperatorClass filter,
                       bool longestAncestor = false) {
    WorkingSet results;
    llvm::erase_if(work, [&](SubgraphOp op) {
      auto fused = fuseConsumer(op, filter, work, longestAncestor);
      if (failed(fused))
        return false;

      results.push_back(fused.value());
      return true;
    });
    work.append(results);
    return results.size();
  }

  size_t fuseConsumersGreedy(WorkingSet &work, OperatorClass filter) {
    size_t result = 0;
    while (auto add = fuseConsumers(work, filter)) {
      result += add;
    }
    return result;
  }

  size_t dissolve(WorkingSet &work) {
    size_t result = 0;
    llvm::erase_if(work, [&](SubgraphOp op) {
      if (!canUnwrapSubgraph(op))
        return false;

      unwrapSubgraph(op);
      ++result;
      return true;
    });
    return result;
  }

  size_t apply(WorkingSet &work, const Tactic &tactic) {
    switch (tactic.method) {
    case Tactic::Method::Seed:
      return seed(work, tactic.filter);
    case Tactic::Method::FuseProducers:
      return fuseProducers(work, tactic.filter);
    case Tactic::Method::FuseProducersGreedy:
      return fuseProducersGreedy(work, tactic.filter);
    case Tactic::Method::FuseConsumers:
      return fuseConsumers(work, tactic.filter);
    case Tactic::Method::FuseConsumersGreedy:
      return fuseConsumersGreedy(work, tactic.filter);
    case Tactic::Method::FuseLongestAncestorConsumer:
      return fuseConsumers(work, tactic.filter, /* longestAncestor */ true);
    case Tactic::Method::Dissolve:
      return dissolve(work);
    default:
      llvm_unreachable("unknown tactic");
    }
  }

  size_t apply(WorkingSet &work, const Strategy &strategy) {
    size_t result = 0;
    for (auto &tactic : strategy) {
      result += apply(work, tactic);
    }
    return result;
  }

  size_t apply(const Strategy &strategy) {
    WorkingSet work;
    return apply(work, strategy);
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLinalgStructuralFusionPass() {
  return std::make_unique<LinalgStructuralFusionPass>();
}
