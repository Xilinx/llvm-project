#include "mlir/Dialect/SCF/Transforms/Patterns.h"

using namespace mlir;
using namespace mlir::scf;

namespace {
struct ForallLowering : public OpRewritePattern<mlir::scf::ForallOp> {
  using OpRewritePattern<mlir::scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override;
};
struct ParallelLowering : public OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern<mlir::scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult ForallLowering::matchAndRewrite(ForallOp forallOp,
                                              PatternRewriter &rewriter) const {
  Location loc = forallOp.getLoc();
  if (!forallOp.getOutputs().empty())
    return rewriter.notifyMatchFailure(
        forallOp,
        "only fully bufferized scf.forall ops can be lowered to scf.parallel");

  // Convert mixed bounds and steps to SSA values.
  SmallVector<Value> lbs = getValueOrCreateConstantIndexOp(
      rewriter, loc, forallOp.getMixedLowerBound());
  SmallVector<Value> ubs = getValueOrCreateConstantIndexOp(
      rewriter, loc, forallOp.getMixedUpperBound());
  SmallVector<Value> steps =
      getValueOrCreateConstantIndexOp(rewriter, loc, forallOp.getMixedStep());

  // Create empty scf.parallel op.
  auto parallelOp = rewriter.create<ParallelOp>(loc, lbs, ubs, steps);
  rewriter.eraseBlock(&parallelOp.getRegion().front());
  rewriter.inlineRegionBefore(forallOp.getRegion(), parallelOp.getRegion(),
                              parallelOp.getRegion().begin());
  // Replace the terminator.
  rewriter.setInsertionPointToEnd(&parallelOp.getRegion().front());
  rewriter.replaceOpWithNewOp<scf::ReduceOp>(
      parallelOp.getRegion().front().getTerminator());

  // Erase the scf.forall op.
  rewriter.replaceOp(forallOp, parallelOp);
  return success();
}

LogicalResult
ParallelLowering::matchAndRewrite(ParallelOp parallelOp,
                                  PatternRewriter &rewriter) const {
  Location loc = parallelOp.getLoc();
  auto reductionOp = cast<ReduceOp>(parallelOp.getBody()->getTerminator());

  // For a parallel loop, we essentially need to create an n-dimensional loop
  // nest. We do this by translating to scf.for ops and have those lowered in
  // a further rewrite. If a parallel loop contains reductions (and thus returns
  // values), forward the initial values for the reductions down the loop
  // hierarchy and bubble up the results by modifying the "yield" terminator.
  SmallVector<Value, 4> iterArgs = llvm::to_vector<4>(parallelOp.getInitVals());
  SmallVector<Value, 4> ivs;
  ivs.reserve(parallelOp.getNumLoops());
  bool first = true;
  SmallVector<Value, 4> loopResults(iterArgs);
  for (auto [iv, lower, upper, step] :
       llvm::zip(parallelOp.getInductionVars(), parallelOp.getLowerBound(),
                 parallelOp.getUpperBound(), parallelOp.getStep())) {
    ForOp forOp = rewriter.create<ForOp>(loc, lower, upper, step, iterArgs);
    ivs.push_back(forOp.getInductionVar());
    auto iterRange = forOp.getRegionIterArgs();
    iterArgs.assign(iterRange.begin(), iterRange.end());

    if (first) {
      // Store the results of the outermost loop that will be used to replace
      // the results of the parallel loop when it is fully rewritten.
      loopResults.assign(forOp.result_begin(), forOp.result_end());
      first = false;
    } else if (!forOp.getResults().empty()) {
      // A loop is constructed with an empty "yield" terminator if there are
      // no results.
      rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
      rewriter.create<scf::YieldOp>(loc, forOp.getResults());
    }

    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // First, merge reduction blocks into the main region.
  SmallVector<Value> yieldOperands;
  yieldOperands.reserve(parallelOp.getNumResults());
  for (int64_t i = 0, e = parallelOp.getNumResults(); i < e; ++i) {
    Block &reductionBody = reductionOp.getReductions()[i].front();
    Value arg = iterArgs[yieldOperands.size()];
    yieldOperands.push_back(
        cast<ReduceReturnOp>(reductionBody.getTerminator()).getResult());
    rewriter.eraseOp(reductionBody.getTerminator());
    rewriter.inlineBlockBefore(&reductionBody, reductionOp,
                               {arg, reductionOp.getOperands()[i]});
  }
  rewriter.eraseOp(reductionOp);

  // Then merge the loop body without the terminator.
  Block *newBody = rewriter.getInsertionBlock();
  if (newBody->empty())
    rewriter.mergeBlocks(parallelOp.getBody(), newBody, ivs);
  else
    rewriter.inlineBlockBefore(parallelOp.getBody(), newBody->getTerminator(),
                               ivs);

  // Finally, create the terminator if required (for loops with no results, it
  // has been already created in loop construction).
  if (!yieldOperands.empty()) {
    rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
    rewriter.create<scf::YieldOp>(loc, yieldOperands);
  }

  rewriter.replaceOp(parallelOp, loopResults);

  return success();
}
} // namespace

void mlir::scf::populateSCFForAllExpandPatterns(RewritePatternSet &patterns) {
  patterns.add<ForallLowering, ParallelLowering>(patterns.getContext());
}
