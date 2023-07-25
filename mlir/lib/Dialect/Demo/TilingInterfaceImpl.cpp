#include "mlir/Dialect/Demo/TilingInterfaceImpl.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Demo/DemoDialect.h"
#include "mlir/Dialect/Demo/DemoOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include <optional>

using namespace mlir;
using namespace mlir::demo;

#define DEBUG_TYPE "demo-tilinginterface"

//===----------------------------------------------------------------------===//
// Below functions are from LinAlg's utility library, changed to pass-through
// information that is gathered from the LinAlg op interface otherwise.
//===----------------------------------------------------------------------===//

SmallVector<std::optional<linalg::SliceParameters>> computeAllSliceParameters(
    OpBuilder &builder, Location loc, ArrayRef<OpOperand> opOperands,
    ArrayRef<AffineMap> indexingMaps, ValueRange valuesToTile,
    ArrayRef<OpFoldResult> ivs, ArrayRef<OpFoldResult> tileSizes,
    ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck) {
  assert(ivs.size() == static_cast<size_t>(llvm::count_if(
                           llvm::make_range(tileSizes.begin(), tileSizes.end()),
                           [](OpFoldResult v) { return !isZeroIndex(v); })) &&
         "expected as many ivs as non-zero sizes");

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<OpFoldResult> lbs =
      linalg::computeTileOffsets(builder, loc, ivs, tileSizes);
  SmallVector<OpFoldResult> subShapeSizes =
      linalg::computeTileSizes(builder, loc, tileSizes, sizeBounds);

  assert(valuesToTile.size() <= opOperands.size() &&
         "more value to tile than operands.");
  SmallVector<std::optional<linalg::SliceParameters>> allSliceParams;
  allSliceParams.reserve(valuesToTile.size());

  LLVM_DEBUG({
    llvm::dbgs() << "computeAllSliceParameters\n";
    for (auto [opOperand, map, val] :
         llvm::zip(opOperands, indexingMaps, valuesToTile)) {
      llvm::dbgs() << "  opOperand: ";
      opOperand.get().dump();

      llvm::dbgs() << "  indexingMap: ";
      map.dump();

      llvm::dbgs() << "  valuesToTile:\n";
      for (auto vtt : valuesToTile) {
        llvm::dbgs() << "    ";
        vtt.dump();
      }
    }
  });

  for (auto [opOperand, map, val] :
       llvm::zip(opOperands, indexingMaps, valuesToTile)) {
    Value shapedOp = val;
    // Use `opOperand` as is if it is not tiled and not an output tensor. Having
    // an extract/insert slice pair for all output tensors simplifies follow up
    // transformations such as padding and bufferization since the
    // extract/insert slice pairs make the accessed iteration argument
    // subdomains explicit.

    Type operandType = opOperand.get().getType();
    // if (!isTiled(map, tileSizes) && !(isa<RankedTensorType>(operandType) &&
    //                                   linalgOp.isDpsInit(&opOperand))) {
    if (!isTiled(map, tileSizes) && !(isa<RankedTensorType>(operandType))) {
      allSliceParams.push_back(std::nullopt);
      continue;
    }

    allSliceParams.push_back(linalg::computeSliceParameters(
        builder, loc, shapedOp, tileSizes, map, lbs, sizeBounds, subShapeSizes,
        omitPartialTileCheck));
  }

  return allSliceParams;
}

Value materializeTiledShape(OpBuilder &builder, Location loc, Value valueToTile,
                            const linalg::SliceParameters &sliceParams) {
  auto shapedType = dyn_cast<ShapedType>(valueToTile.getType());
  auto *sliceOp = TypeSwitch<ShapedType, Operation *>(shapedType)
                      .Case([&](MemRefType) {
                        return builder.create<memref::SubViewOp>(
                            loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Case([&](RankedTensorType) {
                        return builder.create<tensor::ExtractSliceOp>(
                            loc, valueToTile, sliceParams.offsets,
                            sliceParams.sizes, sliceParams.strides);
                      })
                      .Default([](ShapedType) -> Operation * {
                        llvm_unreachable("Unexpected shaped type");
                      });
  return sliceOp->getResult(0);
}

SmallVector<Value>
makeTiledShapes(OpBuilder &builder, Location loc,
                ArrayRef<OpOperand> opOperands,
                ArrayRef<AffineMap> indexingMaps, ValueRange valuesToTile,
                ArrayRef<OpFoldResult> ivs, ArrayRef<OpFoldResult> tileSizes,
                ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck) {
  SmallVector<std::optional<linalg::SliceParameters>> allSliceParameter =
      computeAllSliceParameters(builder, loc, opOperands, indexingMaps,
                                valuesToTile, ivs, tileSizes, sizeBounds,
                                omitPartialTileCheck);
  LLVM_DEBUG({
    llvm::dbgs() << "AllSliceParameter\n";
    for (auto &sp : allSliceParameter) {
      llvm::dbgs() << "  SliceParameter\n";
      llvm::dbgs() << "    offsets:\n";
      for (auto &off : sp->offsets) {
        llvm::dbgs() << "      " << off << "\n";
      }

      llvm::dbgs() << "    sizes:\n";
      for (auto &si : sp->sizes) {
        llvm::dbgs() << "      " << si << "\n";
      }

      llvm::dbgs() << "    strides:\n";
      for (auto &st : sp->strides) {
        llvm::dbgs() << "      " << st << "\n";
      }
    }
  });

  SmallVector<Value> tiledShapes;
  for (auto item : llvm::zip(valuesToTile, allSliceParameter)) {
    Value valueToTile = std::get<0>(item);
    std::optional<linalg::SliceParameters> sliceParams = std::get<1>(item);
    tiledShapes.push_back(
        sliceParams.has_value()
            ? materializeTiledShape(builder, loc, valueToTile, *sliceParams)
            : valueToTile);
  }
  return tiledShapes;
}

SmallVector<AffineExpr> getSymbolBindings(MLIRContext *context) {
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

SmallVector<Type> getTensorOutputTypes(ValueRange operands, int resultIdx) {
  SmallVector<Type> ret;
  ret.push_back(operands[resultIdx].getType());
  return ret;
}

//===----------------------------------------------------------------------===//
// Implementation of the TilingInterface for the Demo Dialect.
//===----------------------------------------------------------------------===//

namespace {
template <typename DemoOpTy>
struct DemoOpTilingInterface
    : public TilingInterface::ExternalModel<DemoOpTilingInterface<DemoOpTy>,
                                            DemoOpTy> {

  SmallVector<std::string> indexingMaps = {
      "affine_map<(d0, d1, d2) -> (d0, d2)>",
      "affine_map<(d0, d1, d2) -> (d2, d1)>",
      "affine_map<(d0, d1, d2) -> (d0, d1)>"};
  int resultIdx = 2;

  SmallVector<AffineMap> getIndexingMaps(MLIRContext *context) const {
    SmallVector<AffineMap> maps;
    auto symbolBindings = getSymbolBindings(context);
    for (auto &indexingMap : indexingMaps) {
      maps.push_back(
          llvm::cast<AffineMapAttr>(mlir::parseAttribute(indexingMap, context))
              .getValue());
      maps.back() = simplifyAffineMap(
          maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
    }
    return maps;
  }

  /// Return the loop iterator type.
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    LLVM_DEBUG(llvm::dbgs() << "<<<<<<<<<<<< getLoopIteratorTypes\n");
    assert(false && "getLoopIteratorTypes: Method not implemented.");
  }

  /// Return the iteration domain range.
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    LLVM_DEBUG(llvm::dbgs() << "<<<<<<<<<<<< getIterationDomain\n");

    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();

    SmallVector<AffineMap> indexingAffMaps = getIndexingMaps(context);
    AffineMap shapeToLoopsMap =
        inversePermutation(concatAffineMaps(indexingAffMaps));

    SmallVector<OpFoldResult> allShapesSizes;
    for (auto &operand : op->getOpOperands()) {
      auto shapedType = llvm::cast<ShapedType>(operand.get().getType());
      for (int64_t dim = 0; dim < shapedType.getRank(); ++dim) {
        allShapesSizes.push_back(
            linalg::createFoldedDimOp(b, loc, operand.get(), dim));
      }
    }

    return llvm::to_vector(
        llvm::map_range(shapeToLoopsMap.getResults(), [&](AffineExpr loopExpr) {
          OpFoldResult ofr = affine::makeComposedFoldedAffineApply(
              b, loc, loopExpr, allShapesSizes);
          return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
        }));
  }

  // Instantiate the tiled implementation of the operation.
  FailureOr<TilingResult>
  getTiledImplementation(Operation *op, OpBuilder &b,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes) const {
    LLVM_DEBUG(llvm::dbgs() << "<<<<<<<<<<<< getTiledImplementation\n");

    Location loc = op->getLoc();
    SmallVector<Value> valuesToTile = op->getOperands();

    MLIRContext *context = op->getContext();
    SmallVector<AffineMap> indexingAffMaps = getIndexingMaps(context);

    SmallVector<Value> tiledOperands =
        makeTiledShapes(b, loc, op->getOpOperands(), indexingAffMaps,
                        valuesToTile, offsets, sizes, {}, true);
    LLVM_DEBUG({
      llvm::dbgs() << "Tiled Operands\n";
      for (auto &tiledOperand : tiledOperands) {
        llvm::dbgs() << "  ";
        tiledOperand.dump();
      }
    });

    SmallVector<Type> resultTensorTypes =
        getTensorOutputTypes(tiledOperands, resultIdx);

    Operation *copiedOp = clone(b, op, resultTensorTypes, tiledOperands);
    return TilingResult{{copiedOp}, SmallVector<Value>(copiedOp->getResults())};
  }

  // Return the details of the output tile generated by the tiled
  // implementation.
  LogicalResult
  getResultTilePosition(Operation *op, OpBuilder &b, unsigned resultNumber,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes,
                        SmallVector<OpFoldResult> &resultOffsets,
                        SmallVector<OpFoldResult> &resultSizes) const {
    LLVM_DEBUG(llvm::dbgs() << "<<<<<<<<<<<< getResultTilePosition\n");

    Location loc = op->getLoc();

    AffineExpr d0;
    bindDims(b.getContext(), d0);
    SmallVector<OpFoldResult> subShapeSizes =
        llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
          return affine::makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
        }));

    OpOperand *outOperand = &op->getOpOperand(resultIdx);

    MLIRContext *context = op->getContext();
    SmallVector<AffineMap> indexingAffMaps = getIndexingMaps(context);

    linalg::SliceParameters sliceParams = linalg::computeSliceParameters(
        b, loc, outOperand->get(), sizes, indexingAffMaps[resultIdx], offsets,
        /*ubs*/ {}, subShapeSizes, true);
    resultOffsets = sliceParams.offsets;
    resultSizes = sliceParams.sizes;

    return success();
  }

  FailureOr<TilingResult>
  generateResultTileValue(Operation *op, OpBuilder &b, unsigned resultNumber,
                          ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> sizes) const {
    LLVM_DEBUG(llvm::dbgs() << "<<<<<<<<<<<< generateResultTileValue\n");
    assert(false && "generateResultTileValue: Method not implemented.");
  }

  LogicalResult generateScalarImplementation(Operation *op, OpBuilder &builder,
                                             Location loc,
                                             ValueRange ivs) const {
    LLVM_DEBUG(llvm::dbgs() << "<<<<<<<<<<<< generateScalarImplementation\n");
    assert(false && "generateScalarImplementation: Method not implemented.");
  }
};
} // namespace

template <typename OpType>
static void registerOne(MLIRContext *ctx) {
  OpType::template attachInterface<DemoOpTilingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerAll(MLIRContext *ctx) {
  (registerOne<OpTypes>(ctx), ...);
}

#define GET_OP_LIST

void mlir::demo::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, demo::DemoDialect *dialect) {
    registerAll<
#include "mlir/Dialect/Demo/DemoOps.cpp.inc"
        >(ctx);
  });
}
