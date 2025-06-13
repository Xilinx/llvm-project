//===- ConversionUtils.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions for TOSA lowering
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::tosa;

SmallVector<utils::IteratorType>
mlir::tosa::getNParallelLoopsAttrs(unsigned nParallelLoops) {
  return SmallVector<utils::IteratorType>(nParallelLoops,
                                          utils::IteratorType::parallel);
}

SmallVector<Value>
mlir::tosa::condenseValues(const SmallVector<Value> &values) {
  SmallVector<Value> condensedValues;
  for (auto value : values)
    if (value)
      condensedValues.push_back(value);
  return condensedValues;
}

Value mlir::tosa::clampFloatHelper(Location loc, Value arg, Value min,
                                   Value max, OpBuilder &rewriter) {
  Value minValue = rewriter.create<arith::MinimumFOp>(loc, arg, max);
  return rewriter.create<arith::MaximumFOp>(loc, minValue, min);
}

Value mlir::tosa::clampIntHelper(Location loc, Value arg, Value min, Value max,
                                 OpBuilder &rewriter, bool isUnsigned) {
  if (isUnsigned) {
    auto minOrArg = rewriter.create<arith::MaxUIOp>(loc, min, arg);
    return rewriter.create<arith::MinUIOp>(loc, max, minOrArg);
  }
  auto minOrArg = rewriter.create<arith::MaxSIOp>(loc, min, arg);
  return rewriter.create<arith::MinSIOp>(loc, max, minOrArg);
}

bool mlir::tosa::validIntegerRange(IntegerType ty, int64_t value) {
  uint64_t bitwidth = ty.getIntOrFloatBitWidth();
  if (ty.getSignedness() == IntegerType::Unsigned) {
    uint64_t uvalue = value;
    APInt intMin = APInt::getMinValue(bitwidth);
    APInt intMax = APInt::getMaxValue(bitwidth);
    return uvalue >= intMin.getZExtValue() && uvalue <= intMax.getZExtValue();
  }

  APInt intMin = APInt::getSignedMinValue(bitwidth);
  APInt intMax = APInt::getSignedMaxValue(bitwidth);
  return value >= intMin.getSExtValue() && value <= intMax.getSExtValue();
}

namespace {
// Given two tensors of high and low ranks, derive the output shape
// to reshape the lower rank to.
// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
LogicalResult
computeReshapeOutput(ArrayRef<int64_t> higherRankShape,
                     ArrayRef<int64_t> lowerRankShape,
                     SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      return failure();
  }
  return success();
}
} // namespace

LogicalResult mlir::tosa::EqualizeRanks(PatternRewriter &rewriter, Location loc,
                                        Value &input1, Value &input2) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return EqualizeRanks(builder, input1, input2);
}

LogicalResult mlir::tosa::EqualizeRanks(ImplicitLocOpBuilder &builder,
                                        Value &input1, Value &input2) {
  auto input1Ty = llvm::dyn_cast<RankedTensorType>(input1.getType());
  auto input2Ty = llvm::dyn_cast<RankedTensorType>(input2.getType());

  if (!input1Ty || !input2Ty) {
    return failure();
  }

  int64_t input1Rank = input1Ty.getRank();
  int64_t input2Rank = input2Ty.getRank();

  if (input1Rank == input2Rank)
    return success();

  Value higherTensorValue, lowerTensorValue;
  if (input1Rank > input2Rank) {
    higherTensorValue = input1;
    lowerTensorValue = input2;
  } else {
    higherTensorValue = input2;
    lowerTensorValue = input1;
  }

  ArrayRef<int64_t> higherRankShape =
      llvm::cast<RankedTensorType>(higherTensorValue.getType()).getShape();
  ArrayRef<int64_t> lowerRankShape =
      llvm::cast<RankedTensorType>(lowerTensorValue.getType()).getShape();

  SmallVector<int64_t, 4> reshapeOutputShape;

  if (computeReshapeOutput(higherRankShape, lowerRankShape, reshapeOutputShape)
          .failed())
    return failure();

  auto reshapeInputType =
      llvm::cast<RankedTensorType>(lowerTensorValue.getType());
  auto reshapeOutputType = RankedTensorType::get(
      ArrayRef<int64_t>(reshapeOutputShape), reshapeInputType.getElementType());
  auto reshapeOutputShapeValue = getTosaConstShape(builder, reshapeOutputShape);

  auto reshapeLower = builder.create<tosa::ReshapeOp>(
      reshapeOutputType, lowerTensorValue, reshapeOutputShapeValue);

  if (input1Rank > input2Rank) {
    input1 = higherTensorValue;
    input2 = reshapeLower.getResult();
  } else {
    input1 = reshapeLower.getResult();
    input2 = higherTensorValue;
  }

  return success();
}

std::optional<int32_t> mlir::tosa::getConstTosaMulShift(tosa::MulOp mulOp) {
  int32_t shift = 0;
  if (mulOp.getShift().getImpl()) {
    ElementsAttr shiftElem;
    if (!matchPattern(mulOp.getShift(), m_Constant(&shiftElem))) {
      return std::nullopt;
    }
    shift = shiftElem.getValues<IntegerAttr>()[0].getInt();
  }
  return shift;
}

Value mlir::tosa::getTosaConstShape(ImplicitLocOpBuilder &builder,
                                    llvm::ArrayRef<int64_t> shape) {
  auto attr = builder.getIndexTensorAttr(convertFromMlirShape(shape));
  auto type = mlir::tosa::shapeType::get(builder.getContext(), shape.size());
  mlir::Operation *mlir_op = builder.create<tosa::ConstShapeOp>(type, attr);
  return mlir_op->getResult(0);
}

Value mlir::tosa::getTosaConstShape(PatternRewriter &rewriter, Location loc,
                                    llvm::ArrayRef<int64_t> shape) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return getTosaConstShape(builder, shape);
}

SmallVector<int64_t> mlir::tosa::convertFromMlirShape(ArrayRef<int64_t> shape) {
  return to_vector(llvm::map_range(shape, [](int64_t dim) {
    return ShapedType::isDynamic(dim) ? -1 : dim;
  }));
}

// AMD: Picked from torch-mlir 12250739bfe85b702f9503cad45c2e535ea8eb18
// Get accumulator type for TOSA convolution ops
LogicalResult mlir::tosa ::getConvOpsAccType(PatternRewriter &rewriter,
                                             RankedTensorType inputTy,
                                             RankedTensorType weightTy,
                                             RankedTensorType outputTy,
                                             TypeAttr &accType) {
  auto inputElemTy = inputTy.getElementType();
  auto weightElemTy = weightTy.getElementType();
  auto outputElemTy = outputTy.getElementType();

  auto quantTy = dyn_cast<quant::QuantizedType>(inputElemTy);
  if (quantTy)
    inputElemTy = quantTy.getStorageType();

  // Get TOSA conv ops acc type based on input, weight, and output types
  // according to the spec:
  // https://www.mlplatform.org/tosa/tosa_spec.html#_conv2d
  // https://www.mlplatform.org/tosa/tosa_spec.html#_depthwise_conv2d
  // https://www.mlplatform.org/tosa/tosa_spec.html#_conv3d
  //
  // For undefined dtypes in TOSA like I64 and F64, acc_type will be set to the
  // output type but does not offer any guarantee on the numerical precision
  // since such cases will fail TOSA validation.
  if ((inputElemTy.isF32() && weightElemTy.isF32() && outputElemTy.isF32()) ||
      (inputElemTy.isF16() && weightElemTy.isF16() && outputElemTy.isF16()) ||
      (inputElemTy.isBF16() && weightElemTy.isBF16() &&
       outputElemTy.isBF16())) {
    accType = mlir::TypeAttr::get(rewriter.getF32Type());
  } else if (inputElemTy.isInteger(8) &&
             (weightElemTy.isInteger(8) || weightElemTy.isInteger(4)) &&
             outputElemTy.isInteger(32)) {
    accType = mlir::TypeAttr::get(rewriter.getIntegerType(32));
  } else if (inputElemTy.isInteger(16) && weightElemTy.isInteger(8) &&
             outputElemTy.isInteger(48)) {
    accType = mlir::TypeAttr::get(rewriter.getIntegerType(48));
  } else if ((isa<Float8E4M3FNType>(inputElemTy) &&
              isa<Float8E4M3FNType>(weightElemTy) && outputElemTy.isF16()) ||
             (isa<Float8E5M2Type>(inputElemTy) &&
              isa<Float8E5M2Type>(weightElemTy) && outputElemTy.isF16())) {
    accType = mlir::TypeAttr::get(rewriter.getF16Type());
  } else {
    accType = mlir::TypeAttr::get(outputElemTy);
  }

  return success();
}

bool mlir::tosa::getConstShapeValue(Operation *op,
                                    llvm::SmallVector<int64_t> &result_shape) {
  if (!op) {
    return false;
  }
  if (auto constOp = mlir::dyn_cast<tosa::ConstShapeOp>(op)) {
    Attribute constOpAttr = constOp->getAttr("value");
    DenseElementsAttr elementsAttr = cast<DenseElementsAttr>(constOpAttr);
    for (int i = 0; i < elementsAttr.size(); i++) {
      int64_t val = elementsAttr.getValues<int64_t>()[i];
      result_shape.push_back(val);
    }
    return true;
  }
  // for undefined op, return false.
  return false;
}
