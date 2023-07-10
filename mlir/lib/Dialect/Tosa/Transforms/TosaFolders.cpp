//===- TosaFolders.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA operations
//
//===----------------------------------------------------------------------===//

#include <functional>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

/// Type that represents tensor dimensions.
using DimensionType = ArrayRef<int64_t>;

/// Type for tensor offsets.
using OffsetType = size_t;

/// Rounding mode to be used on floating point operations that require rounding.
static constexpr llvm::RoundingMode tosaRoundingMode =
    llvm::APFloat::rmNearestTiesToEven;

OffsetType indexToOffset(DimensionType shape, DimensionType index) {
  OffsetType offset = 0;
  for (size_t i = 0; i < shape.size(); i++) {
    offset = offset * shape[i] + index[i];
  }
  return offset;
}

SmallVector<int64_t> offsetToIndex(DimensionType shape,
                                               OffsetType offset) {
  auto rank = shape.size();
  // The rank of the index will be equal to the rank of the shape
  SmallVector<int64_t> resultIndex;
  resultIndex.reserve(rank);
  // Compute all the index values from the last to the first one, reverse the
  // vector afterwards as there is no convenient push_front.
  for (int32_t i = rank - 1; i >= 0; i--) {
    resultIndex.push_back(offset % shape[i]);
    offset /= shape[i];
  }
  std::reverse(resultIndex.begin(), resultIndex.end());
  return resultIndex;
}

SmallVector<int64_t>
getBroadcastedIndex(DimensionType desiredShape,
                                DimensionType toBeBroadcastedShape,
                                DimensionType index) {
  SmallVector<int64_t> broadCasted;
  broadCasted.reserve(desiredShape.size());
  for (size_t i = 0; i < desiredShape.size(); i++) {
    auto toInsert = 0;
    if (toBeBroadcastedShape[i] == desiredShape[i]) {
      toInsert = index[i];
    }
    broadCasted.push_back(toInsert);
  }
  return broadCasted;
}

OffsetType getBroadcastedOffset(DimensionType desiredShape,
                                            DimensionType toBeBroadcastedShape,
                                            OffsetType offset) {
  // Simply return the offset if the shapes are equal.
  if (desiredShape.equals(toBeBroadcastedShape)) {
    return offset;
  }
  auto indexInTarget = offsetToIndex(desiredShape, offset);
  auto indexBroadcasted =
      getBroadcastedIndex(desiredShape, toBeBroadcastedShape, indexInTarget);
  return indexToOffset(toBeBroadcastedShape, indexBroadcasted);
}


/// Apply the given transformation \p toApply to every element of the tensor to
/// be transformed \p toTransform.
///
/// Elements of \p toTransform are extracted as \p SrcValueType.
///
/// \returns A tensor with the same size as \p toTransform, containing
/// \p TargetValueType values of type \p TargetType.
template <class SrcValType, class TargetValType, class TargetType>
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<TargetValType(const SrcValType &, TargetType)> &toApply,
    TargetType targetType) {
  SmallVector<TargetValType> transformedValues;
  // We already know the amount of values we will insert, reserve space for
  // all of them to avoid dynamic resizing
  transformedValues.reserve(toTransform.getNumElements());
  for (auto val : toTransform.getValues<SrcValType>()) {
    auto transformedVal = toApply(val, targetType);
    transformedValues.push_back(transformedVal);
  }

  // Make sure that the output tensor has the expected output type
  auto inShape = toTransform.getType();
  auto outTy = inShape.cloneWith({}, targetType);

  return DenseElementsAttr::get(outTy, transformedValues);
}

template DenseElementsAttr applyElementWise<APFloat, APFloat, FloatType>(
    const DenseElementsAttr &toTransform,
    const std::function<APFloat(const APFloat &, FloatType)> &toApply,
    FloatType targetType);

/// Function that checks if the type contained in \p toCheck is float.
LogicalResult notifyIfNotFloat(TypedValue<TensorType> toCheck, TosaOp location,
                               PatternRewriter &rewriter) {
  if (isa<FloatType>(toCheck.getType().getElementType())) {
    return success();
  }
  return rewriter.notifyMatchFailure(location,
                                     "Unexpected input tensor type: the "
                                     "TOSA spec only allows floats");
}

template <class ElementType, class ResultType>
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &first, const DenseElementsAttr &second,
    TensorType targetType,
    const std::function<ResultType(const ElementType &, const ElementType &)>
        &toApply) {
  // Make sure to use the correct values in case broadcasting is required
  SmallVector<ResultType> transformedValues;
  // We already know the amount of values we will insert, reserve space for
  // all of them to avoid dynamic resizing
  auto targetSize = 1;
  auto targetShape = targetType.getShape();
  for (const auto &dimSize : targetShape) {
    targetSize *= dimSize;
  }
  transformedValues.reserve(targetSize);

  // Apply the given function to each pair of values from the input tensors.
  // Make sure to broadcast the offsets properly.
  auto firstIt = first.getValues<ElementType>();
  auto firstShape = first.getType().getShape();
  auto secondIt = second.getValues<ElementType>();
  auto secondShape = second.getType().getShape();
  for (auto offset = 0; offset < targetSize; offset++) {
    OffsetType offsetInTargetFirst =
        getBroadcastedOffset(targetShape, firstShape, offset);
    OffsetType offsetInTargetSecond =
        getBroadcastedOffset(targetShape, secondShape, offset);
    auto res =
        toApply(firstIt[offsetInTargetFirst], secondIt[offsetInTargetSecond]);
    transformedValues.push_back(res);
  }

  // Generate a tensor with the computed values.
  auto newTensor = DenseElementsAttr::get(targetType, transformedValues);
  return newTensor;
}

/// Function that checks if \p toCheck is a dense TOSA constant tensor.
LogicalResult notifyIfNoTosaDenseConstantTensor(Value toCheck,
                                                TosaOp location,
                                                PatternRewriter &rewriter) {
  // Check whether the tensor is constant and dense
  // TODO We currently ensure the tensor is dense by using the correct type for
  // the bind_value, however we do not actually need this value. It would be
  // nicer to only have a check here.
  DenseElementsAttr tmp;
  if (!matchPattern(toCheck, m_Constant(&tmp))) {
    return rewriter.notifyMatchFailure(location,
                                       "Non-const or non-dense input tensor");
  }

  // Make sure it actually is a TOSA constant (the match allows for other
  // constants as well)
  if (isa<ConstOp>(toCheck.getDefiningOp())) {
    return success();
  }

  return rewriter.notifyMatchFailure(location,
                                     "The reciprocal can only be folded if "
                                     "it operates on a TOSA constant");
}

/// Function that checks if \p toCheck is a dense TOSA constant float tensor.
LogicalResult notifyIfNotConstantFloatTosaTensor(TypedValue<TensorType> toCheck,
                                                 TosaOp location,
                                                 PatternRewriter &rewriter) {
  auto floatCheck = notifyIfNotFloat(toCheck, location, rewriter);
  if (failed(floatCheck)) {
    return floatCheck;
  }
  return notifyIfNoTosaDenseConstantTensor(toCheck, location, rewriter);
}


template <typename BaseType>
DenseElementsAttr transposeType(ElementsAttr attr, ShapedType inputType,
                                ShapedType outputType,
                                llvm::ArrayRef<int64_t> permValues) {
  if (inputType.getNumElements() == 0)
    return DenseElementsAttr::get(outputType, llvm::ArrayRef<BaseType>{});

  auto attrValues = attr.getValues<BaseType>();
  auto inputShape = inputType.getShape();

  // The inverted permutation map and strides of the output are used to compute
  // the contribution of a given dimension to the destination linear index in
  // an order-independent way.
  auto outputStrides = computeStrides(outputType.getShape());
  auto invertedPermValues = invertPermutationVector(permValues);

  auto initialValue = *std::begin(attrValues);
  SmallVector<BaseType> outputValues(inputType.getNumElements(), initialValue);

  for (const auto &it : llvm::enumerate(attrValues)) {
    auto srcLinearIndex = it.index();

    uint64_t dstLinearIndex = 0;
    for (int64_t dim = inputShape.size() - 1; dim >= 0; --dim) {
      // Compute the index into the current dimension of the source vector.
      auto sourceIndexForDim = srcLinearIndex % inputShape[dim];
      srcLinearIndex /= inputShape[dim];

      // Add the contribution of the current dimension to the output using the
      // permutation map.
      dstLinearIndex +=
          outputStrides[invertedPermValues[dim]] * sourceIndexForDim;
    }

    outputValues[dstLinearIndex] = it.value();
  }

  return DenseElementsAttr::get(outputType,
                                llvm::ArrayRef<BaseType>(outputValues));
}

// A type specialized transposition of an ElementsAttr.
// This implementation tries to operate on the underlying data in its raw
// representation when possible to avoid allocating a large number of Attribute
// objects.
DenseElementsAttr transpose(ElementsAttr attr, ShapedType inputType,
                            ShapedType outputType,
                            llvm::ArrayRef<int64_t> permValues) {
  auto baseType = inputType.getElementType();

  // Handle possible integer types
  if (auto intType = dyn_cast<IntegerType>(baseType)) {
    switch (intType.getWidth()) {
    case 1:
      return transposeType<bool>(attr, inputType, outputType, permValues);
    case 8:
      return transposeType<int8_t>(attr, inputType, outputType, permValues);
    case 16:
      return transposeType<int16_t>(attr, inputType, outputType, permValues);
    case 32:
      return transposeType<int32_t>(attr, inputType, outputType, permValues);
    case 64:
      return transposeType<int64_t>(attr, inputType, outputType, permValues);
    default:
      return transposeType<APInt>(attr, inputType, outputType, permValues);
    }
  }

  // Handle possible float types
  if (baseType.isF32()) {
    return transposeType<float>(attr, inputType, outputType, permValues);
  }

  return transposeType<APFloat>(attr, inputType, outputType, permValues);
}

template<typename TosaOp>
struct TosaFoldConstantBase: public OpRewritePattern<TosaOp> {
  TosaFoldConstantBase(MLIRContext* ctxt, bool foldSplatOrSingleUseOnly) : OpRewritePattern<TosaOp>(ctxt), foldSplatOrSingleUseOnly(foldSplatOrSingleUseOnly) {}

  bool foldSplatOrSingleUseOnly;

  /// Heuristic to decide when to replace a unary operation on a constant with the
  /// folded value.
  /// Folding operations on constants can lead to an increased memory usage
  /// whenever the input cannot be replaced but a new constant is inserted. Hence,
  /// this will currently only suggest folding when the memory impact is
  /// negligible.
  /// Takes the \p unaryOp and the constant input \p values.
  /// \returns Whether folding should be applied.
  bool constantUnaryOpShouldBeFolded(TosaOp unaryOp, DenseElementsAttr values) const {
    if (!foldSplatOrSingleUseOnly)
      return true;
    assert(unaryOp->getNumOperands() == 1);
    auto inputOp = unaryOp->getOperand(0);

    // If the input is a splat, we don't care for the number of users
    if (isa<SplatElementsAttr>(values)) {
      return true;
    }

    // If this is the only use of the tensor it should be replaced as no
    // additional memory is required
    return inputOp.hasOneUse();
  }

  bool constantBinaryOpShouldBeFolded(
      TosaOp binaryOp, DenseElementsAttr valuesFirst,
      DenseElementsAttr valuesSecond) const {
    if (!foldSplatOrSingleUseOnly)
      return true;
    assert(binaryOp->getNumOperands() == 2);
    auto firstOp = binaryOp->getOperand(0);
    auto secondOp = binaryOp->getOperand(1);

    // If both tensors are splat, we don't care for the number of users
    if (isa<SplatElementsAttr>(valuesFirst) &&
        isa<SplatElementsAttr>(valuesSecond)) {
      return true;
    }

    // If this is the only use of one of the tensors, it will be replaced an no
    // additional memory is required.
    if (firstOp.hasOneUse() || secondOp.hasOneUse()) {
      return true;
    }

    // Fold it both inputs are equal and those are the only uses. Don't fold
    // otherwise.
    auto numUsers =
        std::distance(firstOp.getUses().begin(), firstOp.getUses().end());
    return firstOp == secondOp && numUsers == 2;
  }
};

template <typename BaseClass, typename TosaOp>
struct TosaFoldConstantUnaryElementwise : public TosaFoldConstantBase<TosaOp> {
  using TosaFoldConstantBase<TosaOp>::TosaFoldConstantBase;

  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = op.getOperand();
    auto resultType = op.getType();
    // Check that we can apply folding
    auto preCondCheck =
        notifyIfNoTosaDenseConstantTensor(inputTensor, op, rewriter);
    if (failed(preCondCheck)) {
      return preCondCheck;
    }

    // Extract the tensor values
    DenseElementsAttr inputValues;
    matchPattern(inputTensor, m_Constant(&inputValues));

    // Check whether this should be folded.
    if (!TosaFoldConstantBase<TosaOp>::constantUnaryOpShouldBeFolded(
            op, inputValues)) {
      return rewriter.notifyMatchFailure(
          op, "Currently, unary ops will only be folded if the input "
              "tensor has a single user");
    }

    DenseElementsAttr newTensor = static_cast<const BaseClass *>(this)->compute(
        inputValues, resultType, op);
    if (!newTensor) {
      return rewriter.notifyMatchFailure(op,
                                         "Type or values cannot be folded.");
    }
    rewriter.replaceOpWithNewOp<ConstOp>(op, newTensor.getType(), newTensor);
    return success();
  }

  DenseElementsAttr compute(DenseElementsAttr values, TensorType resultType,
                            TosaOp op) const {
    if (isa<IntegerType>(values.getElementType()))
      return static_cast<const BaseClass *>(this)->computeInteger(
          values, resultType, op);

    assert(isa<FloatType>(values.getElementType()));
    return static_cast<const BaseClass *>(this)->computeFloat(values,
                                                              resultType, op);
  }

  /// Called when the values.getElementType() is IntegerType.
  DenseElementsAttr computeInteger(DenseElementsAttr values,
                                   TensorType resultType, TosaOp op) const {
    return {};
  }

  /// Called when the values.getElementType() is FloatType.
  DenseElementsAttr computeFloat(DenseElementsAttr values,
                                 TensorType resultType, TosaOp op) const {
    return {};
  }
};

template<typename BaseClass, typename TosaOp>
struct TosaFoldConstantBinary : public TosaFoldConstantBase<TosaOp> {
  using TosaFoldConstantBase<TosaOp>::TosaFoldConstantBase;

  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    auto leftOp = op.getOperand(0);
    auto rightOp = op.getOperand(1);

    auto lhsTensorType = dyn_cast<TensorType>(leftOp.getType());
    auto rhsTensorType = dyn_cast<TensorType>(rightOp.getType());
    if (!lhsTensorType || !rhsTensorType) {
      return rewriter.notifyMatchFailure(
          op, "Expected types to be tensors.");
    }

    auto resultType = op.getType();
    auto lhsElemType = lhsTensorType.getElementType();
    auto rhsElemType = rhsTensorType.getElementType();
    if (lhsElemType != rhsElemType) {
      return rewriter.notifyMatchFailure(
          op, "Expected type of binary op arguments to match.");
    }

    // Check if both tensors are constant
    auto rhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(leftOp, op, rewriter);
    if (failed(rhsIsConstantCheck)) {
      return rhsIsConstantCheck;
    }
    auto lhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(rightOp, op, rewriter);
    if (failed(lhsIsConstantCheck)) {
      return lhsIsConstantCheck;
    }

    // Extract the tensor values
    DenseElementsAttr lhsValues;
    matchPattern(leftOp, m_Constant(&lhsValues));

    DenseElementsAttr rhsValues;
    matchPattern(rightOp, m_Constant(&rhsValues));

    if (!TosaFoldConstantBase<TosaOp>::constantBinaryOpShouldBeFolded(op, lhsValues, rhsValues)) {
      return rewriter.notifyMatchFailure(
          op, "Currently, binary ops will only be folded if this requires only "
                 "little additional memory usage.");
    }

    DenseElementsAttr newTensor = static_cast<const BaseClass*>(this)->compute(lhsValues, rhsValues, resultType, op);
    if (!newTensor) {
        return rewriter.notifyMatchFailure(
          op, "Type or values cannot be folded.");
    }
    rewriter.replaceOpWithNewOp<ConstOp>(op, newTensor.getType(), newTensor);
    return success();
  }

  DenseElementsAttr compute(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            TosaOp op) const {
    if (isa<IntegerType>(lhsValues.getElementType()))
      return static_cast<const BaseClass*>(this)->computeInteger(lhsValues, rhsValues, resultType, op);

    assert(isa<FloatType>(lhsValues.getElementType()));
    return static_cast<const BaseClass*>(this)->computeFloat(lhsValues, rhsValues, resultType, op);
  }

  /// Called when the lhsValues.getElementType() is IntegerType.
  DenseElementsAttr computeInteger(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            TosaOp op) const {
    return {};
  }

  /// Called when the lhsValues.getElementType() is FloatType.
  DenseElementsAttr computeFloat(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            TosaOp op) const {
    return {};
  }
};

struct TosaFoldConstantTranspose : public TosaFoldConstantBase<tosa::TransposeOp> {
  using TosaFoldConstantBase::TosaFoldConstantBase;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = cast<ShapedType>(op.getType());
    // TOSA supports quantized types.
    if (!outputType.getElementType().isIntOrIndexOrFloat())
      return failure();

    ElementsAttr inputValues;
    if (!matchPattern(op.getInput1(), m_Constant(&inputValues)))
      return failure();
    // Make sure the input is a constant that has a single user.
    if (!llvm::hasSingleElement(op.getInput1().getDefiningOp()->getUsers()))
      return failure();

    DenseIntElementsAttr permAttr;
    if (!matchPattern(op.getPerms(), m_Constant(&permAttr)))
      return failure();
    auto permValues = llvm::to_vector<6>(llvm::map_range(
        // TOSA allows both 32- and 64-bit integer tensors here.
        permAttr.getValues<APInt>(),
        [](const APInt &val) { return val.getSExtValue(); }));

    auto inputType = cast<ShapedType>(op.getInput1().getType());

    auto resultAttr = transpose(inputValues, inputType, outputType, permValues);
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputType, resultAttr);
    return success();
  }
};

static APFloat computeReciprocal(const APFloat &floatVal, FloatType floatTy) {
  auto recipAttr = FloatAttr::get(floatTy, 1.0);
  APFloat recip = recipAttr.getValue();
  recip.divide(floatVal, tosaRoundingMode);
  return recip;
}

struct TosaFoldConstantReciprocal
    : public TosaFoldConstantUnaryElementwise<TosaFoldConstantReciprocal, ReciprocalOp> {
  using TosaFoldConstantUnaryElementwise<TosaFoldConstantReciprocal,
                              ReciprocalOp>::TosaFoldConstantUnaryElementwise;

  DenseElementsAttr computeFloat(DenseElementsAttr values,
                                 TensorType resultType, TosaOp op) const {
    return applyElementWise<APFloat, APFloat, FloatType>(
        values, &computeReciprocal, cast<FloatType>(values.getElementType()));
  }
};

struct TosaFoldConstantRSQRT
    : public TosaFoldConstantUnaryElementwise<TosaFoldConstantRSQRT, RsqrtOp> {
  using TosaFoldConstantUnaryElementwise<TosaFoldConstantRSQRT,
                              RsqrtOp>::TosaFoldConstantUnaryElementwise;

  static APFloat computeRSQRT(const APFloat &apFloatVal, FloatType floatTy) {
    // The result for negative values (apart from zero) is always NaN
    if (apFloatVal.isNegative() && !apFloatVal.isNegZero()) {
      return APFloat::getNaN(apFloatVal.getSemantics());
    }

    // Compute the square root (APFloat unfortunately does not provide this
    // function, such that we need to unpack here)
    auto floatVal = apFloatVal.convertToFloat();
    auto sqrtVal = std::sqrt(floatVal);
    APFloat apSqrtVal(sqrtVal);

    // Compute the reciprocal
    return computeReciprocal(apSqrtVal, floatTy);
  }

  DenseElementsAttr computeFloat(DenseElementsAttr values,
                                 TensorType resultType, TosaOp op) const {
    return applyElementWise<APFloat, APFloat, FloatType>(
        values, &computeRSQRT, cast<FloatType>(values.getElementType()));
  }
};

struct TosaFoldConstantPow : public TosaFoldConstantBase<PowOp> {

  using TosaFoldConstantBase::TosaFoldConstantBase;

  static APFloat computePower(const APFloat &base, const APFloat &exp) {
    // Propagate NaN
    if (base.isNaN() || exp.isNaN()) {
      return APFloat::getNaN(base.getSemantics());
    }
    // TOSA defines 0.0**0.0 as NaN
    if (base.isZero() && exp.isZero()) {
      return APFloat::getNaN(base.getSemantics());
    }
    // In case the value is negative, the exponent needs to be an integer
    if (base.isNegative() && !base.isZero()) {
      if (!exp.isInteger()) {
        return APFloat::getNaN(base.getSemantics());
      }
    }

    // Actually compute base**exp. Special cases for [-]infinity and [-]0 are
    // already handled in accordance with the TOSA spec.
    auto powFloat = std::pow(base.convertToFloat(), exp.convertToFloat());
    auto res = APFloat(powFloat);

    bool lostPrecision;
    res.convert(base.getSemantics(), APFloat::rmNearestTiesToEven,
                &lostPrecision);
    return res;
  }

  LogicalResult matchAndRewrite(PowOp powOp,
                                PatternRewriter &rewriter) const override {
    auto baseOp = powOp.getInput1();
    auto expOp = powOp.getInput2();

    if (baseOp.getType().getElementType() != expOp.getType().getElementType()) {
      return rewriter.notifyMatchFailure(
          powOp, "Expected type of pow arguments to match.");
    }

    // Check if both tensors are constant
    auto baseIsConstCheck =
        notifyIfNotConstantFloatTosaTensor(baseOp, powOp, rewriter);
    if (failed(baseIsConstCheck)) {
      return baseIsConstCheck;
    }
    auto expIsConstCheck =
        notifyIfNotConstantFloatTosaTensor(expOp, powOp, rewriter);
    if (failed(expIsConstCheck)) {
      return expIsConstCheck;
    }

    // Extract the tensor values
    DenseElementsAttr baseValues;
    matchPattern(baseOp, m_Constant(&baseValues));

    DenseElementsAttr expValues;
    matchPattern(expOp, m_Constant(&expValues));

    if (!constantBinaryOpShouldBeFolded(powOp, baseValues, expValues)) {
      return rewriter.notifyMatchFailure(
          powOp, "Currently, pows will only be folded if this requires only "
                 "little additional memory usage.");
    }

    auto newTensor = applyElementWise<APFloat, APFloat>(
        baseValues, expValues, powOp.getType(), &computePower);
    rewriter.replaceOpWithNewOp<ConstOp>(powOp, newTensor.getType(), newTensor);

    return success();
  }
};


struct TosaFoldConstantMul : public TosaFoldConstantBase<MulOp> {

  using TosaFoldConstantBase::TosaFoldConstantBase;

  LogicalResult matchAndRewrite(MulOp mulOp,
                                PatternRewriter &rewriter) const override {
    if (mulOp.getShift() > 0) {
      return rewriter.notifyMatchFailure(
          mulOp, "Non-zero shift folding is currently not implemented.");
    }

    auto leftOp = mulOp.getInput1();
    auto rightOp = mulOp.getInput2();

    // Check if both tensors are constant
    auto rhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(leftOp, mulOp, rewriter);
    if (failed(rhsIsConstantCheck)) {
      return rhsIsConstantCheck;
    }
    auto lhsIsConstantCheck =
        notifyIfNoTosaDenseConstantTensor(rightOp, mulOp, rewriter);
    if (failed(lhsIsConstantCheck)) {
      return lhsIsConstantCheck;
    }

    // Extract the tensor values
    DenseElementsAttr lhsValues;
    matchPattern(leftOp, m_Constant(&lhsValues));

    DenseElementsAttr rhsValues;
    matchPattern(rightOp, m_Constant(&rhsValues));

    if (!constantBinaryOpShouldBeFolded(mulOp, lhsValues, rhsValues)) {
      return rewriter.notifyMatchFailure(
          mulOp, "Currently, muls will only be folded if this requires only "
                 "little additional memory usage.");
    }

    DenseElementsAttr newTensor;

    auto lhsElemType = leftOp.getType().getElementType();
    auto rhsElemType = rightOp.getType().getElementType();
    assert(lhsElemType == rhsElemType);

    auto resultType = mulOp.getType();
    auto resultElementType = resultType.getElementType();
    if (isa<IntegerType>(lhsElemType)) {
      assert(isa<IntegerType>(rhsElemType) &&
             isa<IntegerType>(resultElementType));
      auto resultElementWidth = resultElementType.getIntOrFloatBitWidth();
      assert(resultElementWidth >= lhsElemType.getIntOrFloatBitWidth() &&
             "The multiplication is expected to have an at least as big output "
             "as input type");

      // Compute the multiplication and track if an overflow occurred to enable
      // emitting a warning
      bool mulOverflowed = false;
      auto intMulFun = [&resultElementWidth, &mulOverflowed](
                           const APInt &first, const APInt &second) {
        bool didOverflow;
        auto res = first.sext(resultElementWidth)
                       .smul_ov(second.sext(resultElementWidth), didOverflow);
        mulOverflowed |= didOverflow;
        return res;
      };
      newTensor = applyElementWise<APInt, APInt>(lhsValues, rhsValues,
                                                 resultType, intMulFun);
      if (mulOverflowed) {
        mulOp.emitWarning(
            "Multiplication did overflow. The results are unspecified.");
      }
    } else {
      assert(isa<FloatType>(lhsElemType) && isa<FloatType>(rhsElemType) &&
             isa<FloatType>(resultType.getElementType()));
      auto mulFun = [](const APFloat &first, const APFloat &second) {
        return first * second;
      };
      newTensor = applyElementWise<APFloat, APFloat>(lhsValues, rhsValues,
                                                     resultType, mulFun);
    }
    rewriter.replaceOpWithNewOp<ConstOp>(mulOp, newTensor.getType(), newTensor);

    return success();
  }
};


struct TosaFoldConstantClamp : public TosaFoldConstantBase<ClampOp> {

  using TosaFoldConstantBase::TosaFoldConstantBase;

  static void
  changeSemanticsLossless(APFloat &floatVal,
                          const llvm::fltSemantics *floatSemantics) {
    bool losesInfo;
    floatVal.convert(*floatSemantics, tosaRoundingMode, &losesInfo);
    assert(!losesInfo);
  }

  DenseElementsAttr applyClamp(DenseElementsAttr inputValues,
                               const APInt &lowerBound, const APInt &upperBound,
                               TensorType resultType) const {

    // Determine the width for the APInt comparison
    auto comparisonWidth =
        std::max(inputValues.getElementType().getIntOrFloatBitWidth(),
                 lowerBound.getBitWidth());
    // Sign-extend the upper and lower bound
    auto extUpperBound = upperBound.sext(comparisonWidth);
    auto extLowerBound = lowerBound.sext(comparisonWidth);

    // Determine the result type
    auto resultingIntType = cast<IntegerType>(resultType.getElementType());

    // Lambda to perform the clamp
    auto clampFun = [&extLowerBound, &extUpperBound,
                     &comparisonWidth](const APInt &val, IntegerType type) {
      auto clampedUpper =
          llvm::APIntOps::smin(val.sext(comparisonWidth), extUpperBound);
      auto fullyClamped = llvm::APIntOps::smax(clampedUpper, extLowerBound);
      assert(type.getWidth() >= fullyClamped.getSignificantBits());
      return fullyClamped.trunc(type.getWidth());
    };
    auto newTensor = applyElementWise<APInt, APInt, IntegerType>(
        inputValues, clampFun, resultingIntType);

    return newTensor;
  }

  DenseElementsAttr applyClamp(DenseElementsAttr inputValues,
                               APFloat lowerBound, APFloat upperBound,
                               TensorType resultType) const {
    auto inputValType = cast<FloatType>(inputValues.getElementType());
    auto inputWidth = inputValType.getWidth();
    auto bWidth = APFloat::semanticsSizeInBits(lowerBound.getSemantics());
    auto *comparisonSem = inputWidth < bWidth
                              ? &lowerBound.getSemantics()
                              : &inputValType.getFloatSemantics();

    changeSemanticsLossless(lowerBound, comparisonSem);
    changeSemanticsLossless(upperBound, comparisonSem);

    auto resultingFloatType = cast<FloatType>(resultType.getElementType());

    // Ensure that the value is larger than the lower bound and smaller than the
    // upper bound
    auto clampFun = [&lowerBound, &upperBound, &comparisonSem](APFloat val,
                                                               FloatType type) {
      if (val.isNaN()) {
        return APFloat::getNaN(type.getFloatSemantics());
      }
      changeSemanticsLossless(val, comparisonSem);
      auto clampedUpper = val < upperBound ? val : upperBound;
      auto fullyClamped = clampedUpper < lowerBound ? lowerBound : clampedUpper;
      changeSemanticsLossless(fullyClamped, &type.getFloatSemantics());
      return fullyClamped;
    };
    auto newTensor = applyElementWise<APFloat, APFloat, FloatType>(
        inputValues, clampFun, resultingFloatType);

    return newTensor;
  }

  LogicalResult matchAndRewrite(ClampOp clampOp,
                                PatternRewriter &rewriter) const override {
    auto valsToClamp = clampOp.getInput();
    auto inputElementType = valsToClamp.getType().getElementType();

    // Check if the input is constant
    if (failed(notifyIfNoTosaDenseConstantTensor(valsToClamp, clampOp,
                                                 rewriter))) {
      return failure();
    }

    if (isa<IntegerType>(inputElementType) &&
        cast<IntegerType>(inputElementType).isUnsigned()) {
      return rewriter.notifyMatchFailure(
          clampOp, "Currently, unsigned integer clamps are unsupported.");
    }

    // Extract the tensor values
    DenseElementsAttr inputValues;
    matchPattern(valsToClamp, m_Constant(&inputValues));

    if (!constantUnaryOpShouldBeFolded(clampOp, inputValues)) {
      return rewriter.notifyMatchFailure(
          clampOp,
          "Currently, clamps will only be folded if this requires only "
          "little additional memory usage.");
    }

    // Apply the clamp to all values of the int/float tensor
    auto resultType = clampOp.getType();
    DenseElementsAttr newTensor;
    if (isa<IntegerType>(inputElementType)) {
      auto lowerBoundVal = clampOp.getMinIntAttr().getValue();
      auto upperBoundVal = clampOp.getMaxIntAttr().getValue();
      assert(lowerBoundVal.getBitWidth() == upperBoundVal.getBitWidth());

      newTensor =
          applyClamp(inputValues, lowerBoundVal, upperBoundVal, resultType);
    } else {
      assert(isa<FloatType>(inputElementType));
      auto lowerBoundVal = clampOp.getMinFp();
      auto upperBoundVal = clampOp.getMaxFp();
      assert(APFloat::getSizeInBits(lowerBoundVal.getSemantics()) ==
             APFloat::getSizeInBits(upperBoundVal.getSemantics()));

      newTensor =
          applyClamp(inputValues, lowerBoundVal, upperBoundVal, resultType);
    }

    rewriter.replaceOpWithNewOp<ConstOp>(clampOp, newTensor.getType(),
                                         newTensor);

    return success();
  }
};


struct TosaFoldConstantCast : public TosaFoldConstantBase<CastOp> {

  using TosaFoldConstantBase::TosaFoldConstantBase;

  static APFloat convertIntToFloat(const APInt &toConvert,
                                   FloatType targetType) {
    APFloat res(targetType.getFloatSemantics());
    res.convertFromAPInt(toConvert, true /* isSigned */, tosaRoundingMode);
    return res;
  }

  static APFloat convertFloatToFloat(const APFloat &toConvert,
                                     FloatType targetType) {
    APFloat res(toConvert);
    bool didLosePrecision;
    res.convert(targetType.getFloatSemantics(), tosaRoundingMode,
                &didLosePrecision);
    return res;
  }

  static APInt convertFloatToInt(const APFloat &toConvert,
                                 IntegerType targetType) {
    auto targetWidth = targetType.getIntOrFloatBitWidth();
    // Converting NaN to an integer results in an unpredictable value. Pick 0.
    if (toConvert.isNaN()) {
      return APInt::getZero(targetWidth);
    }

    // Make sure to properly translate booleans
    if (targetWidth == 1) {
      return toConvert.isZero() ? APInt::getZero(1) : APInt::getAllOnes(1);
    }

    // Use the built-in functionality of APFloats to convert to integers.
    // The result of this conversion should be an integer which might still be
    // outside of the target integer range.
    auto floatSize = APFloat::getSizeInBits(toConvert.getSemantics());
    APSInt converted(std::max(floatSize, targetWidth), targetType.isUnsigned());
    bool ignored = false;
    toConvert.convertToInteger(converted, APFloat::rmNearestTiesToEven,
                               &ignored);
    // Clip to allowed range.
    if (targetWidth < floatSize) {
      if (targetType.isUnsigned()) {
        return converted.truncUSat(targetWidth);
      }
      return converted.truncSSat(targetWidth);
    }
    return converted;
  }

  static APInt convertIntToInt(const APInt &toConvert, IntegerType targetType) {
    // Make sure to properly translate booleans
    if (targetType.getWidth() == 1) {
      return toConvert.isZero() ? APInt::getZero(1) : APInt::getAllOnes(1);
    }
    if (targetType.isUnsigned()) {
      return toConvert.zextOrTrunc(targetType.getIntOrFloatBitWidth());
    }
    return toConvert.sextOrTrunc(targetType.getIntOrFloatBitWidth());
  }

  static void warnAboutNaNToIntCast(DenseElementsAttr elements, CastOp location,
                                    PatternRewriter &rewriter) {
    // This is only relevant if the input values are float
    if (!isa<FloatType>(elements.getElementType())) {
      return;
    }
    // Check if it is an float to integer conversion
    auto resultType = location.getOutput().getType();
    if (!isa<IntegerType>(cast<TensorType>(resultType).getElementType())) {
      return;
    }

    // Report encountered NaNs
    auto checkNan = [](const APFloat &val) { return val.isNaN(); };
    if (any_of(elements.getValues<APFloat>(), checkNan)) {
      location->emitWarning(
          "Float tensor is casted to integer and it contains NaN values. The "
          "cast results in an unspecified value.");
    }
  }

  LogicalResult matchAndRewrite(CastOp tosaCast,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = tosaCast.getInput();

    // If the input tensor is not constant, we cannot fold it.
    if (failed(notifyIfNoTosaDenseConstantTensor(inputTensor, tosaCast,
                                                 rewriter))) {
      return failure();
    }

    auto fromType = inputTensor.getType().getElementType();
    auto toType = tosaCast.getOutput().getType().getElementType();

    DenseElementsAttr elements;
    matchPattern(inputTensor, m_Constant(&elements));

    // Issue a warning if we convert float -> int and NaNs are present; the
    // result value is unspecified in that case
    warnAboutNaNToIntCast(elements, tosaCast, rewriter);

    // Only fold splat tensors and those used only once to avoid duplicating
    // them.
    if (!inputTensor.hasOneUse() && !isa<SplatElementsAttr>(elements)) {
      return rewriter.notifyMatchFailure(tosaCast,
                                         "Currently, casts will only be folded "
                                         "if its input only has a single user");
    }

    // Report a match failure for unexpected types
    if (!toType.isIntOrFloat() || !fromType.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          tosaCast, "Only casts from/to int/float are supported.");
    }

    auto isUnsigned = [](Type toCheck) {
      return isa<IntegerType>(toCheck) &&
             cast<IntegerType>(toCheck).isUnsigned();
    };
    auto typesToCheck = {toType, fromType};
    if (llvm::any_of(typesToCheck, isUnsigned)) {
      // TOSA casts currently don't support unsigned integers.
      // To support them by here, one could use APSInt instead of APInts,
      // however, this causes trouble with `getValues` which does not support
      // APSInts currently.
      return rewriter.notifyMatchFailure(
          tosaCast, "Cast folding from/to unsigned integers is not supported.");
    }

    DenseElementsAttr res;
    if (auto intOutTy = dyn_cast<IntegerType>(toType)) {
      if (isa<FloatType>(fromType)) {
        res = applyElementWise<APFloat, APInt, IntegerType>(
            elements, &convertFloatToInt, intOutTy);
      } else {
        assert(isa<IntegerType>(fromType));
        res = applyElementWise<APInt, APInt, IntegerType>(
            elements, &convertIntToInt, intOutTy);
      }
    } else {
      assert(isa<FloatType>(toType));
      auto floatOutTy = cast<FloatType>(toType);
      if (isa<FloatType>(fromType)) {
        res = applyElementWise<APFloat, APFloat, FloatType>(
            elements, &convertFloatToFloat, floatOutTy);
      } else {
        assert(isa<IntegerType>(fromType));
        res = applyElementWise<APInt, APFloat, FloatType>(
            elements, &convertIntToFloat, floatOutTy);
      }
    }

    rewriter.replaceOpWithNewOp<ConstOp>(tosaCast, res.getType(), res);
    return success();
  }
};

struct TosaFoldConstantFloatCasts : TosaFoldConstantCast {

  TosaFoldConstantFloatCasts(MLIRContext *ctx, bool foldSplatOrSingleUseOnly) : TosaFoldConstantCast(ctx, foldSplatOrSingleUseOnly) {}

  LogicalResult matchAndRewrite(CastOp tosaCast,
                                PatternRewriter &rewriter) const override {
    if (isa<IntegerType>(tosaCast.getInput().getType().getElementType())) {
      return rewriter.notifyMatchFailure(
          tosaCast, "Folding casts from int is currently disabled.");
    }

    return TosaFoldConstantCast::matchAndRewrite(tosaCast, rewriter);
  }
};

struct TosaFoldConstantAdd : public TosaFoldConstantBinary<TosaFoldConstantAdd, AddOp> {
  using TosaFoldConstantBinary<TosaFoldConstantAdd, AddOp>::TosaFoldConstantBinary;

  DenseElementsAttr computeInteger(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            AddOp op) const {
      bool addOverflowed = false;
      auto intAdd = [&addOverflowed](const APInt &first, const APInt &second) {
        bool didOverflow;
        auto res = first.sadd_ov(second, didOverflow);
        addOverflowed |= didOverflow;
        return res;
      };
      auto newTensor = applyElementWise<APInt, APInt>(lhsValues, rhsValues,
                                                 resultType, intAdd);
      if (addOverflowed) {
        op->emitWarning(
            "Addition did overflow. The results are unspecified.");
      }
      return newTensor;
  }

  DenseElementsAttr computeFloat(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            AddOp op) const {
    auto floatAdd = [](const APFloat &first, const APFloat &second) {
      return first + second;
    };
    return applyElementWise<APFloat, APFloat>(lhsValues, rhsValues,
                                                    resultType, floatAdd);
  }
};

struct TosaFoldConstantGreater : public TosaFoldConstantBinary<TosaFoldConstantGreater, GreaterOp> {
  using TosaFoldConstantBinary<TosaFoldConstantGreater, GreaterOp>::TosaFoldConstantBinary;

  DenseElementsAttr computeInteger(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            GreaterOp op) const {
      return applyElementWise<APInt, APInt>(lhsValues, rhsValues,
                                                 resultType, [](const APInt &first, const APInt &second) {
        return APInt(1, first.sgt(second));
      });
  }

  DenseElementsAttr computeFloat(DenseElementsAttr lhsValues,
                            DenseElementsAttr rhsValues,
                            TensorType resultType,
                            GreaterOp op) const {
    return applyElementWise<APFloat, APInt>(lhsValues, rhsValues,
                                                    resultType,[](const APFloat &first, const APFloat &second) {
        return APInt(1, first > second);
      });
  }
};

} // namespace

void mlir::tosa::populateTosaFoldConstantPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns,
    bool foldSplatOrSingleUseOnly,
    bool enableIntCastFolding) {
  patterns.add<TosaFoldConstantTranspose>(ctx, foldSplatOrSingleUseOnly);
  patterns.add<TosaFoldConstantReciprocal>(ctx, foldSplatOrSingleUseOnly);
  patterns.add<TosaFoldConstantRSQRT>(ctx, foldSplatOrSingleUseOnly);
  patterns.add<TosaFoldConstantPow>(ctx, foldSplatOrSingleUseOnly);
  patterns.add<TosaFoldConstantMul>(ctx, foldSplatOrSingleUseOnly);
  patterns.add<TosaFoldConstantClamp>(ctx, foldSplatOrSingleUseOnly);
  if (enableIntCastFolding) {
    patterns.add<TosaFoldConstantCast>(ctx, foldSplatOrSingleUseOnly);
  } else {
    patterns.add<TosaFoldConstantFloatCasts>(ctx, foldSplatOrSingleUseOnly);
  }
  patterns.add<TosaFoldConstantAdd>(ctx, foldSplatOrSingleUseOnly);
  patterns.add<TosaFoldConstantGreater>(ctx, foldSplatOrSingleUseOnly);
}
