#include <cassert>
#include <cstdint>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/PDL/IR/Builtins.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;

namespace mlir::pdl {
namespace builtin {
mlir::Attribute createDictionaryAttr(mlir::PatternRewriter &rewriter) {
  return rewriter.getDictionaryAttr({});
}

mlir::Attribute addEntryToDictionaryAttr(mlir::PatternRewriter &rewriter,
                                         mlir::Attribute dictAttr,
                                         mlir::Attribute attrName,
                                         mlir::Attribute attrEntry) {
  assert(isa<DictionaryAttr>(dictAttr));
  auto attr = dictAttr.cast<DictionaryAttr>();
  auto name = attrName.cast<StringAttr>();
  std::vector<NamedAttribute> values = attr.getValue().vec();

  // Remove entry if it exists in the dictionary.
  llvm::erase_if(values, [&](NamedAttribute &namedAttr) {
    return namedAttr.getName() == name.getValue();
  });

  values.push_back(rewriter.getNamedAttr(name, attrEntry));
  return rewriter.getDictionaryAttr(values);
}

mlir::Attribute createArrayAttr(mlir::PatternRewriter &rewriter) {
  return rewriter.getArrayAttr({});
}

mlir::Attribute addElemToArrayAttr(mlir::PatternRewriter &rewriter,
                                   mlir::Attribute attr,
                                   mlir::Attribute element) {
  assert(isa<ArrayAttr>(attr));
  auto values = cast<ArrayAttr>(attr).getValue().vec();
  values.push_back(element);
  return rewriter.getArrayAttr(values);
}

LogicalResult addOrSub(mlir::PatternRewriter &rewriter,
                       mlir::PDLResultList &results,
                       llvm::ArrayRef<mlir::PDLValue> args, const bool useAdd) {
  assert(args.size() == 2 && "Expected 2 arguments");
  auto lhsAttr = args[0].cast<Attribute>();
  auto rhsAttr = args[1].cast<Attribute>();

  // Integer
  if (auto lhsIntAttr = dyn_cast_or_null<IntegerAttr>(lhsAttr)) {
    auto rhsIntAttr = dyn_cast_or_null<IntegerAttr>(rhsAttr);
    if (!rhsIntAttr || lhsIntAttr.getType() != rhsIntAttr.getType())
      return failure();

    auto integerType = lhsIntAttr.getType();
    bool isOverflow;
    llvm::APInt resultAPInt;
    if (integerType.isUnsignedInteger() || integerType.isSignlessInteger()) {
      if (useAdd) {
        resultAPInt =
            lhsIntAttr.getValue().uadd_ov(rhsIntAttr.getValue(), isOverflow);
      } else {
        resultAPInt =
            lhsIntAttr.getValue().usub_ov(rhsIntAttr.getValue(), isOverflow);
      }
    } else {
      if (useAdd) {
        resultAPInt =
            lhsIntAttr.getValue().sadd_ov(rhsIntAttr.getValue(), isOverflow);
      } else {
        resultAPInt =
            lhsIntAttr.getValue().ssub_ov(rhsIntAttr.getValue(), isOverflow);
      }
    }

    if (isOverflow) {
      return failure();
    }

    results.push_back(rewriter.getIntegerAttr(integerType, resultAPInt));
    return success();
  }

  // Float
  if (auto lhsFloatAttr = dyn_cast_or_null<FloatAttr>(lhsAttr)) {
    auto rhsFloatAttr = dyn_cast_or_null<FloatAttr>(rhsAttr);
    if (!rhsFloatAttr || lhsFloatAttr.getType() != rhsFloatAttr.getType())
      return failure();

    APFloat lhsVal = lhsFloatAttr.getValue();
    APFloat rhsVal = rhsFloatAttr.getValue();
    APFloat resultVal(lhsVal);
    auto floatType = lhsFloatAttr.getType();

    APFloat::opStatus operationStatus;
    if (useAdd) {
      operationStatus =
          resultVal.add(rhsVal, llvm::APFloatBase::rmNearestTiesToEven);
    } else {
      operationStatus =
          resultVal.subtract(rhsVal, llvm::APFloatBase::rmNearestTiesToEven);
    }

    if (operationStatus != APFloat::opOK) {
      return failure();
    }

    results.push_back(rewriter.getFloatAttr(floatType, resultVal));
    return success();
  }
  return failure();
}

LogicalResult mulOrDiv(mlir::PatternRewriter &rewriter,
                       mlir::PDLResultList &results,
                       llvm::ArrayRef<PDLValue> args, const bool useDiv) {
  assert(args.size() == 2 && "Expected 2 arguments");
  auto lhsAttr = args[0].cast<Attribute>();
  auto rhsAttr = args[1].cast<Attribute>();

  // Integer
  if (auto lhsIntAttr = dyn_cast_or_null<IntegerAttr>(lhsAttr)) {
    auto rhsIntAttr = dyn_cast_or_null<IntegerAttr>(rhsAttr);
    if (!rhsIntAttr || lhsIntAttr.getType() != rhsIntAttr.getType())
      return failure();

    auto integerType = lhsIntAttr.getType();

    bool isOverflow = false;
    llvm::APInt resultAPInt;
    if (integerType.isUnsignedInteger() || integerType.isSignlessInteger()) {
      if (useDiv) {
        resultAPInt = lhsIntAttr.getValue().udiv(rhsIntAttr.getValue());
      } else {
        resultAPInt =
            lhsIntAttr.getValue().umul_ov(rhsIntAttr.getValue(), isOverflow);
      }
    } else {
      if (useDiv) {
        resultAPInt =
            lhsIntAttr.getValue().sdiv_ov(rhsIntAttr.getValue(), isOverflow);
      } else {
        resultAPInt =
            lhsIntAttr.getValue().smul_ov(rhsIntAttr.getValue(), isOverflow);
      }
    }

    if (isOverflow) {
      return failure();
    }

    results.push_back(rewriter.getIntegerAttr(integerType, resultAPInt));
    return success();
  }

  // Float
  if (auto lhsFloatAttr = dyn_cast_or_null<FloatAttr>(lhsAttr)) {
    auto rhsFloatAttr = dyn_cast_or_null<FloatAttr>(rhsAttr);
    if (!rhsFloatAttr || lhsFloatAttr.getType() != rhsFloatAttr.getType())
      return failure();

    APFloat lhsVal = lhsFloatAttr.getValue();
    APFloat rhsVal = rhsFloatAttr.getValue();
    APFloat resultVal(lhsVal);
    auto floatType = lhsFloatAttr.getType();

    APFloat::opStatus operationStatus;
    if (useDiv) {
      operationStatus =
          resultVal.divide(rhsVal, llvm::APFloatBase::rmNearestTiesToEven);
    } else {
      operationStatus =
          resultVal.multiply(rhsVal, llvm::APFloatBase::rmNearestTiesToEven);
    }

    if (operationStatus != APFloat::opOK) {
      return failure();
    }

    results.push_back(rewriter.getFloatAttr(floatType, resultVal));
    return success();
  }
  return failure();
}

LogicalResult add(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  return addOrSub(rewriter, results, args, true);
}

LogicalResult sub(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  return addOrSub(rewriter, results, args, false);
}

LogicalResult mul(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args) {
  return mulOrDiv(rewriter, results, args, false);
}

LogicalResult div(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args) {
  return mulOrDiv(rewriter, results, args, true);
}

LogicalResult mod(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args) {
  assert(args.size() == 2 && "Expected 2 arguments");
  auto lhsAttr = args[0].cast<Attribute>();
  auto rhsAttr = args[1].cast<Attribute>();

  // Integer
  if (auto lhsIntAttr = dyn_cast_or_null<IntegerAttr>(lhsAttr)) {
    auto rhsIntAttr = dyn_cast_or_null<IntegerAttr>(rhsAttr);
    if (!rhsIntAttr || lhsIntAttr.getType() != rhsIntAttr.getType())
      return failure();

    auto integerType = lhsIntAttr.getType();

    llvm::APInt resultAPInt;
    if (integerType.isSignlessInteger() || integerType.isUnsignedInteger()) {
      resultAPInt = lhsIntAttr.getValue().urem(rhsIntAttr.getValue());
    } else {
      resultAPInt = lhsIntAttr.getValue().srem(rhsIntAttr.getValue());
    }

    results.push_back(rewriter.getIntegerAttr(integerType, resultAPInt));
    return success();
  }

  // Float
  if (auto lhsFloatAttr = dyn_cast_or_null<FloatAttr>(lhsAttr)) {
    auto rhsFloatAttr = dyn_cast_or_null<FloatAttr>(rhsAttr);
    if (!rhsFloatAttr || lhsFloatAttr.getType() != rhsFloatAttr.getType())
      return failure();

    APFloat lhsVal = lhsFloatAttr.getValue();
    APFloat rhsVal = rhsFloatAttr.getValue();
    APFloat resultVal(lhsVal);
    auto floatType = lhsFloatAttr.getType();

    APFloat::opStatus operationStatus;
    operationStatus = resultVal.mod(rhsVal);

    if (operationStatus != APFloat::opOK) {
      return failure();
    }

    results.push_back(rewriter.getFloatAttr(floatType, resultVal));
    return success();
  }
  return failure();
}
} // namespace builtin

void registerBuiltins(PDLPatternModule &pdlPattern) {
  using namespace builtin;
  // See Parser::defineBuiltins()
  pdlPattern.registerRewriteFunction("__builtin_createDictionaryAttr",
                                     createDictionaryAttr);
  pdlPattern.registerRewriteFunction("__builtin_addEntryToDictionaryAttr",
                                     addEntryToDictionaryAttr);
  pdlPattern.registerRewriteFunction("__builtin_createArrayAttr",
                                     createArrayAttr);
  pdlPattern.registerRewriteFunction("__builtin_addElemToArrayAttr",
                                     addElemToArrayAttr);
  pdlPattern.registerConstraintFunctionWithResults("__builtin_mul", mul);
  pdlPattern.registerConstraintFunctionWithResults("__builtin_div", div);
  pdlPattern.registerConstraintFunctionWithResults("__builtin_mod", mod);
  pdlPattern.registerConstraintFunctionWithResults("__builtin_add", add);
  pdlPattern.registerConstraintFunctionWithResults("__builtin_sub", sub);
}
} // namespace mlir::pdl
