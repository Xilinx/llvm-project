#include <cassert>
#include <cstdint>
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

LogicalResult add(mlir::PatternRewriter &rewriter, mlir::PDLResultList &results,
                  llvm::ArrayRef<mlir::PDLValue> args) {
  assert(args.size() == 2 && "Expected 2 arguments");
  auto lhsAttr = args[0].cast<Attribute>();
  auto rhsAttr = args[1].cast<Attribute>();
  if (auto lhsIntAttr = dyn_cast_or_null<IntegerAttr>(lhsAttr)) {
    auto rhsIntAttr = dyn_cast_or_null<IntegerAttr>(rhsAttr);
    if (!rhsIntAttr || lhsIntAttr.getType() != rhsIntAttr.getType())
      return failure();

    APInt lhsVal = lhsIntAttr.getValue();
    APInt rhsVal = rhsIntAttr.getValue();
    auto resultAPInt = lhsVal + rhsVal;

    IntegerAttr result;
    if (lhsIntAttr.getType().isInteger(8)) {
      result = rewriter.getI8IntegerAttr((int8_t)resultAPInt.getSExtValue());
    } else if (lhsIntAttr.getType().isInteger(16)) {
      result = rewriter.getI16IntegerAttr((int16_t)resultAPInt.getSExtValue());
    } else if (lhsIntAttr.getType().isInteger(32)) {
      result = rewriter.getI32IntegerAttr((int32_t)resultAPInt.getSExtValue());
    } else if (lhsIntAttr.getType().isInteger(64)) {
      result = rewriter.getI64IntegerAttr((int64_t)resultAPInt.getSExtValue());
    } else {
      return failure();
    }
    results.push_back(result);
    return success();
  }

  if (auto lhsFloatAttr = dyn_cast_or_null<FloatAttr>(lhsAttr)) {
    auto rhsFloatAttr = dyn_cast_or_null<FloatAttr>(rhsAttr);
    if (!rhsFloatAttr || lhsFloatAttr.getType() != rhsFloatAttr.getType())
      return failure();

    APFloat lhsVal = lhsFloatAttr.getValue();
    APFloat rhsVal = rhsFloatAttr.getValue();
    auto resultAPFloat = lhsVal + rhsVal;

    FloatAttr result;
    if (lhsFloatAttr.getType().isF16()) {
      result = rewriter.getF16FloatAttr(resultAPFloat.convertToFloat());
    } else if (lhsFloatAttr.getType().isF32()) {
      result = rewriter.getF32FloatAttr(resultAPFloat.convertToFloat());
    } else if (lhsFloatAttr.getType().isF64()) {
      result = rewriter.getF64FloatAttr(resultAPFloat.convertToFloat());
    } else {
      // other float types not supported
      return failure();
    }

    results.push_back(result);
    return success();
  }
  return failure();
}

LogicalResult equals(mlir::PatternRewriter &, mlir::Attribute lhs,
                     mlir::Attribute rhs) {
  if (auto lhsAttr = dyn_cast_or_null<IntegerAttr>(lhs)) {
    auto rhsAttr = dyn_cast_or_null<IntegerAttr>(rhs);
    if (!rhsAttr || lhsAttr.getType() != rhsAttr.getType())
      return failure();

    APInt lhsVal = lhsAttr.getValue();
    APInt rhsVal = rhsAttr.getValue();
    return success(lhsVal.eq(rhsVal));
  }

  if (auto lhsAttr = dyn_cast_or_null<FloatAttr>(lhs)) {
    auto rhsAttr = dyn_cast_or_null<FloatAttr>(rhs);
    if (!rhsAttr || lhsAttr.getType() != rhsAttr.getType())
      return failure();

    APFloat lhsVal = lhsAttr.getValue();
    APFloat rhsVal = rhsAttr.getValue();
    return success(lhsVal.compare(rhsVal) == llvm::APFloatBase::cmpEqual);
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
  pdlPattern.registerConstraintFunction("__builtin_equals", equals);
  pdlPattern.registerConstraintFunctionWithResults("__builtin_add", add);
}
} // namespace mlir::pdl
