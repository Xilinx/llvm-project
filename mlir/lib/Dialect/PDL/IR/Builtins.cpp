#include <mlir/Dialect/PDL/IR/Builtins.h>
#include <mlir/IR/PatternMatch.h>

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
}
} // namespace mlir::pdl
