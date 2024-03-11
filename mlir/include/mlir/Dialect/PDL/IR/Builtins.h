//===- Builtins.h - Builtin functions of the PDL dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines builtin functions of the PDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_BUILTINS_H_
#define MLIR_DIALECT_PDL_IR_BUILTINS_H_

#include "mlir/Support/LogicalResult.h"
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
class PDLPatternModule;
class Attribute;
class PatternRewriter;

namespace pdl {
void registerBuiltins(PDLPatternModule &pdlPattern);

namespace builtin {
Attribute createDictionaryAttr(PatternRewriter &rewriter);
Attribute addEntryToDictionaryAttr(PatternRewriter &rewriter,
                                   Attribute dictAttr, Attribute attrName,
                                   Attribute attrEntry);
Attribute createArrayAttr(PatternRewriter &rewriter);
Attribute addElemToArrayAttr(PatternRewriter &rewriter, Attribute attr,
                             Attribute element);
LogicalResult add(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args);
} // namespace builtin
} // namespace pdl
} // namespace mlir

#endif // MLIR_DIALECT_PDL_IR_BUILTINS_H_
