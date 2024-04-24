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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class PDLPatternModule;
class Attribute;
class PatternRewriter;

namespace pdl {
void registerBuiltins(PDLPatternModule &pdlPattern);

namespace builtin {
enum class BinaryOpKind {
  add,
  sub,
  mul,
  div,
  mod,
};

enum class UnaryOpKind {
  log2,
  exp2,
};

LogicalResult createDictionaryAttr(PatternRewriter &rewriter,
                                   PDLResultList &results,
                                   ArrayRef<PDLValue> args);
LogicalResult addEntryToDictionaryAttr(PatternRewriter &rewriter,
                                       PDLResultList &results,
                                       ArrayRef<PDLValue> args);
Attribute createArrayAttr(PatternRewriter &rewriter);
Attribute addElemToArrayAttr(PatternRewriter &rewriter, Attribute attr,
                             Attribute element);
template <BinaryOpKind T>
LogicalResult binaryOp(PatternRewriter &rewriter, PDLResultList &results,
                       llvm::ArrayRef<PDLValue> args);
LogicalResult mul(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args);
LogicalResult div(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args);
LogicalResult mod(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args);
LogicalResult add(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args);
LogicalResult sub(PatternRewriter &rewriter, PDLResultList &results,
                  llvm::ArrayRef<PDLValue> args);
LogicalResult log2(PatternRewriter &rewriter, PDLResultList &results,
                   llvm::ArrayRef<PDLValue> args);
LogicalResult exp2(PatternRewriter &rewriter, PDLResultList &results,
                   llvm::ArrayRef<PDLValue> args);

template <BinaryOpKind T>
LogicalResult binaryOp(PatternRewriter &rewriter, PDLResultList &results,
                       llvm::ArrayRef<PDLValue> args);
} // namespace builtin
} // namespace pdl
} // namespace mlir

#endif // MLIR_DIALECT_PDL_IR_BUILTINS_H_
