//===- ArithToEmitC.h - Convert Arith to EmitC ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H
#define MLIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;

#define GEN_PASS_DECL_ARITHTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

void populateArithToEmitCConversionPatterns(RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H
