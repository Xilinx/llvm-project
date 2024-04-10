//===-- Passes.td - Math pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_TRANSFORMS_PASSES
#define MLIR_DIALECT_MATH_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MathUpliftToFMA : Pass<"math-uplift-to-fma"> {
  let summary = "Uplift arith ops to math.fma.";
  let description = [{
    Uplift sequence of addf and mulf ops to math.fma if fastmath flags allows it.
  }];
  let dependentDialects = ["math::MathDialect"];
}

def MathExpansion : Pass<"expand-math"> {
  let summary = "Expand math operations";
  let description = [{
    Expand complex `math` dialect operations using simpler arithmetic
    and math operations.
  }];
  let options = [
    Option<"expandCtlz", "expand-ctlz", "bool", /*default=*/"false", "Expand Ctlz op">,
    Option<"expandTan", "expand-tan", "bool", /*default=*/"false", "Expand Tan op">,
    Option<"expandTanh", "expand-tanh", "bool", /*default=*/"false", "Expand Tanh op">,
    Option<"expandFmaF", "expand-fmaf", "bool", /*default=*/"false", "Expand FmaF op">,
    Option<"expandFloorF", "expand-floorf", "bool", /*default=*/"false", "Expand FloorF op">,
    Option<"expandCeilF", "expand-ceilf", "bool", /*default=*/"false", "Expand CeilF op">,
    Option<"expandExp2F", "expand-exp2f", "bool", /*default=*/"false", "Expand Exp2F op">,
    Option<"expandPowF", "expand-powf", "bool", /*default=*/"false", "Expand PowF op">,
    Option<"expandRoundF", "expand-roundf", "bool", /*default=*/"false", "Expand RoundF op">,
    Option<"expandRoundEven", "expand-roundeven", "bool", /*default=*/"false", "Expand RoundEven op">,
    Option<"expandRsqrt", "expand-rsqrt", "bool", /*default=*/"false", "Expand Rsqrt op">,
  ];
  let dependentDialects = ["math::MathDialect", "arith::ArithDialect"];
}

#endif // MLIR_DIALECT_MATH_TRANSFORMS_PASSES
