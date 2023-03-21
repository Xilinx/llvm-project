// RUN: mlir-opt --split-input-file -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" %s | FileCheck %s

func.func @test_cast(%arg0: tensor<1xf32>) -> tensor<1xui3> {
  // CHECK: linalg.generic
  // CHECK: arith.constant -4.000000e+00
  // CHECK: arith.constant 3.000000e+00
  // CHECK: math.roundeven
  // CHECK: arith.minf
  // CHECK: arith.maxf
  // CHECK: arith.fptoui
  // CHECK: builtin.unrealized_conversion_cast
  %1 = "tosa.cast"(%arg0) : (tensor<1xf32>) -> tensor<1xui3>

  return %1 : tensor<1xui3>
}
  