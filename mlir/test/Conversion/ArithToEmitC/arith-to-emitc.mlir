// RUN: mlir-opt -split-input-file -convert-arith-to-emitc %s | FileCheck %s

// CHECK-LABEL: arith_constants
func.func @arith_constants() {
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : index
  %c_index = arith.constant 0 : index
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : i32
  %c_signless_int_32 = arith.constant 0 : i32
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0.{{0+}}e+00 : f32
  %c_float_32 = arith.constant 0.0 : f32
  // CHECK: emitc.constant
  // CHECK-SAME: value = dense<0> : tensor<i32>
  %c_tensor_single_value = arith.constant dense<0> : tensor<i32>
  // CHECK: emitc.constant
  // CHECK-SAME: value{{.*}}[1, 2], [-3, 9], [0, 0], [2, -1]{{.*}}tensor<4x2xi64>
  %c_tensor_value = arith.constant dense<[[1, 2], [-3, 9], [0, 0], [2, -1]]> : tensor<4x2xi64>
  return
}
