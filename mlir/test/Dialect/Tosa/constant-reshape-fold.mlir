// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s
// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold="fold-splat-or-single-use-only=0" %s | FileCheck %s --check-prefix CHECK-MULTI

// CHECK-LABEL: @reshape_single_user
func.func @reshape_single_user() -> tensor<1x2xf32> {
  // CHECK: %[[RES:.*]] = "tosa.const"{{.*}}-> tensor<1x2xf32>
  // CHECK: return %[[RES]]
  %0 = "tosa.const"() {value = dense<4.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 2>}: (tensor<2xf32>) -> tensor<1x2xf32>
  return %1 : tensor<1x2xf32>
}

// CHECK-LABEL: @reshape_multi_user_splat
func.func @reshape_multi_user_splat() -> (tensor<1x2xf32>, tensor<2xf32>) {
  // CHECK-DAG: %[[RES:.*]] = "tosa.const"{{.*}}-> tensor<2xf32>
  // CHECK-DAG: %[[RESHAPED:.*]] = "tosa.const"{{.*}}-> tensor<1x2xf32>
  // CHECK: return %[[RESHAPED]], %[[RES]]
  %0 = "tosa.const"() {value = dense<4.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 2>}: (tensor<2xf32>) -> tensor<1x2xf32>
  return %1, %0 : tensor<1x2xf32>, tensor<2xf32>
}

// CHECK-LABEL: @reshape_multi_user_non_splat
func.func @reshape_multi_user_non_splat() -> (tensor<1x2xf32>, tensor<2xf32>) {
  // CHECK: %[[CONST:.*]] = "tosa.const"{{.*}}-> tensor<2xf32>
  // CHECK: %[[RES:.*]] = tosa.reshape
  // CHECK: return %[[RES]], %[[CONST]]
  // CHECK-MULTI-DAG: %[[RES:.*]] = "tosa.const"{{.*}}-> tensor<2xf32>
  // CHECK-MULTI-DAG: %[[RESHAPED:.*]] = "tosa.const"{{.*}}-> tensor<1x2xf32>
  // CHECK-MULTI: return %[[RESHAPED]], %[[RES]]
  %0 = "tosa.const"() {value = dense<[4.0, 3.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 2>}: (tensor<2xf32>) -> tensor<1x2xf32>
  return %1, %0 : tensor<1x2xf32>, tensor<2xf32>
}
