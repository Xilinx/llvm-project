// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s

// CHECK-LABEL: @slice_int8
func.func @slice_int8() -> (tensor<1x1xi8>) {
  // CHECK: "tosa.const"() <{value = dense<3>
  %0 = "tosa.const"() {value = dense<[[3, 4], [5, 6]]> : tensor<2x2xi8>} : () -> tensor<2x2xi8>
  %1 = "tosa.slice"(%0){size = array<i64: 1, 1>, start = array<i64: 0, 0>} : (tensor<2x2xi8>) -> tensor<1x1xi8>
  return %1 : tensor<1x1xi8>
}

func.func @slice_int16() -> (tensor<2x1xi16>) {
  // CHECK: "tosa.const"() <{value = dense<{{\[\[}}3], [5]]>
  %0 = "tosa.const"() {value = dense<[[3, 4], [5, 6]]> : tensor<2x2xi16>} : () -> tensor<2x2xi16>
  %1 = "tosa.slice"(%0){size = array<i64: 2, 1>, start = array<i64: 0, 0>} : (tensor<2x2xi16>) -> tensor<2x1xi16>
  return %1 : tensor<2x1xi16>
}

// CHECK-LABEL: @slice_int32
func.func @slice_int32() -> (tensor<2x1xi32>) {
  // CHECK: "tosa.const"() <{value = dense<{{\[\[}}4], [6]]>
  %0 = "tosa.const"() {value = dense<[[3, 4], [5, 6]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.slice"(%0){size = array<i64: 2, 1>, start = array<i64: 0, 1>} : (tensor<2x2xi32>) -> tensor<2x1xi32>
  return %1 : tensor<2x1xi32>
}

// CHECK-LABEL: @slice_int32_default_value
func.func @slice_int32_default_value() -> (tensor<3x1xi32>) {
  // CHECK: "tosa.const"() <{value = dense<{{\[\[}}3], [6], [9]]>
  %0 = "tosa.const"() {value = dense<[[3, 4, 5], [6, 7, 8], [9, 10, 11]]> : tensor<3x3xi32>} : () -> tensor<3x3xi32>
  %1 = "tosa.slice"(%0){size = array<i64: 3, 1>, start = array<i64: 0, 0>} : (tensor<3x3xi32>) -> tensor<3x1xi32>
  return %1 : tensor<3x1xi32>
}

// CHECK-LABEL: @slice_bf16_default_value
func.func @slice_bf16_default_value() -> (tensor<3x2xbf16>) {
  // CHECK: "tosa.const"() <{value = dense<{{\[\[}}4.000000e+00, 5.000000e+00], [7.000000e+00, 8.000000e+00], [1.000000e+01, 1.100000e+01]]>
  %0 = "tosa.const"() {value = dense<[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]> : tensor<3x3xbf16>} : () -> tensor<3x3xbf16>
  %1 = "tosa.slice"(%0){size = array<i64: 3, 2>, start = array<i64: 0, 1>} : (tensor<3x3xbf16>) -> tensor<3x2xbf16>
  return %1 : tensor<3x2xbf16>
}
