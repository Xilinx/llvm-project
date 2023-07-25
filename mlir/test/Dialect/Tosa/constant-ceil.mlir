// RUN: mlir-opt --split-input-file --tosa-layerwise-constant-fold %s | FileCheck %s

// CHECK-LABEL: @ceil_fold_single_valued
func.func @ceil_fold_single_valued() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}1.300000e+01{{.*}}tensor<f32>
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<12.2> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.ceil"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @ceil_int
func.func @ceil_int() -> tensor<i32> {
  // CHECK: tosa.const{{.*}}12{{.*}}tensor<i32>
  // CHECK: [[RES:]] ={{.*}}tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<-12> : tensor<i32>} : () -> tensor<i32>
  %1 = "tosa.ceil"(%0) : (tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: @ceil_fold_splat
func.func @ceil_fold_splat() -> tensor<12x7xf32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}5.000000e+00
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<4.2> : tensor<12x7xf32>} : () -> tensor<12x7xf32>
  %1 = "tosa.ceil"(%0) : (tensor<12x7xf32>) -> tensor<12x7xf32>
  return %1 : tensor<12x7xf32>
}

// CHECK-LABEL: @ceil_fold_bf16
func.func @ceil_fold_bf16() -> tensor<12x7xbf16> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}5.000000e+00
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<4.2> : tensor<12x7xbf16>} : () -> tensor<12x7xbf16>
  %1 = "tosa.ceil"(%0) : (tensor<12x7xbf16>) -> tensor<12x7xbf16>
  return %1 : tensor<12x7xbf16>
}

// CHECK-LABEL: @ceil_fold_f16
func.func @ceil_fold_f16() -> tensor<12x7xf16> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}5.000000e+00
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<4.2> : tensor<12x7xf16>} : () -> tensor<12x7xf16>
  %1 = "tosa.ceil"(%0) : (tensor<12x7xf16>) -> tensor<12x7xf16>
  return %1 : tensor<12x7xf16>
}

// CHECK-LABEL: @ceil_nan
func.func @ceil_nan() -> tensor<f32> {
  // 0x7FC00000 is the value for NAN
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}0x7FC00000
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0x7FC00000> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.ceil"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @ceil_infinity
func.func @ceil_infinity() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}<0x7F800000
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0x7F800000> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.ceil"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @ceil_neg_infinity
func.func @ceil_neg_infinity() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}0xFF800000
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<0xFF800000> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.ceil"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @ceil_neg_value
func.func @ceil_neg_value() -> tensor<f32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const{{.*}}-4.000000e+00
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() {value = dense<-4.4> : tensor<f32>} : () -> tensor<f32>
  %1 = "tosa.ceil"(%0) : (tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @ceil_no_fold
// The folding optimization works only intra-procedurally, so we won't be able
// to fold anything here
func.func @ceil_no_fold(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: tosa.ceil
  // CHECK-NEXT: return
  %0 = "tosa.ceil"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @ceil_fold
func.func @ceil_fold() -> tensor<4x6xf32> {
  // CHECK: [[RES:]] ={{.*}}tosa.const
  // CHECK-SAME{LITERAL}: [[1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, -1.000000e+00, 3.000000e+00],
  // CHECK-SAME{LITERAL}:  [-0.000000e+00, 2.000000e+00, 2.000000e+00, -0.000000e+00, -0.000000e+00, -1.000000e+00],
  // CHECK-SAME{LITERAL}:  [-0.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 1.000000e+00],
  // CHECK-SAME{LITERAL}:  [1.000000e+00, -0.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, -1.000000e+00]]
  // CHECK-NOT: tosa.ceil
  // CHECK: return [[RES]]
  %0 = "tosa.const"() { value = dense<[
                        [ 0.1758,  0.0874,  0.5924,  1.4700, -1.1424,  2.9213],
                        [-0.2078,  1.4325,  1.5283, -0.0121, -0.2306, -1.3377],
                        [-0.0804,  0.0761,  0.5277,  1.1292,  0.2446,  0.6946],
                        [ 0.4929, -0.6524,  1.8092,  0.1397,  1.5505, -1.0270]]>
                        : tensor<4x6xf32>
                      } : () -> tensor<4x6xf32>
  %1 = "tosa.ceil"(%0) : (tensor<4x6xf32>) -> tensor<4x6xf32>
  return %1 : tensor<4x6xf32>
}

// CHECK-LABEL: @ceil_of_const_sparse
// Sparse tensors are currently not supported
func.func @ceil_of_const_sparse() -> tensor<32xbf16> {
  // CHECK: tosa.const
  // CHECK: tosa.ceil
    %0 = "tosa.const"() { value = sparse<
          [[0], [3], [11], [17], [20], [23], [25], [30], [31]],
          [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]>
          : tensor<32xbf16> } : () -> tensor<32xbf16>
    %1 = "tosa.ceil"(%0) : (tensor<32xbf16>) -> tensor<32xbf16>
    return %1 : tensor<32xbf16>
}
