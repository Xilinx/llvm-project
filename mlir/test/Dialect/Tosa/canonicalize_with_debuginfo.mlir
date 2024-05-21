// RUN: mlir-opt -mlir-print-debuginfo -canonicalize="test-convergence" %s | FileCheck %s

// CHECK-LABEL: @clamp_twice_is_single_clamp
func.func @clamp_twice_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_fp = 3.000000e+00 : f32, max_int = 2 : i64, min_fp = -3.000000e+00 : f32, min_int = -2 : i64} {{.*}} loc(#[[FUSED:.*]])
  // CHECK-DAG: #[[A:.*]] = loc("Clamp_A")
  // CHECK-DAG: #[[B:.*]] = loc("Clamp_B")
  // CHECK:     #[[FUSED]] = loc(fused[#[[B]], #[[A]]])
  %0 = tosa.clamp %arg0 {max_fp = 3.0 : f32, max_int = 4 : i64, min_fp = -5.0 : f32, min_int = -2 : i64} :  (tensor<4xi8>) -> tensor<4xi8> loc(#loc0)
  %1 = tosa.clamp %0 {max_fp = 5.0 : f32, max_int = 2 : i64, min_fp = -3.0 : f32, min_int = -4 : i64} :  (tensor<4xi8>) -> tensor<4xi8> loc(#loc1)
  return %1 : tensor<4xi8>
}
#loc0 = loc("Clamp_A")
#loc1 = loc("Clamp_B")