// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s

emitc.global @myglobal : !emitc.array<2xf32> = dense<4.000000e+00>
// CHECK: float myglobal[2] = {4.000000000e+00f, 4.000000000e+00f};

emitc.global constant @myconstant : !emitc.array<2xf32> = dense<4.000000e+00>
// CHECK: const float myglobal[2] = {4.000000000e+00f, 4.000000000e+00f};

func.func @use_global(%i: index) -> f32 {
  %0 = emitc.get_global @myglobal : !emitc.array<2xf32>
  %1 = emitc.subscript %0[%i] : <2xf32>
  return %1 : f32
  // CHECK: float use_global(size_t v1) {
  // CHECK:   return myglobal[v1];
}
