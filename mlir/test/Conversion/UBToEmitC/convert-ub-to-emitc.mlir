// RUN: mlir-opt -convert-ub-to-emitc %s | FileCheck %s

// CHECK-LABEL: func.func @poison
func.func @poison() {
  // CHECK: "emitc.variable"{{.*}} -> i32
  %0 = ub.poison : i32
  // CHECK: "emitc.variable"{{.*}} -> f32
  %1 = ub.poison : f32
  // CHECK: "emitc.variable"{{.*}} -> !emitc.size_t
  %2 = ub.poison : index
  return
}
