// RUN: mlir-opt -p 'builtin.module(convert-ub-to-emitc{no-initialization})' %s | FileCheck %s

// CHECK-LABEL: func.func @poison
func.func @poison() {
  // CHECK: "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
  %0 = ub.poison : i32
  // CHECK: "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  %1 = ub.poison : f32
  // CHECK: "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.size_t
  %2 = ub.poison : index
  return
}
