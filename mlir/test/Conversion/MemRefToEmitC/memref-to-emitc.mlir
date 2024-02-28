// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref_arg
// CHECK-SAME:  !emitc.array<32xf32>)
func.func @memref_arg(%arg0 : memref<32xf32>) {
  func.return
}

// -----

// CHECK-LABEL: memref_return
// CHECK-SAME:  %[[arg0:.*]]: !emitc.array<32xf32>) -> !emitc.array<32xf32>
func.func @memref_return(%arg0 : memref<32xf32>) -> memref<32xf32> {
// CHECK: return %[[arg0]] : !emitc.array<32xf32>
  func.return %arg0 : memref<32xf32>
}

// CHECK-LABEL: memref_call
// CHECK-SAME:  %[[arg0:.*]]: !emitc.array<32xf32>)
func.func @memref_call(%arg0 : memref<32xf32>) {
// CHECK: call @memref_return(%[[arg0]]) : (!emitc.array<32xf32>) -> !emitc.array<32xf32>
  func.call @memref_return(%arg0) : (memref<32xf32>) -> memref<32xf32>
  func.return
}
