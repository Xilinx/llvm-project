// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

// CHECK-LABEL: memref_store
// CHECK-SAME:  %[[v:.*]]: f32, %[[argi:.*]]: index, %[[argj:.*]]: index
func.func @memref_store(%v : f32, %i: index, %j: index) {
  // CHECK: %[[i:.*]] = builtin.unrealized_conversion_cast %[[argi]] : index to !emitc.size_t 
  // CHECK: %[[j:.*]] = builtin.unrealized_conversion_cast %[[argj]] : index to !emitc.size_t 

  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK: %[[SUBSCRIPT:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, !emitc.size_t, !emitc.size_t) -> f32
  // CHECK: emitc.assign %[[v]] : f32 to %[[SUBSCRIPT:.*]] : f32
  memref.store %v, %0[%i, %j] : memref<4x8xf32>
  return
}

// -----

// CHECK-LABEL: memref_load
// CHECK-SAME:  %[[argi:.*]]: index, %[[argj:.*]]: index
func.func @memref_load(%i: index, %j: index) -> f32 {
  // CHECK: %[[i:.*]] = builtin.unrealized_conversion_cast %[[argi]] : index to !emitc.size_t 
  // CHECK: %[[j:.*]] = builtin.unrealized_conversion_cast %[[argj]] : index to !emitc.size_t 

  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8xf32>
  %0 = memref.alloca() : memref<4x8xf32>

  // CHECK: %[[LOAD:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8xf32>, !emitc.size_t, !emitc.size_t) -> f32
  // CHECK: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f32
  // CHECK: emitc.assign %[[LOAD]] : f32 to %[[VAR]] : f32
  %1 = memref.load %0[%i, %j] : memref<4x8xf32>
  // CHECK: return %[[VAR]] : f32
  return %1 : f32
}

// -----

// CHECK-LABEL: globals
module @globals {
  memref.global "private" constant @internal_global : memref<3x7xf32> = dense<4.0>
  // CHECK: emitc.global static const @internal_global : !emitc.array<3x7xf32> = dense<4.000000e+00>
  memref.global @public_global : memref<3x7xf32>
  // CHECK: emitc.global extern @public_global : !emitc.array<3x7xf32>
  memref.global @uninitialized_global : memref<3x7xf32> = uninitialized
  // CHECK: emitc.global extern @uninitialized_global : !emitc.array<3x7xf32>

  func.func @use_global() {
    // CHECK: emitc.get_global @public_global : !emitc.array<3x7xf32>
    %0 = memref.get_global @public_global : memref<3x7xf32>
    return
  }
}

// -----

// CHECK-LABEL: memref_index_values
// CHECK-SAME:  %[[argi:.*]]: index, %[[argj:.*]]: index
// CHECK-SAME: -> index
func.func @memref_index_values(%i: index, %j: index) -> index {
  // CHECK: %[[i:.*]] = builtin.unrealized_conversion_cast %[[argi]] : index to !emitc.size_t
  // CHECK: %[[j:.*]] = builtin.unrealized_conversion_cast %[[argj]] : index to !emitc.size_t

  // CHECK: %[[ALLOCA:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.array<4x8x!emitc.size_t>
  %0 = memref.alloca() : memref<4x8xindex>

  // CHECK: %[[LOAD:.*]] = emitc.subscript %[[ALLOCA]][%[[i]], %[[j]]] : (!emitc.array<4x8x!emitc.size_t>, !emitc.size_t, !emitc.size_t) -> !emitc.size_t
  // CHECK: %[[VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.size_t
  // CHECK: emitc.assign %[[LOAD]] : !emitc.size_t to %[[VAR]] : !emitc.size_t
  %1 = memref.load %0[%i, %j] : memref<4x8xindex>

  // CHECK: %[[CAST_RET:.*]] = builtin.unrealized_conversion_cast %[[VAR]] : !emitc.size_t to index
  // CHECK: return %[[CAST_RET]] : index
  return %1 : index
}
