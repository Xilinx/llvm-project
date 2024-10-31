// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file -verify-diagnostics

func.func @memref_op(%arg0 : memref<2x4xf32>) {
  // expected-error@+1 {{failed to legalize operation 'memref.copy'}}
  memref.copy %arg0, %arg0 : memref<2x4xf32> to memref<2x4xf32>
  return
}

// -----

func.func @alloca_with_dynamic_shape() {
  %0 = index.constant 1
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %1 = memref.alloca(%0) : memref<4x?xf32>
  return
}

// -----

func.func @alloca_with_alignment() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() {alignment = 64 : i64}: memref<4xf32>
  return
}

// -----

func.func @non_identity_layout() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<4x3xf32, affine_map<(d0, d1) -> (d1, d0)>>
  return
}

// -----

func.func @zero_rank() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<f32>
  return
}

// -----

// expected-error@+1 {{failed to legalize operation 'memref.global'}}
memref.global "nested" constant @nested_global : memref<3x7xf32>

// -----

// CHECK-LABEL: memref_expand_dyn_shape
func.func @memref_expand_dyn_shape(%arg: memref<?xi32>, %size: index) -> memref<?x5xi32> {
  // expected-error@+1 {{failed to legalize operation 'memref.expand_shape'}}
  %0 = memref.expand_shape %arg [[0, 1]] output_shape [%size, 5] : memref<?xi32> into memref<?x5xi32>
  return %0 : memref<?x5xi32>
}

// -----

// CHECK-LABEL: memref_collapse_dyn_shape
func.func @memref_collapse_dyn_shape(%arg: memref<?x5xi32>) -> memref<?xi32> {
  // expected-error@+1 {{failed to legalize operation 'memref.collapse_shape'}}
  %0 = memref.collapse_shape %arg [[0, 1]] : memref<?x5xi32> into memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL: memref_reinterpret_cast_dyn_shape
func.func @memref_reinterpret_cast_dyn_shape(%arg: memref<2x5xi32>, %size: index) -> memref<?xi32> {
  // expected-error@+1 {{failed to legalize operation 'memref.reinterpret_cast'}}
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [%size], strides: [1] : memref<2x5xi32> to memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL: memref_reinterpret_cast_dyn_offset
func.func @memref_reinterpret_cast_dyn_offset(%arg: memref<2x5xi32>, %offset: index) -> memref<10xi32, strided<[1], offset: ?>> {
  // expected-error@+1 {{failed to legalize operation 'memref.reinterpret_cast'}}
  %0 = memref.reinterpret_cast %arg to offset: [%offset], sizes: [10], strides: [1] : memref<2x5xi32> to memref<10xi32, strided<[1], offset: ?>>
  return %0 : memref<10xi32, strided<[1], offset:? >>
}

// -----

// CHECK-LABEL: memref_reinterpret_cast_static_offset
func.func @memref_reinterpret_cast_static_offset(%arg: memref<2x5xi32>) -> memref<10xi32, strided<[1], offset: 10>> {
  // expected-error@+1 {{failed to legalize operation 'memref.reinterpret_cast'}}
  %0 = memref.reinterpret_cast %arg to offset: [10], sizes: [10], strides: [1] : memref<2x5xi32> to memref<10xi32, strided<[1], offset: 10>>
  return %0 : memref<10xi32, strided<[1], offset: 10>>
}

// -----

// CHECK-LABEL: memref_reinterpret_cast_static_strides
func.func @memref_reinterpret_cast_offset(%arg: memref<2x5xi32>) -> memref<10xi32, strided<[2], offset: 0>> {
  // expected-error@+1 {{failed to legalize operation 'memref.reinterpret_cast'}}
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [10], strides: [2] : memref<2x5xi32> to memref<10xi32, strided<[2], offset: 0>>
  return %0 : memref<10xi32, strided<[2], offset: 0>>
}

// -----

// CHECK-LABEL: memref_reinterpret_cast_dyn_strides
func.func @memref_reinterpret_cast_offset(%arg: memref<2x5xi32>, %stride: index) -> memref<10xi32, strided<[?], offset: 0>> {
  // expected-error@+1 {{failed to legalize operation 'memref.reinterpret_cast'}}
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [10], strides: [%stride] : memref<2x5xi32> to memref<10xi32, strided<[?], offset: 0>>
  return %0 : memref<10xi32, strided<[?], offset: 0>>
}
