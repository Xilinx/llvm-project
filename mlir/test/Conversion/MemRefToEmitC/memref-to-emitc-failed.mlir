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

func.func @unsupported_type_f128() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<4xf128>
  return
}

// -----

func.func @unsupported_type_i4() {
  // expected-error@+1 {{failed to legalize operation 'memref.alloca'}}
  %0 = memref.alloca() : memref<4xi4>
  return
}

// -----

func.func @memref_expand_dyn_shape(%arg: memref<?xi32>, %size: index) -> memref<?x5xi32> {
  // expected-error@+1 {{failed to legalize operation 'memref.expand_shape'}}
  %0 = memref.expand_shape %arg [[0, 1]] output_shape [%size, 5] : memref<?xi32> into memref<?x5xi32>
  return %0 : memref<?x5xi32>
}

// -----

func.func @memref_collapse_dyn_shape(%arg: memref<?x5xi32>) -> memref<?xi32> {
  // expected-error@+1 {{failed to legalize operation 'memref.collapse_shape'}}
  %0 = memref.collapse_shape %arg [[0, 1]] : memref<?x5xi32> into memref<?xi32>
  return %0 : memref<?xi32>
}
