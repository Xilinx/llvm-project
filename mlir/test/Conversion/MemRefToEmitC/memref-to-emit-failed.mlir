// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file -verify-diagnostics

// Unranked memrefs are not converted
// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @memref_unranked(%arg0 : memref<*xf32>) {
  return
}

// -----

// Memrefs with dynamic shapes are not converted
// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @memref_dynamic_shape(%arg0 : memref<2x?xf32>) {
  return
}

// -----

// Memrefs with dynamic shapes are not converted
func.func @memref_op(%arg0 : memref<2x4xf32>) {
  // expected-error@+1 {{failed to legalize operation 'memref.copy' that was explicitly marked illegal}}
  memref.copy %arg0, %arg0 : memref<2x4xf32> to memref<2x4xf32>
  return
}
