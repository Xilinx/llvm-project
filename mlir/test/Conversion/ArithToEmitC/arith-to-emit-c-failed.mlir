// RUN: mlir-opt -split-input-file -convert-arith-to-emitc -verify-diagnostics %s

func.func @arith_constant_complex_tensor() -> (tensor<complex<i32>>) {
  // expected-error @+1 {{failed to legalize operation 'arith.constant' that was explicitly marked illegal}}
  %c = arith.constant dense<(2, 2)> : tensor<complex<i32>>
  return %c : tensor<complex<i32>>
}

// -----

func.func @arith_constant_invalid_int_type() -> (i10) {
  // expected-error @+1 {{failed to legalize operation 'arith.constant' that was explicitly marked illegal}}
  %c = arith.constant 0 : i10
  return %c : i10
}
