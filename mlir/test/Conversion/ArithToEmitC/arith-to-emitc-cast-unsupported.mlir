// RUN: mlir-opt -split-input-file --pass-pipeline="builtin.module(convert-arith-to-emitc{float-to-int-truncate})" -verify-diagnostics %s

func.func @arith_cast_tensor(%arg0: tensor<5xf32>) -> tensor<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : tensor<5xf32> to tensor<5xi32>
  return %t: tensor<5xi32>
}

// -----

func.func @arith_cast_vector(%arg0: vector<5xf32>) -> vector<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : vector<5xf32> to vector<5xi32>
  return %t: vector<5xi32>
}
