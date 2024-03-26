// RUN: mlir-opt -split-input-file -convert-arith-to-emitc -verify-diagnostics %s

func.func @arith_cmpf_tensor(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xi1> {
  // expected-error @+1 {{failed to legalize operation 'arith.cmpf'}}
  %t = arith.cmpf uno, %arg0, %arg1 : tensor<5xf32>
  return %t: tensor<5xi1>
}

// -----

func.func @arith_cmpf_vector(%arg0: vector<5xf32>, %arg1: vector<5xf32>) -> vector<5xi1> {
  // expected-error @+1 {{failed to legalize operation 'arith.cmpf'}}
  %t = arith.cmpf uno, %arg0, %arg1 : vector<5xf32>
  return %t: vector<5xi1>
}

// -----

func.func @arith_cmpf_ordered(%arg0: f32, %arg1: f32) -> i1 {
  // expected-error @+1 {{failed to legalize operation 'arith.cmpf'}}
  %oge = arith.cmpf oge, %arg0, %arg1 : f32
  return %oge: i1
}
