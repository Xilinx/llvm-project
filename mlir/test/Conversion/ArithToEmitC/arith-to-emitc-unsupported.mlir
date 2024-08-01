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

// -----

func.func @arith_cast_bf16(%arg0: bf16) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : bf16 to i32
  return %t: i32
}

// -----

func.func @arith_cast_f16(%arg0: f16) -> i32 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : f16 to i32
  return %t: i32
}


// -----

func.func @arith_cast_to_bf16(%arg0: i32) -> bf16 {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %t = arith.sitofp %arg0 : i32 to bf16
  return %t: bf16
}

// -----

func.func @arith_cast_to_f16(%arg0: i32) -> f16 {
  // expected-error @+1 {{failed to legalize operation 'arith.sitofp'}}
  %t = arith.sitofp %arg0 : i32 to f16
  return %t: f16
}

// -----

func.func @arith_cast_fptosi_i1(%arg0: f32) -> i1 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptosi'}}
  %t = arith.fptosi %arg0 : f32 to i1
  return %t: i1
}

// -----

func.func @arith_cast_fptoui_i1(%arg0: f32) -> i1 {
  // expected-error @+1 {{failed to legalize operation 'arith.fptoui'}}
  %t = arith.fptoui %arg0 : f32 to i1
  return %t: i1
}

// -----

func.func @arith_negf_tensor(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.negf'}}
  %n = arith.negf %arg0 : tensor<5xf32>
  return %n: tensor<5xf32>
}

// -----

func.func @arith_negf_vector(%arg0: vector<5xf32>) -> vector<5xf32> {
  // expected-error @+1 {{failed to legalize operation 'arith.negf'}}
  %n = arith.negf %arg0 : vector<5xf32>
  return %n: vector<5xf32>
}

// -----

func.func @arith_shli_i1(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.shli'}}
  %shli = arith.shli %arg0, %arg1 : i1
  return
}

// -----

func.func @arith_shrsi_i1(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.shrsi'}}
  %shrsi = arith.shrsi %arg0, %arg1 : i1
  return
}

// -----

func.func @arith_shrui_i1(%arg0: i1, %arg1: i1) {
  // expected-error @+1 {{failed to legalize operation 'arith.shrui'}}
  %shrui = arith.shrui %arg0, %arg1 : i1
  return
}

// -----

func.func @arith_divui_vector(%arg0: vector<5xi32>, %arg1: vector<5xi32>) -> vector<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.divui'}}
  %divui = arith.divui %arg0, %arg1 : vector<5xi32>
  return %divui: vector<5xi32>
}

// -----

func.func @arith_remui_vector(%arg0: vector<5xi32>, %arg1: vector<5xi32>) -> vector<5xi32> {
  // expected-error @+1 {{failed to legalize operation 'arith.remui'}}
  %divui = arith.remui %arg0, %arg1 : vector<5xi32>
  return %divui: vector<5xi32>
}

// -----

func.func @arith_extf_to_bf16(%arg0: f8E4M3FN) {
  // expected-error @+1 {{failed to legalize operation 'arith.extf'}}
  %ext = arith.extf %arg0 : f8E4M3FN to bf16
  return
}

// -----

func.func @arith_extf_to_f16(%arg0: f8E4M3FN) {
  // expected-error @+1 {{failed to legalize operation 'arith.extf'}}
  %ext = arith.extf %arg0 : f8E4M3FN to f16
  return
}


// -----

func.func @arith_extf_to_tf32(%arg0: f8E4M3FN) {
  // expected-error @+1 {{failed to legalize operation 'arith.extf'}}
  %ext = arith.extf %arg0 : f8E4M3FN to tf32
  return
}

// -----

func.func @arith_extf_to_float80(%arg0: f8E4M3FN) {
  // expected-error @+1 {{failed to legalize operation 'arith.extf'}}
  %ext = arith.extf %arg0 : f8E4M3FN to f80
  return
}

// -----

func.func @arith_extf_to_float128(%arg0: f8E4M3FN) {
  // expected-error @+1 {{failed to legalize operation 'arith.extf'}}
  %ext = arith.extf %arg0 : f8E4M3FN to f128
  return
}

// -----

func.func @arith_truncf_to_f80(%arg0: f128) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f128 to f80
  return
}

// -----

func.func @arith_truncf_to_tf32(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to tf32
  return
}

// -----

func.func @arith_truncf_to_f16(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to f16
  return
}

// -----

func.func @arith_truncf_to_bf16(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to bf16
  return
}

// -----

func.func @arith_truncf_to_f8E4M3FN(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to f8E4M3FN
  return
}

// -----

func.func @arith_truncf_to_f8E5M2(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to f8E5M2
  return
}

// -----

func.func @arith_truncf_to_f8E4M3FNUZ(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to f8E4M3FNUZ
  return
}

// -----

func.func @arith_truncf_to_f8E4M3FN(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to f8E4M3FN
  return
}

// -----

func.func @arith_truncf_to_f8E4M3B11FNUZ(%arg0: f64) {
  // expected-error @+1 {{failed to legalize operation 'arith.truncf'}}
  %trunc = arith.truncf %arg0 : f64 to f8E4M3B11FNUZ
  return
}
