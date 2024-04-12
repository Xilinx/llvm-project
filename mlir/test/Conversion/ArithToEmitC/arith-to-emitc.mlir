// RUN: mlir-opt -split-input-file -convert-arith-to-emitc %s | FileCheck %s

// CHECK-LABEL: arith_constants
func.func @arith_constants() {
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : index
  %c_index = arith.constant 0 : index
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0 : i32
  %c_signless_int_32 = arith.constant 0 : i32
  // CHECK: emitc.constant
  // CHECK-SAME: value = 0.{{0+}}e+00 : f32
  %c_float_32 = arith.constant 0.0 : f32
  // CHECK: emitc.constant
  // CHECK-SAME: value = dense<0> : tensor<i32>
  %c_tensor_single_value = arith.constant dense<0> : tensor<i32>
  // CHECK: emitc.constant
  // CHECK-SAME: value{{.*}}[1, 2], [-3, 9], [0, 0], [2, -1]{{.*}}tensor<4x2xi64>
  %c_tensor_value = arith.constant dense<[[1, 2], [-3, 9], [0, 0], [2, -1]]> : tensor<4x2xi64>
  return
}

// -----

func.func @arith_ops(%arg0: f32, %arg1: f32) {
  // CHECK: [[V0:[^ ]*]] = emitc.add %arg0, %arg1 : (f32, f32) -> f32
  %0 = arith.addf %arg0, %arg1 : f32
  // CHECK: [[V1:[^ ]*]] = emitc.div %arg0, %arg1 : (f32, f32) -> f32
  %1 = arith.divf %arg0, %arg1 : f32  
  // CHECK: [[V2:[^ ]*]] = emitc.mul %arg0, %arg1 : (f32, f32) -> f32
  %2 = arith.mulf %arg0, %arg1 : f32
  // CHECK: [[V3:[^ ]*]] = emitc.sub %arg0, %arg1 : (f32, f32) -> f32
  %3 = arith.subf %arg0, %arg1 : f32

  return
}

// -----

func.func @arith_integer_ops(%arg0: i32, %arg1: i32) {
  // CHECK: emitc.add %arg0, %arg1 : (i32, i32) -> i32
  %0 = arith.addi %arg0, %arg1 : i32
  // CHECK: emitc.sub %arg0, %arg1 : (i32, i32) -> i32
  %1 = arith.subi %arg0, %arg1 : i32
  // CHECK: emitc.mul %arg0, %arg1 : (i32, i32) -> i32
  %2 = arith.muli %arg0, %arg1 : i32

  return
}

// -----

func.func @arith_select(%arg0: i1, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> () {
  // CHECK: [[V0:[^ ]*]] = emitc.conditional %arg0, %arg1, %arg2 : tensor<8xi32>
  %0 = arith.select %arg0, %arg1, %arg2 : i1, tensor<8xi32>
  return
}

// -----

func.func @arith_cmpf_ueq(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ueq
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[EQ:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UEQ:[^ ]*]] = emitc.logical_or [[Unordered]], [[EQ]] : i1, i1
  %ueq = arith.cmpf ueq, %arg0, %arg1 : f32
  // CHECK: return [[UEQ]]
  return %ueq: i1
}

// -----

func.func @arith_cmpf_ugt(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ugt
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GT:[^ ]*]] = emitc.cmp gt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UGT:[^ ]*]] = emitc.logical_or [[Unordered]], [[GT]] : i1, i1
  %ugt = arith.cmpf ugt, %arg0, %arg1 : f32
  // CHECK: return [[UGT]]
  return %ugt: i1
}

// -----

func.func @arith_cmpf_uge(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_uge
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GE:[^ ]*]] = emitc.cmp ge, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[UGE:[^ ]*]] = emitc.logical_or [[Unordered]], [[GE]] : i1, i1
  %uge = arith.cmpf uge, %arg0, %arg1 : f32
  // CHECK: return [[UGE]]
  return %uge: i1
}

// -----

func.func @arith_cmpf_ult(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ult
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[LT:[^ ]*]] = emitc.cmp lt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[ULT:[^ ]*]] = emitc.logical_or [[Unordered]], [[LT]] : i1, i1
  %ult = arith.cmpf ult, %arg0, %arg1 : f32
  // CHECK: return [[ULT]]
  return %ult: i1
}

// -----

func.func @arith_cmpf_ule(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ule
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[LE:[^ ]*]] = emitc.cmp le, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  // CHECK-DAG: [[ULE:[^ ]*]] = emitc.logical_or [[Unordered]], [[LE]] : i1, i1
  %ule = arith.cmpf ule, %arg0, %arg1 : f32
  // CHECK: return [[ULE]]
  return %ule: i1
}

// -----

func.func @arith_cmpf_uno(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_uno
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[NaNArg0:[^ ]*]] = emitc.cmp ne, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[NaNArg1:[^ ]*]] = emitc.cmp ne, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Unordered:[^ ]*]] = emitc.logical_or [[NaNArg0]], [[NaNArg1]] : i1, i1
  %2 = arith.cmpf uno, %arg0, %arg1 : f32
  // CHECK: return [[Unordered]]
  return %2: i1
}

// -----

func.func @arith_cmpf_ogt(%arg0: f32, %arg1: f32) -> i1 {
  // CHECK-LABEL: arith_cmpf_ogt
  // CHECK-SAME: ([[Arg0:[^ ]*]]: f32, [[Arg1:[^ ]*]]: f32)
  // CHECK-DAG: [[GT:[^ ]*]] = emitc.cmp gt, [[Arg0]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[OrderedArg0:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg0]] : (f32, f32) -> i1
  // CHECK-DAG: [[OrderedArg1:[^ ]*]] = emitc.cmp eq, [[Arg1]], [[Arg1]] : (f32, f32) -> i1
  // CHECK-DAG: [[Ordered:[^ ]*]] = emitc.logical_and [[OrderedArg0]], [[OrderedArg1]] : i1, i1
  // CHECK-DAG: [[OGT:[^ ]*]] = emitc.logical_and [[Ordered]], [[GT]] : i1, i1
  %ule = arith.cmpf ogt, %arg0, %arg1 : f32
  // CHECK: return [[OGT]]
  return %ule: i1
}

// -----

func.func @arith_int_to_float_cast_ops(%arg0: i8, %arg1: i64) {
  // CHECK: emitc.cast %arg0 : i8 to f32
  %0 = arith.sitofp %arg0 : i8 to f32

  // CHECK: emitc.cast %arg1 : i64 to f32
  %1 = arith.sitofp %arg1 : i64 to f32

  // CHECK: emitc.cast %arg0 : i8 to f32
  %2 = arith.uitofp %arg0 : i8 to f32

  return
}

// -----

func.func @arith_cmpi_eq(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK-LABEL: arith_cmpi_eq
  // CHECK-SAME: ([[Arg0:[^ ]*]]: i32, [[Arg1:[^ ]*]]: i32)
  // CHECK-DAG: [[EQ:[^ ]*]] = emitc.cmp eq, [[Arg0]], [[Arg1]] : (i32, i32) -> i1
  %eq = arith.cmpi eq, %arg0, %arg1 : i32
  // CHECK: return [[EQ]]
  return %eq: i1
}

func.func @arith_cmpi_ult(%arg0: i32, %arg1: i32) -> i1 {
  // CHECK-LABEL: arith_cmpi_ult
  // CHECK-SAME: ([[Arg0:[^ ]*]]: i32, [[Arg1:[^ ]*]]: i32)
  // CHECK-DAG: [[CastArg0:[^ ]*]] = emitc.cast [[Arg0]] : i32 to ui32
  // CHECK-DAG: [[CastArg1:[^ ]*]] = emitc.cast [[Arg1]] : i32 to ui32
  // CHECK-DAG: [[ULT:[^ ]*]] = emitc.cmp lt, [[CastArg0]], [[CastArg1]] : (ui32, ui32) -> i1
  %eq = arith.cmpi ult, %arg0, %arg1 : i32
  // CHECK: return [[ULT]]
  return %eq: i1
}