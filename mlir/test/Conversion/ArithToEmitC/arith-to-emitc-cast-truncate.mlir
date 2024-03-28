// RUN: mlir-opt -split-input-file --pass-pipeline="builtin.module(convert-arith-to-emitc{float-to-int-truncates})" %s | FileCheck %s

func.func @arith_float_to_int_cast_ops(%arg0: f32, %arg1: f64) {
  // CHECK: emitc.cast %arg0 : f32 to i32
  %0 = arith.fptosi %arg0 : f32 to i32

  // CHECK: emitc.cast %arg1 : f64 to i32
  %1 = arith.fptosi %arg1 : f64 to i32

  // CHECK: emitc.cast %arg0 : f32 to i16
  %2 = arith.fptosi %arg0 : f32 to i16

  // CHECK: emitc.cast %arg1 : f64 to i16
  %3 = arith.fptosi %arg1 : f64 to i16

  // CHECK: emitc.cast %arg0 : f32 to i32
  %4 = arith.fptoui %arg0 : f32 to i32
  
  return
}
