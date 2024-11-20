// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-emitc %s | FileCheck %s

func.func @test_if(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
     %0 = emitc.call_opaque "func_const"(%arg1) : (f32) -> i32
  }
  return
}
// CHECK-LABEL: func.func @test_if(
// CHECK-SAME:                     %[[VAL_0:.*]]: i1,
// CHECK-SAME:                     %[[VAL_1:.*]]: f32) {
// CHECK-NEXT:    emitc.if %[[VAL_0]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = emitc.call_opaque "func_const"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @test_if_else(%arg0: i1, %arg1: f32) {
  scf.if %arg0 {
    %0 = emitc.call_opaque "func_true"(%arg1) : (f32) -> i32
  } else {
    %0 = emitc.call_opaque "func_false"(%arg1) : (f32) -> i32
  }
  return
}
// CHECK-LABEL: func.func @test_if_else(
// CHECK-SAME:                          %[[VAL_0:.*]]: i1,
// CHECK-SAME:                          %[[VAL_1:.*]]: f32) {
// CHECK-NEXT:    emitc.if %[[VAL_0]] {
// CHECK-NEXT:      %[[VAL_2:.*]] = emitc.call_opaque "func_true"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[VAL_3:.*]] = emitc.call_opaque "func_false"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @test_if_yield(%arg0: i1, %arg1: f32) {
  %0 = arith.constant 0 : i8
  %x, %y = scf.if %arg0 -> (i32, f64) {
    %1 = emitc.call_opaque "func_true_1"(%arg1) : (f32) -> i32
    %2 = emitc.call_opaque "func_true_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  } else {
    %1 = emitc.call_opaque "func_false_1"(%arg1) : (f32) -> i32
    %2 = emitc.call_opaque "func_false_2"(%arg1) : (f32) -> f64
    scf.yield %1, %2 : i32, f64
  }
  return
}
// CHECK-LABEL: func.func @test_if_yield(
// CHECK-SAME:                           %[[VAL_0:.*]]: i1,
// CHECK-SAME:                           %[[VAL_1:.*]]: f32) {
// CHECK-NEXT:    %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK-NEXT:    %[[VAL_3:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> i32
// CHECK-NEXT:    %[[VAL_4:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> f64
// CHECK-NEXT:    emitc.if %[[VAL_0]] {
// CHECK-NEXT:      %[[VAL_5:.*]] = emitc.call_opaque "func_true_1"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:      %[[VAL_6:.*]] = emitc.call_opaque "func_true_2"(%[[VAL_1]]) : (f32) -> f64
// CHECK-NEXT:      emitc.assign %[[VAL_5]] : i32 to %[[VAL_3]] : i32
// CHECK-NEXT:      emitc.assign %[[VAL_6]] : f64 to %[[VAL_4]] : f64
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[VAL_7:.*]] = emitc.call_opaque "func_false_1"(%[[VAL_1]]) : (f32) -> i32
// CHECK-NEXT:      %[[VAL_8:.*]] = emitc.call_opaque "func_false_2"(%[[VAL_1]]) : (f32) -> f64
// CHECK-NEXT:      emitc.assign %[[VAL_7]] : i32 to %[[VAL_3]] : i32
// CHECK-NEXT:      emitc.assign %[[VAL_8]] : f64 to %[[VAL_4]] : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @test_if_yield_index(%arg0: i1, %arg1: f32) {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %x = scf.if %arg0 -> (index) {
    scf.yield %0 : index
  } else {
    scf.yield %1 : index
  }
  return
}

// CHECK: func.func @test_if_yield_index(
// CHECK-SAME: %[[ARG_0:.*]]: i1, %[[ARG_1:.*]]: f32) {
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_0:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.size_t
// CHECK:   emitc.if %[[ARG_0]] {
// CHECK:     %[[VAL_1:.*]] = builtin.unrealized_conversion_cast %[[C0]] : index to !emitc.size_t
// CHECK:     emitc.assign %[[VAL_1]] : !emitc.size_t to %[[VAL_0]] : !emitc.size_t
// CHECK:   } else {
// CHECK:     %[[VAL_2:.*]] = builtin.unrealized_conversion_cast %[[C1]] : index to !emitc.size_t
// CHECK:     emitc.assign %[[VAL_2]] : !emitc.size_t to %[[VAL_0]] : !emitc.size_t
// CHECK:   }
// CHECK:   return
// CHECK: }
