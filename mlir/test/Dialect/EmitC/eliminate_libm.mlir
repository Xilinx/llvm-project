// RUN: mlir-opt %s --eliminate-libm --verify-diagnostics --split-input-file | FileCheck %s

// CHECK: emitc.include <"cmath">
// CHECK-NOT: func.func private @expm1
// CHECK-DAG: func.func @call_expm1(%[[IN:.*]]: f64) -> f64
// CHECK-DAG: %[[RESULT:.*]] = emitc.call_opaque "expm1"(%[[IN]]) : (f64) -> f64
// CHECK-DAG: return %[[RESULT]]
module {
  func.func private @expm1(f64) -> f64 attributes {libm, llvm.readnone}
  func.func @call_expm1(%in : f64) -> f64 {
    %e1 = func.call @expm1(%in) : (f64) -> f64
    return %e1 : f64
  }
}

// -----

// CHECK-NOT: emitc.include <"cmath">
// CHECK: func.func private @expm1
// CHECK: func.func @call_expm1(%[[IN:.*]]: f64) -> f64
// CHECK-NEXT: %[[RESULT:.*]] = call @expm1(%[[IN]]) : (f64) -> f64
// CHECK-NEXT: return %[[RESULT]]
module {
  func.func private @expm1(f64) -> f64 attributes {llvm.readnone}
  func.func @call_expm1(%in : f64) -> f64 {
    %e1 = func.call @expm1(%in) : (f64) -> f64
    return %e1 : f64
  }
}
