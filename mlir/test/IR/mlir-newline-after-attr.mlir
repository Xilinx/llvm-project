// RUN: mlir-opt %s -mlir-newline-after-attr=2 | FileCheck %s
// Ensure that the printed version is still parseable.
// RUN: mlir-opt %s -mlir-newline-after-attr=2 | mlir-opt

// CHECK: foo.dense_attr =
"test.op"() {foo.dense_attr = dense<1> : tensor<3xi32>} : () -> ()

// CHECK: foo.dense_attr =
// CHECK: foo.second_attr =
"test.op"() {foo.dense_attr = dense<1> : tensor<3xi32>, foo.second_attr = dense<2> : tensor<3xi32>} : () -> ()
