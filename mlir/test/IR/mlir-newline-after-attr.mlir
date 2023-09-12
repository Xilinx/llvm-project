// RUN: mlir-opt %s -mlir-newline-after-attr=2 | FileCheck %s
// Ensure that the printed version is still parseable.
// RUN: mlir-opt %s -mlir-newline-after-attr=2 | mlir-opt

// CHECK: foo.dense_attr =
"test.op"() {foo.dense_attr = dense<1> : tensor<3xi32>} : () -> ()

// CHECK: foo.dense_attr =
// CHECK: foo.second_attr =
"test.op"() {foo.dense_attr = dense<1> : tensor<3xi32>, foo.second_attr = dense<2> : tensor<3xi32>} : () -> ()

// CHECK: Operands = [{foo.vect_attr_1_count = dense<1> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_1_end = dense<0> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_1_start = dense<0> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_2_count = dense<1> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_2_end = dense<0> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_2_start = dense<0> : vector<3xindex>}, {foo.vect_attr_1_count = dense<1> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_1_end = dense<0> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_1_start = dense<0> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_2_count = dense<1> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_2_end = dense<0> : vector<3xindex>,
// CHECK-NEXT: foo.vect_attr_2_start = dense<0> : vector<3xindex>}],
"test.op"() {foo.dense_attr = dense<1> : tensor<3xi32>, foo.second_attr = dense<2> : tensor<3xi32>, Operands = [{foo.vect_attr_1_start = dense<0> : vector<3xindex>, foo.vect_attr_1_end = dense<0> : vector<3xindex>, foo.vect_attr_1_count = dense<1> : vector<3xindex>, foo.vect_attr_2_start = dense<0> : vector<3xindex>, foo.vect_attr_2_end = dense<0> : vector<3xindex>, foo.vect_attr_2_count = dense<1> : vector<3xindex>}, {foo.vect_attr_1_start = dense<0> : vector<3xindex>, foo.vect_attr_1_end = dense<0> : vector<3xindex>, foo.vect_attr_1_count = dense<1> : vector<3xindex>, foo.vect_attr_2_start = dense<0> : vector<3xindex>, foo.vect_attr_2_end = dense<0> : vector<3xindex>, foo.vect_attr_2_count = dense<1> : vector<3xindex>}]} : () -> ()

