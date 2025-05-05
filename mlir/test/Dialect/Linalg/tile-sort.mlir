// RUN: mlir-opt %s -transform-interpreter -split-input-file -debug-only=tile-using-interface 2>&1 | FileCheck %s

func.func @tile_in_op_operand_order(%arg: tensor<256xf32>) -> tensor<256xf32> {
  %empty = tensor.empty() : tensor<256xf32>
  %0 = linalg.ceil ins(%arg: tensor<256xf32>) outs(%empty: tensor<256xf32>) -> tensor<256xf32>
  %empty1 = tensor.empty() : tensor<256xf32>
  %1 = linalg.negf ins(%0 : tensor<256xf32>) outs(%empty1: tensor<256xf32>) -> tensor<256xf32>
  %empty2 = tensor.empty() : tensor<256xf32>
  %2 = linalg.powf {tile} ins(%0, %1: tensor<256xf32>, tensor<256xf32>) outs(%empty2: tensor<256xf32>) -> tensor<256xf32>

  // CHECK: worklist: producer is %{{.*}} = linalg.ceil
  // CHECK: worklist: producer is %{{.*}} = linalg.negf
  // CHECK: worklist: producer is %{{.*}} = linalg.ceil

  return %2 : tensor<256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.powf"]} attributes {"tile"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.test.fuse_and_yield %0 [32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
