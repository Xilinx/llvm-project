// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics

func.func @tile_me(%arg: tensor<256xi32>) -> tensor<256xi32> {
  %one = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<256xi32>
  // expected-remark @below {{Fused op in position 0}}
  // expected-remark @below {{Fused op in position 2}}
  %0 = linalg.generic {indexing_maps=[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types=["parallel"]}
  ins(%arg: tensor<256xi32>) outs(%empty: tensor<256xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %plusone = arith.addi %arg0, %one : i32
    linalg.yield %plusone : i32
  } -> tensor<256xi32>

  %empty1 = tensor.empty() : tensor<256xi32>
  %minustwo = arith.constant -2 : i32
  
  // expected-remark @below {{Fused op in position 1}}
  %1 = linalg.generic {indexing_maps=[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types=["parallel"]}
  ins(%0: tensor<256xi32>) outs(%empty: tensor<256xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %opp = arith.muli %arg0, %minustwo : i32
    linalg.yield %opp : i32
  } -> tensor<256xi32>

  %empty2 = tensor.empty() : tensor<256xi32>
  %2 = linalg.generic {tile, indexing_maps=[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types=["parallel"]}
  ins(%0, %1: tensor<256xi32>, tensor<256xi32>) outs(%empty: tensor<256xi32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %sum = arith.addi %arg0, %arg1 : i32
    linalg.yield %sum : i32
  } -> tensor<256xi32>

  return %2 : tensor<256xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {"tile"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.test.fuse_and_yield %0 [32] debug_worklist true : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @tile_me(%arg: tensor<256xi32>) -> tensor<256xi32> {
  %one = arith.constant 1 : i32
  %empty = tensor.empty() : tensor<256xi32>
  // expected-remark @below {{Fused op in position 1}}
  // expected-remark @below {{Fused op in position 2}}
  %0 = linalg.generic {indexing_maps=[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types=["parallel"]}
  ins(%arg: tensor<256xi32>) outs(%empty: tensor<256xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %plusone = arith.addi %arg0, %one : i32
    linalg.yield %plusone : i32
  } -> tensor<256xi32>

  %empty1 = tensor.empty() : tensor<256xi32>
  %minustwo = arith.constant -2 : i32
  
  // expected-remark @below {{Fused op in position 0}}
  %1 = linalg.generic {indexing_maps=[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types=["parallel"]}
  ins(%0: tensor<256xi32>) outs(%empty: tensor<256xi32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %opp = arith.muli %arg0, %minustwo : i32
    linalg.yield %opp : i32
  } -> tensor<256xi32>

  %empty2 = tensor.empty() : tensor<256xi32>
  %2 = linalg.generic {tile, indexing_maps=[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types=["parallel"]}
  ins(%0, %1: tensor<256xi32>, tensor<256xi32>) outs(%empty: tensor<256xi32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %sum = arith.addi %arg0, %arg1 : i32
    linalg.yield %sum : i32
  } -> tensor<256xi32>

  return %2 : tensor<256xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} attributes {"tile"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loops = transform.test.fuse_and_yield %0 [32] debug_worklist true reverse_worklist true : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
