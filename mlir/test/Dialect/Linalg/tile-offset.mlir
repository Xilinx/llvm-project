// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s


// CHECK-LABEL: func @mod6_tile_size_3
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_3(%arg0 : tensor<9xf32>) -> tensor<6xf32> {
  %empty = tensor.empty() : tensor<6xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][%arg1] [6] [1]
  // CHECK: tensor.extract_slice %arg2[%arg1] [3] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0 + 3)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<9xf32>) outs(%empty : tensor<6xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<6xf32>
  return %generic : tensor<6xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
