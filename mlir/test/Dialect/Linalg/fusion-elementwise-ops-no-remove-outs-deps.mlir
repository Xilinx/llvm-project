// RUN: mlir-opt %s -p 'builtin.module(func.func(linalg-fuse-elementwise-ops{remove-outs-dependency=0}))' -split-input-file | FileCheck %s

#identity = affine_map<(d0) -> (d0)>

func.func @redudant_copy_with_target_burst_size_two(%arg: tensor<4xf32>) -> tensor<4xf32> attributes {plhw.toplevel} {
  // CHECK-NOT: tensor.empty
  %1 = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"] } ins(%arg: tensor<4xf32>) outs(%arg: tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %exp = arith.negf %in: f32
      linalg.yield %exp : f32
  } -> tensor<4xf32>
  %2 = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel"] } ins(%1: tensor<4xf32>) outs(%arg: tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %exp = arith.mulf %in,%in: f32
      linalg.yield %exp : f32
  } -> tensor<4xf32>
  return %2 : tensor<4xf32>
}