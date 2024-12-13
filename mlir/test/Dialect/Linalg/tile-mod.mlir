// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s


// tiling with 'mod' in affine maps is only safe when
// either 
// - the mod expression doesn't wrap around in the tile
// - OR the dimension is full


// CHECK: #[[MAP:[a-zA-Z0-9_]+]] = affine_map<(d0) -> (d0 mod 6)>
// CHECK: func @mod6_tile_size_full
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_full(%arg0 : tensor<60xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: %[[APPLY_OFFSET:[a-zA-Z0-9_]+]] = affine.apply #[[MAP]](%arg1)
  // CHECK: tensor.extract_slice %[[ARG0]][%[[APPLY_OFFSET]]] [6] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0 mod 6)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<60xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [60] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @mod6_tile_size_6
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_6(%arg0 : tensor<60xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][{{.*}}] [6] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0 mod 6)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<60xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [6] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @mod6_tile_size_3
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_3(%arg0 : tensor<60xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][{{.*}}] [3] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0 mod 6)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<60xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [3] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @mod6_tile_size_2
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_2(%arg0 : tensor<60xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][{{.*}}] [2] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0 mod 6)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<60xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @mod6_tile_size_1
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_1(%arg0 : tensor<60xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][{{.*}}] [1] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0 mod 6)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<60xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @mod6_tile_size_full_scaled
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_full_scaled(%arg0 : tensor<200xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][{{.*}}] [16] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> ((d0 mod 6)*3)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<200xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [60] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @mod6_tile_size_2_scaled
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]:
func.func @mod6_tile_size_2_scaled(%arg0 : tensor<60xf32>) -> tensor<60xf32> {
  %empty = tensor.empty() : tensor<60xf32>
  // CHECK: tensor.extract_slice %[[ARG0]][{{.*}}] [8] [1]
  %generic = linalg.generic
    {indexing_maps = [affine_map<(d0) -> ((d0 mod 6)*7)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%arg0: tensor<60xf32>) outs(%empty : tensor<60xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<60xf32>
  return %generic : tensor<60xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d4) -> (d0, ((d1 * 85 + d4) mod 255) floordiv 8)>
#map1 = affine_map<(d0, d1, d4) -> (d0, d1, d4)>
func.func @complex_mod_expr(%arg0: tensor<1x32xi8>) -> tensor<1x3x85xf32> {
    %0 = tensor.empty() : tensor<1x3x85xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%arg0 : tensor<1x32xi8>) outs(%0 : tensor<1x3x85xf32>) {
    ^bb0(%in: i8, %out: f32):
      %2 = arith.sitofp %in : i8 to f32
      linalg.yield %2 : f32
    } -> tensor<1x3x85xf32>
    return %1 : tensor<1x3x85xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %loop = transform.structured.tile_using_for %0 tile_sizes [1,0,0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
