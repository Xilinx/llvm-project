// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-expand-shape-bubbling %s | FileCheck %s

func.func @bubble_parallel_reshapes(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<?x?x?x?xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
  %expand = tensor.expand_shape %collapse [[0], [1], [2, 3]]
              output_shape [%s0, %s1, %s2, %s3] : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
  return %expand : tensor<?x?x?x?xf32>
}
//      CHECK: func @bubble_parallel_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:   %[[S0:.+]]: index, %[[S1:.+]]: index, %[[S2:.+]]: index, %[[S3:.+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-SAME:       output_shape [%[[S0]], %[[DIM1]], %[[DIM2]], %[[S2]], %[[S3]]] : tensor<?x?x?x?xf32> into tensor<?x?x?x?x?xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]] {{\[}}[0], [1, 2], [3], [4]] : tensor<?x?x?x?x?xf32> into tensor<?x?x?x?xf32>
//      CHECK:   return %[[COLLAPSE]]

// -----

func.func @no_bubble_full_intersecting_reshapes(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<?x?x?x?xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
  %expand = tensor.expand_shape %collapse [[0], [1, 2], [3]]
              output_shape [%s0, %s1, %s2, %s3] : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
  return %expand : tensor<?x?x?x?xf32>
}
//      CHECK: func @no_bubble_full_intersecting_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2], [3]]
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0], [1, 2], [3]]
//      CHECK:   return %[[EXPAND]]

// -----

func.func @no_bubble_partial_intersecting_reshapes(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<?x?x?x?xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?xf32>
  %expand = tensor.expand_shape %collapse [[0, 1], [2, 3]]
              output_shape [%s0, %s1, %s2, %s3] : tensor<?x?xf32> into tensor<?x?x?x?xf32>
  return %expand : tensor<?x?x?x?xf32>
}
//      CHECK: func @no_bubble_partial_intersecting_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]]
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0, 1], [2, 3]]
//      CHECK:   return %[[EXPAND]]

// -----

//func.func @bubble_parallel_extract_slice(%arg0: tensor<1x255x20x24xf32>) -> tensor<1x3x85x20x20xf32> {
//%extract = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 255, 20, 20] [1, 1, 1, 1] : tensor<1x255x20x24xf32> to tensor<1x255x20x20xf32>
//%expand = tensor.expand_shape %extract [[0], [1, 2], [3], [4]] output_shape [1, 3, 85, 20, 20] : tensor<1x255x20x20xf32> into tensor<1x3x85x20x20xf32>
//return %expand : tensor<1x3x85x20x20xf32>
//}
//      CHECK: func @bubble_parallel_extract_slice
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x255x20x24xf32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1, 2], [3], [4]] output_shape [1, 3, 85, 20, 24] : tensor<1x255x20x24xf32> into tensor<1x3x85x20x24xf32>
//      CHECK:   %[[EXTRACT:.+]] = tensor.extract_slice %[[EXPAND]][0, 0, 0, 0, 0] [1, 3, 85, 20, 20] [1, 1, 1, 1, 1] : tensor<1x3x85x20x24xf32> to tensor<1x3x85x20x20xf32>
//      CHECK:   return %[[EXTRACT]]


// -----

func.func @bubble_parallel_reshapes_unit(%arg0: tensor<33600x1xf32>) -> tensor<4x1x8400xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0, 1]] : tensor<33600x1xf32> into tensor<33600xf32>
  %expand = tensor.expand_shape %collapse [[0, 1, 2]] output_shape [4, 1, 8400] : tensor<33600xf32> into tensor<4x1x8400xf32>
  return %expand : tensor<4x1x8400xf32>
}
//      CHECK: func @bubble_parallel_reshapes_unit
// CHECK-SAME:   %[[ARG0:.+]]: tensor<33600x1xf32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]] output_shape [4, 1, 8400, 1]
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]] {{\[}}[0], [1], [2, 3]]
//      CHECK:   return %[[COLLAPSE]]
