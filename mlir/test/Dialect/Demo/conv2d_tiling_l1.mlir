// RUN: mlir-opt --test-transform-dialect-interpreter --cse --split-input-file %s | FileCheck %s

// CHECK:       #map = affine_map<(d0) -> (d0 * 4)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @fn(%arg0: tensor<28x28xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<28x28xf64>) -> tensor<28x28xf64> {
// CHECK-NEXT:      %0 = tensor.empty() : tensor<28x28xf64>
// CHECK-NEXT:      %1 = scf.forall (%arg3, %arg4) in (7, 7) shared_outs(%arg5 = %0) -> (tensor<28x28xf64>) {
// CHECK-NEXT:        %2 = affine.apply #map(%arg3)
// CHECK-NEXT:        %3 = affine.apply #map(%arg4)
// CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%2, %3] [4, 4] [1, 1] : tensor<28x28xf64> to tensor<4x4xf64>
// CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [1, 1] [1, 1] : tensor<1x1xf64> to tensor<1x1xf64>
// CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg2[%2, %3] [4, 4] [1, 1] : tensor<28x28xf64> to tensor<4x4xf64>
// CHECK-NEXT:        %4 = demo.foo %extracted_slice, %extracted_slice_0, %extracted_slice_1 {opName = "conv_2d"} : tensor<4x4xf64>, tensor<1x1xf64>, tensor<4x4xf64> to tensor<4x4xf64>
// CHECK-NEXT:        scf.forall.in_parallel {
// CHECK-NEXT:          tensor.parallel_insert_slice %4 into %arg5[%2, %3] [4, 4] [1, 1] : tensor<4x4xf64> into tensor<28x28xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return %1 : tensor<28x28xf64>
// CHECK-NEXT:    }

module {
  func.func @fn(%arg0: tensor<28x28xf64>, %arg1: tensor<1x1xf64>, %output: tensor<28x28xf64>) -> tensor<28x28xf64> {
    %res = demo.foo %arg0, %arg1, %output {opName = "conv_2d"} : tensor<28x28xf64>, tensor<1x1xf64>, tensor<28x28xf64> to tensor<28x28xf64>
    return %res : tensor<28x28xf64>
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"demo.foo">):
    %loop_l1, %tiled_l1 = transform.structured.tile_to_forall_op %arg0 tile_sizes [4, 4]
      : (!transform.op<"demo.foo">) -> (!transform.any_op, !transform.any_op)
  }
}

// -----

// CHECK:       #map = affine_map<(d0) -> (d0 * 4)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @fn(%arg0: tensor<28x28xf64>, %arg1: tensor<1x1xf64>, %arg2: tensor<28x28xf64>) -> tensor<28x28xf64> {
// CHECK-NEXT:      %0 = scf.forall (%arg3, %arg4) in (7, 7) shared_outs(%arg5 = %arg2) -> (tensor<28x28xf64>) {
// CHECK-NEXT:        %1 = affine.apply #map(%arg3)
// CHECK-NEXT:        %2 = affine.apply #map(%arg4)
// CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%1, %2] [4, 4] [1, 1] : tensor<28x28xf64> to tensor<4x4xf64>
// CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, 0] [1, 1] [1, 1] : tensor<1x1xf64> to tensor<1x1xf64>
// CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [4, 4] [1, 1] : tensor<28x28xf64> to tensor<4x4xf64>
// CHECK-NEXT:        %3 = linalg.conv_2d ins(%extracted_slice, %extracted_slice_0 : tensor<4x4xf64>, tensor<1x1xf64>) outs(%extracted_slice_1 : tensor<4x4xf64>) -> tensor<4x4xf64>
// CHECK-NEXT:        scf.forall.in_parallel {
// CHECK-NEXT:          tensor.parallel_insert_slice %3 into %arg5[%1, %2] [4, 4] [1, 1] : tensor<4x4xf64> into tensor<28x28xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return %0 : tensor<28x28xf64>
// CHECK-NEXT:    }

module {
  func.func @fn(%arg0: tensor<28x28xf64>, %arg1: tensor<1x1xf64>, %output: tensor<28x28xf64>) -> tensor<28x28xf64> {
    %res = linalg.conv_2d ins(%arg0, %arg1: tensor<28x28xf64>, tensor<1x1xf64>)
                                outs(%output: tensor<28x28xf64>) -> tensor<28x28xf64>
    return %res : tensor<28x28xf64>
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"linalg.conv_2d">):
    %loop_l1, %tiled_l1 = transform.structured.tile_to_forall_op %arg0 tile_sizes [4, 4]
      : (!transform.op<"linalg.conv_2d">) -> (!transform.any_op, !transform.any_op)
  }
}
