// RUN: mlir-opt --test-transform-dialect-interpreter --cse %s | FileCheck %s

// CHECK:       #map = affine_map<(d0) -> (d0 * 4)>
// CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 * 8)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @bar(%arg0: tensor<128x256xf64>, %arg1: tensor<256x512xf64>, %arg2: tensor<128x512xf64>) -> tensor<128x512xf64> {
// CHECK-NEXT:      %0 = tensor.empty() : tensor<128x512xf64>
// CHECK-NEXT:      %1 = scf.forall (%arg3, %arg4) in (32, 64) shared_outs(%arg5 = %0) -> (tensor<128x512xf64>) {
// CHECK-NEXT:        %2 = affine.apply #map(%arg3)
// CHECK-NEXT:        %3 = affine.apply #map1(%arg4)
// CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%2, 0] [4, 256] [1, 1] : tensor<128x256xf64> to tensor<4x256xf64>
// CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, %3] [256, 8] [1, 1] : tensor<256x512xf64> to tensor<256x8xf64>
// CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg2[%2, %3] [4, 8] [1, 1] : tensor<128x512xf64> to tensor<4x8xf64>
// CHECK-NEXT:        %4 = demo.foo %extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<4x256xf64>, tensor<256x8xf64>, tensor<4x8xf64> to tensor<4x8xf64>
// CHECK-NEXT:        scf.forall.in_parallel {
// CHECK-NEXT:          tensor.parallel_insert_slice %4 into %arg5[%2, %3] [4, 8] [1, 1] : tensor<4x8xf64> into tensor<128x512xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return %1 : tensor<128x512xf64>
// CHECK-NEXT:    }

module {
  func.func @bar(%arg0: tensor<128x256xf64>, %arg1: tensor<256x512xf64>, %output: tensor<128x512xf64>) -> tensor<128x512xf64> {
      %res = demo.foo %arg0, %arg1, %output : tensor<128x256xf64>, tensor<256x512xf64>, tensor<128x512xf64> to tensor<128x512xf64>
      return %res : tensor<128x512xf64>
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"demo.foo">):
    %loop_l1, %tiled_l1 = transform.structured.tile_to_forall_op %arg0 tile_sizes [4, 8]
      : (!transform.op<"demo.foo">) -> (!transform.any_op, !transform.any_op)
  }
}
