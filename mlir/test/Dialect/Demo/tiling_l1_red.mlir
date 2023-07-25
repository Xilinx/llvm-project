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
// CHECK-NEXT:        %4 = tensor.empty() : tensor<4x8xf64>
// CHECK-NEXT:        %c32 = arith.constant 32 : index
// CHECK-NEXT:        %c0 = arith.constant 0 : index
// CHECK-NEXT:        %c256 = arith.constant 256 : index
// CHECK-NEXT:        %5 = scf.for %arg6 = %c0 to %c256 step %c32 iter_args(%arg7 = %4) -> (tensor<4x8xf64>) {
// CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg6] [4, 32] [1, 1] : tensor<4x256xf64> to tensor<4x32xf64>
// CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [32, 8] [1, 1] : tensor<256x8xf64> to tensor<32x8xf64>
// CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, 0] [4, 8] [1, 1] : tensor<4x8xf64> to tensor<4x8xf64>
// CHECK-NEXT:          %6 = demo.foo %extracted_slice_2, %extracted_slice_3, %extracted_slice_4 : tensor<4x32xf64>, tensor<32x8xf64>, tensor<4x8xf64> to tensor<4x8xf64>
// CHECK-NEXT:          %inserted_slice = tensor.insert_slice %6 into %arg7[0, 0] [4, 8] [1, 1] : tensor<4x8xf64> into tensor<4x8xf64>
// CHECK-NEXT:          scf.yield %inserted_slice : tensor<4x8xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.forall.in_parallel {
// CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg5[%2, %3] [4, 8] [1, 1] : tensor<4x8xf64> into tensor<128x512xf64>
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
    %loop_red, %tiled_red = transform.structured.tile_to_scf_for %tiled_l1 [0, 0, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  }
}
