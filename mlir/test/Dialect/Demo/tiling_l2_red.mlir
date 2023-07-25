// RUN: mlir-opt --test-transform-dialect-interpreter --cse %s | FileCheck %s

// CHECK:       #map = affine_map<(d0) -> (d0 * 4)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @bar(%arg0: tensor<128x256xf64>, %arg1: tensor<256x512xf64>, %arg2: tensor<128x512xf64>) -> tensor<128x512xf64> {
// CHECK-NEXT:      %0 = tensor.empty() : tensor<128x512xf64>
// CHECK-NEXT:      %1 = scf.forall (%arg3, %arg4) in (32, 128) shared_outs(%arg5 = %0) -> (tensor<128x512xf64>) {
// CHECK-NEXT:        %2 = affine.apply #map(%arg3)
// CHECK-NEXT:        %3 = affine.apply #map(%arg4)
// CHECK-NEXT:        %extracted_slice = tensor.extract_slice %arg0[%2, 0] [4, 256] [1, 1] : tensor<128x256xf64> to tensor<4x256xf64>
// CHECK-NEXT:        %extracted_slice_0 = tensor.extract_slice %arg1[0, %3] [256, 4] [1, 1] : tensor<256x512xf64> to tensor<256x4xf64>
// CHECK-NEXT:        %extracted_slice_1 = tensor.extract_slice %arg2[%2, %3] [4, 4] [1, 1] : tensor<128x512xf64> to tensor<4x4xf64>
// CHECK-NEXT:        %4 = tensor.empty() : tensor<4x4xf64>
// CHECK-NEXT:        %5 = scf.forall (%arg6, %arg7) in (1, 1) shared_outs(%arg8 = %4) -> (tensor<4x4xf64>) {
// CHECK-NEXT:          %6 = affine.apply #map(%arg6)
// CHECK-NEXT:          %7 = affine.apply #map(%arg7)
// CHECK-NEXT:          %extracted_slice_2 = tensor.extract_slice %extracted_slice[%6, 0] [4, 256] [1, 1] : tensor<4x256xf64> to tensor<4x256xf64>
// CHECK-NEXT:          %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0, %7] [256, 4] [1, 1] : tensor<256x4xf64> to tensor<256x4xf64>
// CHECK-NEXT:          %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[%6, %7] [4, 4] [1, 1] : tensor<4x4xf64> to tensor<4x4xf64>
// CHECK-NEXT:          %c16 = arith.constant 16 : index
// CHECK-NEXT:          %c0 = arith.constant 0 : index
// CHECK-NEXT:          %c256 = arith.constant 256 : index
// CHECK-NEXT:          %8 = scf.for %arg9 = %c0 to %c256 step %c16 iter_args(%arg10 = %4) -> (tensor<4x4xf64>) {
// CHECK-NEXT:            %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[0, %arg9] [4, 16] [1, 1] : tensor<4x256xf64> to tensor<4x16xf64>
// CHECK-NEXT:            %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[%arg9, 0] [16, 4] [1, 1] : tensor<256x4xf64> to tensor<16x4xf64>
// CHECK-NEXT:            %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, 0] [4, 4] [1, 1] : tensor<4x4xf64> to tensor<4x4xf64>
// CHECK-NEXT:            %9 = demo.foo %extracted_slice_5, %extracted_slice_6, %extracted_slice_7 : tensor<4x16xf64>, tensor<16x4xf64>, tensor<4x4xf64> to tensor<4x4xf64>
// CHECK-NEXT:            %inserted_slice = tensor.insert_slice %9 into %arg10[0, 0] [4, 4] [1, 1] : tensor<4x4xf64> into tensor<4x4xf64>
// CHECK-NEXT:            scf.yield %inserted_slice : tensor<4x4xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.forall.in_parallel {
// CHECK-NEXT:            tensor.parallel_insert_slice %8 into %arg8[%6, %7] [4, 4] [1, 1] : tensor<4x4xf64> into tensor<4x4xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.forall.in_parallel {
// CHECK-NEXT:          tensor.parallel_insert_slice %5 into %arg5[%2, %3] [4, 4] [1, 1] : tensor<4x4xf64> into tensor<128x512xf64>
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
    %loop_l2, %tiled_l2 = transform.structured.tile_to_forall_op %arg0 tile_sizes [4, 4]
      : (!transform.op<"demo.foo">) -> (!transform.any_op, !transform.any_op)
    %loop_l1, %tiled_l1 = transform.structured.tile_to_forall_op %tiled_l2 tile_sizes [4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %loop_red, %tiled_red = transform.structured.tile_to_scf_for %tiled_l1 [0, 0, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  }
}
