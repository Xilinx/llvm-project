// RUN: mlir-opt --test-transform-dialect-interpreter %s | FileCheck %s

// CHECK:         func.func @bar(%arg0: tensor<512x512xf64>, %arg1: tensor<512x512xf64>, %arg2: tensor<512x512xf64>) -> tensor<512x512xf64> {
  // CHECK-NEXT:    %0 = tensor.empty() : tensor<512x512xf64>
  // CHECK-NEXT:    %c64 = arith.constant 64 : index
  // CHECK-NEXT:    %c16 = arith.constant 16 : index
  // CHECK-NEXT:    %1 = scf.forall (%arg3, %arg4) in (64, 16) shared_outs(%arg5 = %0) -> (tensor<512x512xf64>) {
  // CHECK-NEXT:      %2 = affine.apply #map(%arg3)
  // CHECK-NEXT:      %3 = affine.apply #map1(%arg4)
  // CHECK-NEXT:      %4 = affine.apply #map(%arg3)
  // CHECK-NEXT:      %5 = affine.apply #map1(%arg4)
  // CHECK-NEXT:      %6 = affine.apply #map(%arg3)
  // CHECK-NEXT:      %7 = affine.apply #map1(%arg4)
  // CHECK-NEXT:      %extracted_slice = tensor.extract_slice %arg0[%4, 0] [8, 512] [1, 1] : tensor<512x512xf64> to tensor<8x512xf64>
  // CHECK-NEXT:      %extracted_slice_0 = tensor.extract_slice %arg1[0, %5] [512, 32] [1, 1] : tensor<512x512xf64> to tensor<512x32xf64>
  // CHECK-NEXT:      %extracted_slice_1 = tensor.extract_slice %arg2[%6, %7] [8, 32] [1, 1] : tensor<512x512xf64> to tensor<8x32xf64>
  // CHECK-NEXT:      %8 = demo.foo %extracted_slice, %extracted_slice_0, %extracted_slice_1 : tensor<8x512xf64>, tensor<512x32xf64>, tensor<8x32xf64> to tensor<8x32xf64>
  // CHECK-NEXT:      %9 = affine.apply #map(%arg3)
  // CHECK-NEXT:      %10 = affine.apply #map1(%arg4)
  // CHECK-NEXT:      scf.forall.in_parallel {
  // CHECK-NEXT:        tensor.parallel_insert_slice %8 into %arg5[%9, %10] [8, 32] [1, 1] : tensor<8x32xf64> into tensor<512x512xf64>
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:    return %1 : tensor<512x512xf64>
  // CHECK-NEXT:  }

module {
  func.func @bar(%arg0: tensor<512x512xf64>, %arg1: tensor<512x512xf64>, %output: tensor<512x512xf64>) -> tensor<512x512xf64> {
      %res = demo.foo %arg0, %arg1, %output : tensor<512x512xf64>, tensor<512x512xf64>, tensor<512x512xf64> to tensor<512x512xf64>
      return %res : tensor<512x512xf64>
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"demo.foo">):

    %loop, %tiled = transform.structured.tile_to_forall_op %arg0 tile_sizes [8, 32]
      : (!transform.op<"demo.foo">) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
