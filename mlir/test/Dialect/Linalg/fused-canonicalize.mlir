// RUN: mlir-opt -split-input-file --canonicalize %s

// CHECK-LABEL: func.func @fused_on_tensors(
// CHECK: %[[arg0:.+]]: tensor<1x3x108x108xf32>
func.func @fused_on_tensors(%arg0: tensor<1x3x108x108xf32>) -> tensor<1x16x106x106xf32> {
    // CHECK: %[[result:.+]] linalg.subgraph (%[[:.+]] = %[[arg0]]
    %result = linalg.subgraph (%ifm = %arg0 : tensor<1x3x108x108xf32>) {
        %weights = arith.constant dense<5.000000e-01> : tensor<16x3x3x3xf32>
        %init = arith.constant dense<0.000000e00> : tensor<1x16x106x106xf32>
        %conv = linalg.conv_2d_nchw_fchw
            {dilation = dense<1> : tensor<2xi64>, stride = dense<1> : tensor<2xi64>}
            ins(%ifm, %weights : tensor<1x3x108x108xf32>, tensor<16x3x3x3xf32>)
            outs(%init : tensor<1x16x106x106xf32>)
            -> tensor<1x16x106x106xf32>
        %act = linalg.relu_2d_nchw
            ins(%conv : tensor<1x16x106x106xf32>)
            outs(%conv : tensor<1x16x106x106xf32>)
            -> tensor<1x16x106x106xf32>
        linalg.yield %act : tensor<1x16x106x106xf32>
    // CHECK: } -> tensor<1x16x106x106xf32>
    } -> tensor<1x16x106x106xf32>
    // CHECK: return %[[result]] : tensor<1x16x106x106xf32>
    return %result : tensor<1x16x106x106xf32>
}

// -----

// CHECK-LABEL: func.func @drop_unused_captures(
// CHECK: %[[arg0:.+]]: tensor<3x3xf32>
func.func @drop_unused_captures(%arg0: tensor<3x3xf32>, %arg1: f32) -> tensor<3x3xf32> {
    // CHECK: linalg.subgraph (%[[:.+]] = %[[arg0]] : tensor<3x3xf32>)
    %result = linalg.subgraph (%0 = %arg0 : tensor<3x3xf32>, %1 = %arg1 : f32) {
        linalg.yield %0 : tensor<3x3xf32>
    } -> tensor<3x3xf32>
    return %result : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: func.func @drop_unused_result(
// CHECK: %[[arg0:.+]]: tensor<3x3xf32>
func.func @drop_unused_result(%arg0: memref<3x3xf32>) -> memref<3x3xf32> {
    // CHECK: linalg.subgraph (%[[dest:.+]] = %[[arg0]] : memref<3x3xf32>)
    %result = linalg.subgraph (%dest = %arg0 : memref<3x3xf32>) {
        %cst0 = arith.constant 0.000000e00 : f32
        // CHECK: linalg.fill
        // CHECK: outs(%[[dest]] : memref<3x3xf32>)
        linalg.fill ins(%cst0 : f32) outs(%dest : memref<3x3xf32>)
        // CHECK: linalg.yield
        // CHECK-NOT: %[[:.+]]
        %0 = bufferization.to_tensor %dest : memref<3x3xf32>
        linalg.yield %0 : tensor<3x3xf32>
    // CHECK: }
    // CHECK-NOT: ->
    } -> tensor<3x3xf32>
    return %arg0 : memref<3x3xf32>
}

// -----

// CHECK-LABEL: func.func @erase_dead(
// CHECK: %[[arg0:.+]]: tensor<3x3xf32>
func.func @erase_dead(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    %arg1 = bufferization.to_memref %arg0 : memref<3x3xf32>
    // Trivially dead
    linalg.subgraph (%0 = %arg0 : tensor<3x3xf32>, %1 = %arg1 : memref<3x3xf32>) {
        linalg.yield
    }
    // No side-effects
    %0 = linalg.subgraph () {
        %0 = tensor.empty() : tensor<3x3xf32>
        linalg.yield %0 : tensor<3x3xf32>
    } -> tensor<3x3xf32>
    // Read-only side-effects
    linalg.subgraph (%0 = %arg1 : memref<3x3xf32>) {
        %1 = memref.alloc() : memref<3x3xf32>
        memref.copy %0, %1 : memref<3x3xf32> to memref<3x3xf32>
        linalg.yield
    }
    // Trivially empty
    %1 = linalg.subgraph (%0 = %arg0 : tensor<3x3xf32>) {
        linalg.yield %0 : tensor<3x3xf32>
    } -> tensor<3x3xf32>
    // CHECK: return %[[arg0]]
    return %1 : tensor<3x3xf32>
}
