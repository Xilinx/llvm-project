// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @no_block() {
    // expected-error @+1 {{region with 1 blocks}}
    linalg.subgraph () {
    }
    return
}

// -----

func.func @no_terminator() {
    // expected-error @+1 {{non-empty block}}
    linalg.subgraph () {
        ^bb0():
    }
    return
}

// -----

func.func @wrong_terminator_args() {
    linalg.subgraph () {
        %0 = arith.constant 0.0 : f32
        // expected-error @+1 {{does not match number of results}}
        linalg.yield %0 : f32
    }
    return
}

// -----

func.func @wrong_terminator_type() {
    linalg.subgraph () {
        %0 = arith.constant dense<0.0> : tensor<1xf32>
        // expected-error @+1 {{does not match result type}}
        linalg.yield %0 : tensor<1xf32>
    } -> tensor<1xf64>
    return
}

// -----

func.func @non_tensor_result() {
    // expected-error @+1 {{must be ranked tensor}}
    linalg.subgraph () {
        %0 = memref.alloc() : memref<1xf32>
        linalg.yield %0 : memref<1xf32>
    } -> memref<1xf32>
    return
}
