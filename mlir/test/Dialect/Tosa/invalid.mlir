// RUN: mlir-opt %s -split-input-file -verify-diagnostics --tosa-validate=strict-op-spec-alignment


func.func @test_conv2d(%arg0: tensor<1x29x29x4xf32>, %arg1: tensor<16x3x3x4xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{expect both input and weight to be float or not together, got 'f32' and 'i8'}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x29x4xf32>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_conv2d(%arg0: tensor<*xi8>, %arg1: tensor<16x3x3x4xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{expect a ranked tensor for input, got <block argument> of type 'tensor<*xi8>' at index: 0}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<*xi8>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_conv2d(%arg0: tensor<1x29x29x4xi8>, %arg1: tensor<*xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{expect a ranked tensor for weight, got <block argument> of type 'tensor<*xi8>' at index: 1}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x29x4xi8>, tensor<*xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}


// -----

func.func @test_conv2d(%arg0: tensor<1x29x29x4xi8>, %arg1: tensor<16x3x3x4xi8>, %arg2: tensor<16xi8>) -> tensor<1x27x27x16xi8> {
  // expected-error@+1 {{'tosa.conv2d' op quantizationattr is required for quantized type, and not allowed for float type}}
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
           : (tensor<1x29x29x4xi8>, tensor<16x3x3x4xi8>, tensor<16xi8>) -> tensor<1x27x27x16xi8>
  return %0 : tensor<1x27x27x16xi8>
}

// -----

func.func @test_concat(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x?xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{Cannot concat tensors with different sizes on the non-axis dimension 1}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_concat_element_type_mismatch(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x?xi8> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.concat' op inferred type(s) 'tensor<3x2xf32>' are incompatible with return type(s) of operation 'tensor<?x?xi8>}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xi8>
  return %0 : tensor<?x?xi8>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3x2xi32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.pad' op padding of pad is not constant}}
  %0 = "tosa.pad"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<i8>) -> tensor<13x22x4xi8> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // expected-error@+1 {{'tosa.pad' op pad_const of pad is not constant}}
  %1 = "tosa.pad"(%arg0, %0, %arg1) : (tensor<13x21x3xi8>, tensor<3x2xi32>, tensor<i8>) -> tensor<13x22x4xi8>
  return %1 : tensor<13x22x4xi8>
}

// -----

func.func @test_pad_const_not_matched_element_type(%arg0: tensor<13x21x3xi8>) -> tensor<13x22x4xi8> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // expected-error@+1 {{'tosa.pad' op pad const has element type ('f32') while the input tensor has element type('i8')}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<13x21x3xi8>, tensor<3x2xi32>, tensor<f32>) -> tensor<13x22x4xi8>
  return %2 : tensor<13x22x4xi8>
}

// -----

func.func @test_pad_must_have_same_rank(%arg0: tensor<13x21x3xi8>) -> tensor<13x22x4x4xi8> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  %1 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<i8>
  // expected-error@+1 {{'tosa.pad' op input type ('tensor<13x21x3xi8>') must have the same rank as the output type ('tensor<13x22x4x4xi8>')}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<13x21x3xi8>, tensor<3x2xi32>, tensor<i8>) -> tensor<13x22x4x4xi8>
  return %2 : tensor<13x22x4x4xi8>
}

// -----

func.func @test_pad_padding_shape(%arg0: tensor<13x21x3xi8>) -> tensor<13x22x4xi8> {
  %0 = "tosa.const"() {value = dense<[[[0, 0], [0, 1], [0, 1]]]> : tensor<1x3x2xi32>} : () -> tensor<1x3x2xi32>
  %1 = "tosa.const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // expected-error@+1 {{'tosa.pad' op padding shape must be in the form of Nx2 where N is the rank of input tensor but got ('tensor<1x3x2xi32>')}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<13x21x3xi8>, tensor<1x3x2xi32>, tensor<f32>) -> tensor<13x22x4xi8>
  return %2 : tensor<13x22x4xi8>
}

// -----

func.func @test_pad_if_padding_has_2_values(%arg0: tensor<2x2xi32>) -> tensor<5x5xi32> { 
  %0 = "tosa.const"() {value = dense<[[1, 2, 3], [1, 2, 3]]> : tensor<2x3xi64>} : () -> tensor<2x3xi64>
  %1 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tosa.pad' op must only have a before-padding and an after-padding for each dimension of the input tensor}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<2x2xi32>, tensor<2x3xi64>, tensor<i32>) -> tensor<5x5xi32>
  return %2 : tensor<5x5xi32>
}

// -----

func.func @test_pad_if_padding_has_the_same_rank(%arg0: tensor<2x2xi32>) -> tensor<5x5xi32> { 
  %0 = "tosa.const"() {value = dense<[[1, 2], [1, 2], [1, 2]]> : tensor<3x2xi64>} : () -> tensor<3x2xi64>
  %1 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tosa.pad' op padding array has (3) pad pairs while the rank of input tensor is 2}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<2x2xi32>, tensor<3x2xi64>, tensor<i32>) -> tensor<5x5xi32>
  return %2 : tensor<5x5xi32>
}

// -----

func.func @test_if_input_output_padding_are_matching_dim_1(%arg0: tensor<2x2xi32>) -> tensor<5x5xi32> { 
  %0 = "tosa.const"() {value = dense<[[1, 3], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %1 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tosa.pad' op output shape ('tensor<5x5xi32>') doesn't match with the input shape ('tensor<2x2xi32>') and the paddings (dense<[[1, 3], [1, 2]]> : tensor<2x2xi64>)}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<2x2xi32>, tensor<2x2xi64>, tensor<i32>) -> tensor<5x5xi32>
  return %2 : tensor<5x5xi32>
}

// -----

func.func @test_if_input_output_padding_are_matching_dim_2(%arg0: tensor<2x2xi32>) -> tensor<5x5xi32> { 
  %0 = "tosa.const"() {value = dense<[[1, 2], [1, 3]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %1 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error@+1 {{'tosa.pad' op output shape ('tensor<5x5xi32>') doesn't match with the input shape ('tensor<2x2xi32>') and the paddings (dense<[[1, 2], [1, 3]]> : tensor<2x2xi64>)}}
  %2 = "tosa.pad"(%arg0, %0, %1) : (tensor<2x2xi32>, tensor<2x2xi64>, tensor<i32>) -> tensor<5x5xi32>
  return %2 : tensor<5x5xi32>
}

// -----

func.func @test_transpose_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3xi32>) -> tensor<3x13x21xf32> {
  // expected-error@+1 {{'tosa.transpose' op perms of transpose is not constant}}
  %0 = "tosa.transpose"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %0 : tensor<3x13x21xf32>
}

// -----

func.func @test_fully_connected_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<273x2xf32> {
  %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
  %1 = "tosa.reshape"(%arg0) {new_shape = array<i64: 273, 3>} : (tensor<13x21x3xf32>) -> tensor<273x3xf32>
  // expected-error@+1 {{'tosa.fully_connected' op weight of fully_connected is not constant}}
  %2 = "tosa.fully_connected"(%1, %arg1, %0) : (tensor<273x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<273x2xf32>
  return %2 : tensor<273x2xf32>
}

// -----

func.func @test_fully_connected_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<2xf32>) -> tensor<273x2xf32> {
  %0 = "tosa.const"() {value = dense<[[-0.613216758, -0.63714242, -0.73500061], [0.180762768, 0.773053169, -0.933686495]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "tosa.reshape"(%arg0) {new_shape = array<i64: 273, 3>} : (tensor<13x21x3xf32>) -> tensor<273x3xf32>
  // expected-error@+1 {{'tosa.fully_connected' op bias of fully_connected is not constant}}
  %2 = "tosa.fully_connected"(%1, %0, %arg1) : (tensor<273x3xf32>, tensor<2x3xf32>, tensor<2xf32>) -> tensor<273x2xf32>
  return %2 : tensor<273x2xf32>
}

// -----

func.func @test_reduce_sum_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_sum' op inferred type(s) 'tensor<1x3x4x5xf32>' are incompatible with return type(s) of operation 'tensor<1x3x4x5xi32>'}}
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<2x3x4x5xf32>) -> tensor<1x3x4x5xi32>
  return
}

// -----

func.func @test_reduce_max_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_max' op inferred type(s) 'tensor<2x3x4x1xf32>' are incompatible with return type(s) of operation 'tensor<2x3x4x1xi32>'}}
  %0 = "tosa.reduce_max"(%arg0) {axis = 3 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x1xi32>
  return
}

// -----

func.func @test_reduce_min_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_min' op inferred type(s) 'tensor<2x1x4x5xf32>' are incompatible with return type(s) of operation 'tensor<2x1x4x5xi32>'}}
  %0 = "tosa.reduce_min"(%arg0) {axis = 1 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x1x4x5xi32>
  return
}

// -----

func.func @test_reduce_prod_type_mismatch(%arg0 : tensor<2x3x4x5xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reduce_prod' op inferred type(s) 'tensor<2x1x4x5xf32>' are incompatible with return type(s) of operation 'tensor<2x3x4x5xf32>'}}
  %0 = "tosa.reduce_prod"(%arg0) {axis = 1 : i64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  return
}

// -----

func.func @test_reshape_type_mismatch(%arg0 : tensor<13x21x3xf32>) -> () {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.reshape' op inferred type(s) 'tensor<13x21x3x1xf32>' are incompatible with return type(s) of operation 'tensor<13x21x3x1xi32>'}}
  %0 = "tosa.reshape"(%arg0) {new_shape = array<i64: 13, 21, 3, 1>} : (tensor<13x21x3xf32>) -> tensor<13x21x3x1xi32>
  return
}

// -----

func.func @test_const_attribute_type_mismatch() -> tensor<100x100xf32> {
  // expected-error@+1 {{'tosa.const' op failed to verify that all of {value, output} have same shape}}
  %0 = "tosa.const"() {value = dense<0.000000e+00> : tensor<1x1xf32>} : () -> tensor<100x100xf32>
  return %0 : tensor<100x100xf32>
}
