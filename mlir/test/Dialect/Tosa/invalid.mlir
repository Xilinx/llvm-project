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

func.func @test_concat_output_shape_mismatch(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> tensor<2x2xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<2x2xf32>}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

func.func @test_concat_output_rank_mismatch(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<?x?x?xf32>}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

func.func @test_concat_input_rank_mismatch(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2x2xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{'tosa.concat' op rank of input 'tensor<2x2x2xf32>' does not match other input rank(s) (2)}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x2x2xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_concat_axis_out_of_range(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{'tosa.concat' op axis must be in range 0 to 1}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = -1 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @test_concat_axis_out_of_range(%arg0 : tensor<10x11x12xf32>, %arg1 : tensor<10x11x21xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+1 {{'tosa.concat' op axis must be in range 0 to 2}}
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 3 : i64} : (tensor<10x11x12xf32>, tensor<10x11x21xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3x2xi32>) -> tensor<13x21x3xf32> {
  // expected-error@+1 {{'tosa.pad' op padding of pad is not constant}}
  %0 = "tosa.pad"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----

func.func @test_pad_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<i8>) -> tensor<?x?x?xi8> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // expected-error@+1 {{'tosa.pad' op pad_const of pad is not constant}}
  %1 = "tosa.pad"(%arg0, %0, %arg1) : (tensor<13x21x3xi8>, tensor<3x2xi32>, tensor<i8>) -> tensor<?x?x?xi8>
  return %1 : tensor<?x?x?xi8>
}

// -----

func.func @test_pad_output_shape_mismatch(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.const"() {value = dense<[[1, 1], [1, 1], [1, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // expected-error@+2 {{'tosa.pad' op failed to infer returned types}}
  // expected-error@+1 {{'tosa.pad' op inferred type(s) 'tensor<15x23x5xf32>' are incompatible with return type(s) of operation 'tensor<13x21x3xf32>}}
  %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %1 : tensor<13x21x3xf32>
}

// -----

func.func @test_pad_type_mismatch(%arg0: tensor<13x21x3xf32>) -> tensor<15x23x5xi32> {
  %0 = "tosa.const"() {value = dense<[[1, 1], [1, 1], [1, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // expected-error@+2 {{'tosa.pad' op failed to infer returned types}}
  // expected-error@+1 {{'tosa.pad' op inferred type(s) 'tensor<15x23x5xf32>' are incompatible with return type(s) of operation 'tensor<15x23x5xi32>}}
  %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<15x23x5xi32>
  return %1 : tensor<15x23x5xi32>
}

// -----

func.func @test_pad_incorret_padding_rank(%arg0: tensor<13x21xf32>) -> tensor<13x21xf32> {
  %0 = "tosa.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error@+1 {{'tosa.pad' op paddings must be a tensor of rank 2}}
  %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21xf32>, tensor<2xi32>) -> tensor<13x21xf32>
  return %1 : tensor<13x21xf32>
}

// -----

func.func @test_pad_incorret_padding_shape(%arg0: tensor<13x21xf32>) -> tensor<13x21xf32> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1], [1, 1]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
  // expected-error@+1 {{'tosa.pad' op paddings must be a tensor of shape [2, 2]}}
  %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21xf32>, tensor<4x2xi32>) -> tensor<13x21xf32>
  return %1 : tensor<13x21xf32>
}

// -----

func.func @test_pad_incorret_padding_shape(%arg0: tensor<13x21xf32>) -> tensor<13x21xf32> {
  %0 = "tosa.const"() {value = dense<[[0, 0, 0, 1], [0, 1, 1, 1]]> : tensor<2x4xi32>} : () -> tensor<2x4xi32>
  // expected-error@+1 {{'tosa.pad' op paddings must be a tensor of shape [2, 2]}}
  %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21xf32>, tensor<2x4xi32>) -> tensor<13x21xf32>
  return %1 : tensor<13x21xf32>
}

// -----

func.func @test_pad_negative_padding(%arg0: tensor<13x21xf32>) -> tensor<?x?xf32> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, -1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  // expected-error@+1 {{'tosa.pad' op number of pad elements must be positive}}
  %1 = "tosa.pad"(%arg0, %0) : (tensor<13x21xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

func.func @test_sigmoid_type_mismatch(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xi8> {
  // expected-error@+1 {{'tosa.sigmoid' op requires the same element type for all operands and results}}
  %0 = "tosa.sigmoid"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x4xi8>
  return %0 : tensor<13x21x4xi8>
}

// -----

func.func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x4xf32> {
  // expected-error@+1 {{'tosa.sigmoid' op input type 'tensor<13x21x3xf32>' and output type 'tensor<13x21x4xf32>' are not compatible}}
  %0 = "tosa.sigmoid"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x4xf32>
  return %0 : tensor<13x21x4xf32>
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
