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

func.func @test_pad_non_const(%arg0: tensor<13x21x3xi8>, %arg1: tensor<i8>) -> tensor<13x21x3xi8> {
  %0 = "tosa.const"() {value = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
  // expected-error@+1 {{'tosa.pad' op pad_const of pad is not constant}}
  %1 = "tosa.pad"(%arg0, %0, %arg1) : (tensor<13x21x3xi8>, tensor<3x2xi32>, tensor<i8>) -> tensor<13x21x3xi8>
  return %1 : tensor<13x21x3xi8>
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

// -----
func.func @test_avg_pool2d_negative_kernel(%arg0: tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op kernel should be greater than one.}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 0, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8>
    return %0 : tensor<1x7x7x9xi8>
}
// -----
func.func @test_avg_pool2d_negative_stride(%arg0: tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op stride should be greater than one.}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: -1, 1>} : (tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8>
    return %0 : tensor<1x7x7x9xi8>
}
// -----
func.func @test_avg_pool2d_negative_pad(%arg0: tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op pad should be positive}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, -1, 0, 1>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8>
    return %0 : tensor<1x7x7x9xi8>
}
// -----
func.func @test_avg_pool2d_kernel_lessthan_pad(%arg0: tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op pad must be less than kernel size}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 3>, stride = array<i64: 1, 1>} : (tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8>
    return %0 : tensor<1x7x7x9xi8>
}
// -----
func.func @test_avg_pool2d_vert_stride_incorrect_mul(%arg0: tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op vertical stride is not in correct multiple.}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x7x7x9xi8>) -> tensor<1x7x7x9xi8>
    return %0 : tensor<1x7x7x9xi8>    
}
// -----
func.func @test_avg_pool2d_hor_stride_incorrect_mul(%arg0: tensor<1x6x6x9xi8>) -> tensor<1x7x4x9xi8> {
  // expected-error@+1 {{'tosa.avg_pool2d' op horizontal stride is not in correct multiple.}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x6x6x9xi8>) -> tensor<1x7x4x9xi8>
    return %0 : tensor<1x7x4x9xi8>
}
// -----
func.func @test_max_pool2d_hor_stride_incorrect_mul(%arg0: tensor<1x6x6x9xi8>) -> tensor<1x7x4x9xi8> {
  // expected-error@+1 {{'tosa.max_pool2d' op horizontal stride is not in correct multiple.}}
    %0 = "tosa.max_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x6x6x9xi8>) -> tensor<1x7x4x9xi8>
    return %0 : tensor<1x7x4x9xi8>
}
// -----
func.func @test_avg_pool2d_output_height_incorrect(%arg0: tensor<1x6x6x9xi8>) -> tensor<1x7x8x9xi8> {  
  // expected-error@+1 {{'tosa.avg_pool2d' op output height is not correct, should be 3.}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<1x6x6x9xi8>) -> tensor<1x7x8x9xi8>
    return %0 : tensor<1x7x8x9xi8>
}
// -----
func.func @test_avg_pool2d_output_width_incorrect(%arg0: tensor<1x6x6x9xi8>) -> tensor<1x3x8x9xi8> {  
  // expected-error@+1 {{'tosa.avg_pool2d' op output width is not correct, should be 3.}}
    %0 = "tosa.avg_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<1x6x6x9xi8>) -> tensor<1x3x8x9xi8>
    return %0 : tensor<1x3x8x9xi8>
}
// -----
func.func @test_max_pool2d_output_width_incorrect(%arg0: tensor<1x6x6x9xi8>) -> tensor<1x3x8x9xi8> {  
  // expected-error@+1 {{'tosa.max_pool2d' op output width is not correct, should be 3.}}
    %0 = "tosa.max_pool2d"(%arg0) {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 3, 3>} : (tensor<1x6x6x9xi8>) -> tensor<1x3x8x9xi8>
    return %0 : tensor<1x3x8x9xi8>
}
// -----
func.func @test_add_incompabitble_type(%arg0: tensor<13x21xf32>, %arg1: tensor<13x21xi8>) -> tensor<13x21xf32> {
  // expected-error@+1 {{'tosa.add' op requires the same element type for all operands and results}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21xf32>, tensor<13x21xi8>) -> tensor<13x21xf32>
  return %0 : tensor<13x21xf32>
}
// -----
func.func @test_add_incorrect_output(%arg0: tensor<13x21xf32>, %arg1: tensor<13x21xf32>) -> tensor<13x2xf32> {
  // expected-error@+1 {{'tosa.add' op result type '13x2' not broadcast compatible with broadcasted operands's shapes '13x21'}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21xf32>, tensor<13x21xf32>) -> tensor<13x2xf32>
  return %0 : tensor<13x2xf32>
}
// -----
func.func @test_add_incorrect_output2(%arg0: tensor<13x21xf32>, %arg1: tensor<2x13x21xf32>) -> tensor<2x13x21xf32> {
  // expected-error@+1 {{'tosa.add' op both operands must have same rank.}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21xf32>, tensor<2x13x21xf32>) -> tensor<2x13x21xf32>
  return %0 : tensor<2x13x21xf32>
}
// -----
func.func @test_const_incorrect_output(%arg0 : index) -> tensor<4xi32> {
  // expected-error@+1{{inferred shape of elements literal ([4]) does not match type ([3])}}
    %0 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<3xi32>} : () -> tensor<4xi32>
    return %0 : tensor<4xi32>
}
// -----
func.func @test_greater_equal_incompatible(%arg0: tensor<13x1x3x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  // expected-error@+1{{'tosa.greater_equal' op operands don't have broadcast-compatible shapes}}
  %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<13x1x3x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}
// -----
func.func @test_greater_equal_unequal_rank(%arg0: tensor<12x13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<?x13x21x3xi1> {
  // expected-error@+1{{'tosa.greater_equal' op both operands must have same rank.}}
  %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<12x13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<?x13x21x3xi1>
  return %0 : tensor<?x13x21x3xi1>
}
// -----
func.func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  // expected-error@+1{{'tosa.greater_equal' op result #0 must be tensor of 1-bit signless integer values, but got 'tensor<13x21x3xf32>'}}
  %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
// -----
func.func @test_mul_incompabitble_type(%arg0: tensor<13x21xf32>, %arg1: tensor<13x21xi8>) -> tensor<13x21xf32> {
  // expected-error@+1 {{'tosa.mul' op requires the same element type for all operands and results}}
  %0 = "tosa.mul"(%arg0, %arg1) { shift = 1 : i32 }: (tensor<13x21xf32>, tensor<13x21xi8>) -> tensor<13x21xf32>
  return %0 : tensor<13x21xf32>
}
// -----
func.func @test_mul_unequal_rank(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x21x3xf32>) -> tensor<?x?x?x?xf32> {
  // expected-error@+1{{'tosa.mul' op both operands must have same rank.}}
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xf32>, tensor<13x1x21x3xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// -----
func.func @test_add_unequal_rank(%arg0: tensor<3x4x8400xf32>, %arg1: tensor<8400xf32>) -> tensor<3x4x8400xf32> {
  // expected-error@+1{{'tosa.add' op both operands must have same rank.}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<3x4x8400xf32>, tensor<8400xf32>) -> tensor<3x4x8400xf32>  
  return %0 : tensor<3x4x8400xf32>
} 

// -----
func.func @test_mul_incompatible(%arg0: tensor<3x4x8400xf32>, %arg1: tensor<3x8400xf32>) -> tensor<1x4x8400xf32> {
  // expected-error@+1{{'tosa.mul' op operands don't have broadcast-compatible shapes}}
  %0 = "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<3x4x8400xf32>, tensor<3x8400xf32>) -> tensor<1x4x8400xf32>  
  return %0 : tensor<1x4x8400xf32>
} 
// -----
func.func @test_mul_need_shift(%arg0: tensor<3x4x8400xf32>, %arg1: tensor<3x4x8400xf32>) -> tensor<3x4x8400xf32> {
  // expected-error@+1{{'tosa.mul' op requires attribute 'shift'}}
  %0 = "tosa.mul"(%arg0, %arg1) : (tensor<3x4x8400xf32>, tensor<3x4x8400xf32>) -> tensor<3x4x8400xf32>  
  return %0 : tensor<3x4x8400xf32>
}
// -----
func.func @test_mul_nonzero_shift(%arg0: tensor<3x4x8400xf32>, %arg1: tensor<3x4x8400xf32>) -> tensor<3x4x8400xf32> {
  // expected-error@+1{{'tosa.mul' op shift attribute should be 0 for non integer input types}}
  %0 = "tosa.mul"(%arg0, %arg1) {shift = 3 : i32}: (tensor<3x4x8400xf32>, tensor<3x4x8400xf32>) -> tensor<3x4x8400xf32>  
  return %0 : tensor<3x4x8400xf32>
}

  // -----
func.func @test_select_unequal_rank_inputs(%arg0: tensor<2xi1>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // expected-error@+1{{'tosa.select' op both operands must have same rank.}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// -----
func.func @test_select_unequal_rank_inputs2(%arg0: tensor<1x2xi1>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // expected-error@+1{{'tosa.select' op both operands must have same rank.}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x2xi1>, tensor<1x3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// -----
func.func @test_select_unequal_rank_inputs3(%arg0: tensor<1x2xi1>, %arg1: tensor<3x2xf32>, %arg2: tensor<1x3x2xf32>) -> tensor<3x2xf32> {
  // expected-error@+1{{'tosa.select' op both operands must have same rank.}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x2xi1>, tensor<3x2xf32>, tensor<1x3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
} 
// -----
func.func @test_select_not_boardcastable_arg1(%arg0: tensor<2x2xi1>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // expected-error@+1{{'tosa.select' op operands don't have broadcast-compatible shapes}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<2x2xi1>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// -----
func.func @test_select_not_boardcastable_result(%arg0: tensor<1x2xi1>, %arg1: tensor<3x2xf32>, %arg2: tensor<3x2xf32>) -> tensor<4x2xf32> {
  // expected-error@+1{{'tosa.select' op result type '4x2' not broadcast compatible with broadcasted operands's shapes '3x2'}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x2xi1>, tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}
// -----
func.func @test_select_not_boardcastable_arg3(%arg0: tensor<1x2xi1>, %arg1: tensor<2x2xf32>, %arg2: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // expected-error@+1{{'tosa.select' op operands don't have broadcast-compatible shapes}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x2xi1>, tensor<2x2xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
// -----
func.func @test_select_incompatible_1(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>) -> tensor<13x21x3xi8> {
  // expected-error@+1{{'tosa.select' op inputs and result should be of same type.}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x1x1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi8>
  return %0 : tensor<13x21x3xi8>
}
// -----
func.func @test_select_incompatible_2(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xi8>) -> tensor<13x21x3xf32> {
  // expected-error@+1{{'tosa.select' op inputs should be of same type.}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x1x1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
// -----
func.func @test_select_incompatible_3(%arg0: tensor<1x1x1xi8>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xi8>) -> tensor<13x21x3xf32> {
  // expected-error@+1{{'tosa.select' op operand #0 must be tensor of 1-bit signless integer values, but got 'tensor<1x1x1xi8>'}}
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x1x1xi8>, tensor<13x21x3xf32>, tensor<13x21x3xi8>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}
// -----
func.func @test_transpose_incorrect_result_shape(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x20xf32> {  
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+2{{'tosa.transpose' op failed to infer returned types}}
  // expected-error@+1{{'tosa.transpose' op inferred type(s) 'tensor<3x13x21xf32>' are incompatible with return type(s) of operation 'tensor<3x13x20xf32>'}}  
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x20xf32>
  return %1 : tensor<3x13x20xf32>
}   
// -----
func.func @test_transpose_incorrect_result_rank(%arg0: tensor<13x21x3xf32>) -> tensor<3x13xf32> {  
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+2{{'tosa.transpose' op failed to infer returned types}}
  // expected-error@+1{{'tosa.transpose' op inferred type(s) 'tensor<3x13x21xf32>' are incompatible with return type(s) of operation 'tensor<3x13xf32>'}}  
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13xf32>
  return %1 : tensor<3x13xf32>
}
// -----
func.func @test_transpose_incorrect_result_type(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xi8> {  
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1{{'tosa.transpose' op inferred type(s) 'tensor<3x13x21xf32>' are incompatible with return type(s) of operation 'tensor<3x13x21xi8>'}}  
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xi8>
  return %1 : tensor<3x13x21xi8>
}
// -----
func.func @test_transpose_high_rank_perm(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21x4xf32> {
  %0 = "tosa.const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  // expected-error@+1 {{failed to infer returned types}}
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<4xi32>) -> tensor<3x13x21x4xf32>
  return %1 : tensor<3x13x21x4xf32>
}
// -----
// CHECK-LABEL: transpose
func.func @test_transpose_low_rank_perm(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21x4xf32> {
  %0 = "tosa.const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // expected-error@+1 {{failed to infer returned types}}
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<3x13x21x4xf32>
  return %1 : tensor<3x13x21x4xf32>
}
// -----
// CHECK-LABEL: transpose
func.func @test_transpose_result_high_rank(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21x4xf32> {
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  // expected-error@+2 {{failed to infer returned types}}
  // expected-error@+1 {{'tosa.transpose' op inferred type(s) 'tensor<3x13x21xf32>' are incompatible with return type(s) of operation 'tensor<3x13x21x4xf32>'}}
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21x4xf32>
  return %1 : tensor<3x13x21x4xf32>
}