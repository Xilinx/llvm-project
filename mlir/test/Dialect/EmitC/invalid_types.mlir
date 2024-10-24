// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @illegal_opaque_type_1() {
    // expected-error @+1 {{expected non empty string in !emitc.opaque type}}
    %1 = "emitc.variable"(){value = "42" : !emitc.opaque<"">} : () -> !emitc.opaque<"mytype">
}

// -----

func.func @illegal_opaque_type_2() {
    // expected-error @+1 {{pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead}}
    %1 = "emitc.variable"(){value = "nullptr" : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
}

// -----

func.func @illegal_array_missing_spec(
    // expected-error @+1 {{expected non-function type}}
    %arg0: !emitc.array<>) {
}

// -----

func.func @illegal_array_missing_shape(
    // expected-error @+1 {{shape must not be empty}}
    %arg9: !emitc.array<i32>) {
}

// -----

func.func @illegal_array_missing_x(
    // expected-error @+1 {{expected 'x' in dimension list}}
    %arg0: !emitc.array<10>
) {
}

// -----

func.func @illegal_array_missing_type(
    // expected-error @+1 {{expected non-function type}}
    %arg0: !emitc.array<10x>
) {
}

// -----

func.func @illegal_array_dynamic_shape(
    // expected-error @+1 {{expected static shape}}
    %arg0: !emitc.array<10x?xi32>
) {
}

// -----

func.func @illegal_array_unranked(
    // expected-error @+1 {{expected non-function type}}
    %arg0: !emitc.array<*xi32>
) {
}

// -----

func.func @illegal_array_with_array_element_type(
    // expected-error @+1 {{invalid array element type}}
    %arg0: !emitc.array<4x!emitc.array<4xi32>>
) {
}

// -----

func.func @illegal_array_with_tensor_element_type(
    // expected-error @+1 {{invalid array element type}}
    %arg0: !emitc.array<4xtensor<4xi32>>
) {
}

// -----

func.func @illegal_integer_type(%arg0: i11, %arg1: i11) -> i11 {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'i11'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (i11, i11) -> i11
    return
}

// -----

func.func @illegal_f8E4M3B11FNUZ_type(%arg0: f8E4M3B11FNUZ, %arg1: f8E4M3B11FNUZ) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f8E4M3B11FNUZ'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f8E4M3B11FNUZ, f8E4M3B11FNUZ) -> f8E4M3B11FNUZ
    return
}

// -----

func.func @illegal_f8E4M3FN_type(%arg0: f8E4M3FN, %arg1: f8E4M3FN) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f8E4M3FN'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f8E4M3FN, f8E4M3FN) -> f8E4M3FN
    return
}

// -----

func.func @illegal_f8E4M3FNUZ_type(%arg0: f8E4M3FNUZ, %arg1: f8E4M3FNUZ) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f8E4M3FNUZ'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f8E4M3FNUZ, f8E4M3FNUZ) -> f8E4M3FNUZ
    return
}

// -----

func.func @illegal_f8E5M2_type(%arg0: f8E5M2, %arg1: f8E5M2) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f8E5M2'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f8E5M2, f8E5M2) -> f8E5M2
    return
}

// -----

func.func @illegal_f8E5M2FNUZ_type(%arg0: f8E5M2FNUZ, %arg1: f8E5M2FNUZ) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f8E5M2FNUZ'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f8E5M2FNUZ, f8E5M2FNUZ) -> f8E5M2FNUZ
    return
}

// -----

func.func @illegal_f16_type(%arg0: f16, %arg1: f16) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f16'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f16, f16) -> f16
    return
}

// -----

func.func @illegal_bf16_type(%arg0: bf16, %arg1: bf16) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'bf16'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (bf16, bf16) -> bf16
    return
}

// -----

func.func @illegal_f80_type(%arg0: f80, %arg1: f80) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f80'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f80, f80) -> f80
    return
}

// -----

func.func @illegal_f128_type(%arg0: f128, %arg1: f128) {
    // expected-error @+1 {{'emitc.mul' op operand #0 must be floating-point type supported by EmitC or integer, index or opaque type supported by EmitC, but got 'f128'}}
    %mul = "emitc.mul" (%arg0, %arg1) : (f128, f128) -> f128
    return
}

// -----

func.func @illegal_pointee_type() {
    // expected-error @+1 {{'emitc.variable' op result #0 must be type supported by EmitC, but got '!emitc.ptr<i11>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> !emitc.ptr<i11>
    return
}

// -----

func.func @illegal_non_static_tensor_shape_type() {
    // expected-error @+1 {{'emitc.variable' op result #0 must be type supported by EmitC, but got 'tensor<?xf32>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> tensor<?xf32>
    return
}

// -----

func.func @illegal_tensor_array_element_type() {
    // expected-error @+1 {{'emitc.variable' op result #0 must be type supported by EmitC, but got 'tensor<!emitc.array<9xi16>>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> tensor<!emitc.array<9xi16>>
    return
}

// -----

func.func @illegal_tensor_integer_element_type() {
    // expected-error @+1 {{'emitc.variable' op result #0 must be type supported by EmitC, but got 'tensor<9xi11>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> tensor<9xi11>
    return
}

// -----

func.func @illegal_tuple_array_element_type() {
    // expected-error @+1 {{'emitc.variable' op result #0 must be type supported by EmitC, but got 'tuple<!emitc.array<9xf32>, f32>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> tuple<!emitc.array<9xf32>, f32>
    return
}

// -----

func.func @illegal_tuple_float_element_type() {
    // expected-error @+1 {{'emitc.variable' op result #0 must be type supported by EmitC, but got 'tuple<i32, f80>'}}
    %v = "emitc.variable"(){value = #emitc.opaque<"">} : () -> tuple<i32, f80>
    return
}
