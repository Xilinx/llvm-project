// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT

// CPP-DEFAULT: void basic(int32_t v1, float v2) {
func.func @basic(%arg0: i32, %arg1: f32) {
  emitc.call_opaque "kernel1"()  : () -> ()
// CPP-DEFAULT:   kernel1();
  %0 = emitc.call_opaque "kernel2"(%arg0)  : (i32) -> i16
// CPP-DEFAULT:   int16_t v3 = kernel2(v1);
  emitc.call_opaque "kernel3"(%arg0, %arg1)  : (i32, f32) -> ()
// CPP-DEFAULT:   kernel3(v1, v2);
  emitc.call_opaque "kernel4"(%arg0, %arg1)  {template_arg_names = ["N", "P"], template_args = [42 : i32, 56]} : (i32, f32) -> ()
// CPP-DEFAULT:   kernel4</*N=*/42, /*P=*/56>(v1, v2);
  emitc.call_opaque "kernel4"(%arg0, %arg1)  {template_arg_names = ["N"], template_args = [#emitc.opaque<"42">]} : (i32, f32) -> ()
// CPP-DEFAULT:   kernel4</*N=*/42>(v1, v2);
  return
// CPP-DEFAULT:   return;
}
// CPP-DEFAULT: }


