// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL: void opaque_types() {
func.func @opaque_types() {
  // CHECK-NEXT: f<int>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"int">>]} : () -> ()
  // CHECK-NEXT: f<byte>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"byte">>]} : () -> ()
  // CHECK-NEXT: f<unsigned>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"unsigned">>]} : () -> ()
  // CHECK-NEXT: f<status_t>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"status_t">>]} : () -> ()
  // CHECK-NEXT: f<std::vector<std::string>>();
  emitc.call_opaque "f"() {template_args = [!emitc.opaque<"std::vector<std::string>">]} : () -> ()
  // CHECK:  f<float>()
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"{}", f32>>]} : () -> ()
  // CHECK: f<int16_t {>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"{} {{", si16>>]} : () -> ()
  // CHECK: f<int8_t {>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"{} {", i8>>]} : () -> ()
  // CHECK: f<status_t>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"{}", !emitc<opaque<"status_t">> >>]} : () -> ()
  // CHECK: f<top<nested<float>,int32_t>>();
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"top<{},{}>", !emitc<opaque<"nested<{}>", f32>>, i32>>]} : () -> ()

  return
}

// CHECK-LABEL: void ptr_types() {
func.func @ptr_types() {
  // CHECK-NEXT: f<int32_t*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<i32>]} : () -> ()
  // CHECK-NEXT: f<int64_t*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<i64>]} : () -> ()
  // CHECK-NEXT: f<_Float16*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<f16>]} : () -> ()
  // CHECK-NEXT: f<__bf16*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<bf16>]} : () -> ()
  // CHECK-NEXT: f<float*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<f32>]} : () -> ()
  // CHECK-NEXT: f<double*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<f64>]} : () -> ()
  // CHECK-NEXT: int32_t* [[V0:[^ ]*]] = f();
  %0 = emitc.call_opaque "f"() : () -> (!emitc.ptr<i32>)
  // CHECK-NEXT: int32_t** [[V1:[^ ]*]] = f([[V0:[^ ]*]]);
  %1 = emitc.call_opaque "f"(%0) : (!emitc.ptr<i32>) -> (!emitc.ptr<!emitc.ptr<i32>>)
  // CHECK-NEXT: f<int*>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<!emitc.opaque<"int">>]} : () -> ()

  return
}

// CHECK-LABEL: void size_types() {
func.func @size_types() {
  // CHECK-NEXT: f<ssize_t>();
  emitc.call_opaque "f"() {template_args = [!emitc.ssize_t]} : () -> ()
  // CHECK-NEXT: f<size_t>();
  emitc.call_opaque "f"() {template_args = [!emitc.size_t]} : () -> ()
  // CHECK-NEXT: f<ptrdiff_t>();
  emitc.call_opaque "f"() {template_args = [!emitc.ptrdiff_t]} : () -> ()

  return
}
