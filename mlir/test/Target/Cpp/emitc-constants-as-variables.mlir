// RUN: mlir-translate -mlir-to-cpp -constants-as-variables=false %s | FileCheck %s -check-prefix=CPP-DEFAULT

func.func @test() {
  %start = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
  %stop = "emitc.constant"() <{value = 10 : index}> : () -> !emitc.size_t
  %step = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t

  emitc.for %iter = %start to %stop step %step {
    emitc.yield
  }

  return
}
// CPP-DEFAULT: void test() {
// CPP-DEFAULT-NEXT:   for (size_t v1 = ((size_t) 0); v1 < ((size_t) 10); v1 += ((size_t) 1)) {
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   return;
// CPP-DEFAULT-NEXT: }

// -----

func.func @test_subscript(%arg0: !emitc.array<4xf32>) -> (!emitc.lvalue<f32>) {
  %c0 = "emitc.constant"() <{value = 0 : index}> : () -> !emitc.size_t
  %0 = emitc.subscript %arg0[%c0] : (!emitc.array<4xf32>, !emitc.size_t) -> !emitc.lvalue<f32>
  return %0 : !emitc.lvalue<f32>
}
// CPP-DEFAULT: float test_subscript(float v1[4]) {
// CPP-DEFAULT-NEXT:  return v1[0];
// CPP-DEFAULT-NEXT: }