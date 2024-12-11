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
// CPP-DEFAULT-LABEL: void test() {
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
// CPP-DEFAULT-LABEL: float test_subscript(float v1[4]) {
// CPP-DEFAULT-NEXT:  return v1[0];
// CPP-DEFAULT-NEXT: }

// -----

func.func @emitc_switch_ui64() {
  %0 = "emitc.constant"(){value = 1 : ui64} : () -> ui64

  emitc.switch %0 : ui64
  case 2 {
    %1 = emitc.call_opaque "func_b" () : () -> i32
    emitc.yield
  }
  case 5 {
    %2 = emitc.call_opaque "func_a" () : () -> i32
    emitc.yield
  }
  default {
    emitc.call_opaque "func2" (%0) : (ui64) -> ()
    emitc.yield
  }
  return
}
// CPP-DEFAULT-LABEL: void emitc_switch_ui64() {
// CPP-DEFAULT:   switch (1) {
// CPP-DEFAULT-NEXT:   case 2: {
// CPP-DEFAULT-NEXT:     int32_t v1 = func_b();
// CPP-DEFAULT-NEXT:     break;
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   case 5: {
// CPP-DEFAULT-NEXT:     int32_t v2 = func_a();
// CPP-DEFAULT-NEXT:     break;
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   default: {
// CPP-DEFAULT-NEXT:     func2(((uint64_t) 1));
// CPP-DEFAULT-NEXT:     break;
// CPP-DEFAULT-NEXT:   }
// CPP-DEFAULT-NEXT:   return;
// CPP-DEFAULT-NEXT: }
