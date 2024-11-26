// RUN: mlir-translate -mlir-to-cpp -propagate-constants %s | FileCheck %s -check-prefix=CPP-DEFAULT

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