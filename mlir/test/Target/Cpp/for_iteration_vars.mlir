// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s -check-prefix=CPP-DEFAULT

func.func @test_for_siblings(%arg0 : !emitc.size_t, %arg1 : !emitc.size_t, %arg2 : !emitc.size_t) {
  %lb = emitc.expression : !emitc.size_t {
    %a = emitc.add %arg0, %arg1 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    emitc.yield %a : !emitc.size_t
  }
  %ub = emitc.expression : !emitc.size_t {
    %a = emitc.mul %arg1, %arg2 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    emitc.yield %a : !emitc.size_t
  }
  %step = emitc.expression : !emitc.size_t {
    %a = emitc.div %arg0, %arg2 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    emitc.yield %a : !emitc.size_t
  }
  emitc.for %i0 = %lb to %ub step %step {
    emitc.for %i1 = %lb to %ub step %step {
      %0 = emitc.call_opaque "f"() : () -> i32
    }
  }
  emitc.for %ki2 = %lb to %ub step %step {
    emitc.for %i3 = %lb to %ub step %step {
      %1 = emitc.call_opaque "f"() : () -> i32
    }
  }
  return
}
// CPP-DEFAULT: void test_for_siblings(size_t [[V1:[^ ]*]], size_t [[V2:[^ ]*]], size_t [[V3:[^ ]*]]) {
// CPP-DEFAULT-NEXT: size_t [[V4:[^ ]*]] = [[V1]] + [[V2]];
// CPP-DEFAULT-NEXT: size_t [[V5:[^ ]*]] = [[V2]] * [[V3]];
// CPP-DEFAULT-NEXT: size_t [[V6:[^ ]*]] = [[V1]] / [[V3]];
// CPP-DEFAULT-NEXT: for (size_t [[ITER0:i]] = [[V4]]; [[ITER0]] < [[V5]]; [[ITER0]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITER1:j]] = [[V4]]; [[ITER1]] < [[V5]]; [[ITER1]] += [[V6]]) {
// CPP-DEFAULT-NEXT: int32_t [[V7:[^ ]*]] = f();
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: for (size_t [[ITER2:k]] = [[V4]]; [[ITER2]] < [[V5]]; [[ITER2]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITER3:l]] = [[V4]]; [[ITER3]] < [[V5]]; [[ITER3]] += [[V6]]) {
// CPP-DEFAULT-NEXT: int32_t [[V8:[^ ]*]] = f();
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;

func.func @test_for_nesting(%arg0 : !emitc.size_t, %arg1 : !emitc.size_t, %arg2 : !emitc.size_t) {
  %lb = emitc.expression : !emitc.size_t {
    %a = emitc.add %arg0, %arg1 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    emitc.yield %a : !emitc.size_t
  }
  %ub = emitc.expression : !emitc.size_t {
    %a = emitc.mul %arg1, %arg2 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    emitc.yield %a : !emitc.size_t
  }
  %step = emitc.expression : !emitc.size_t {
    %a = emitc.div %arg0, %arg2 : (!emitc.size_t, !emitc.size_t) -> !emitc.size_t
    emitc.yield %a : !emitc.size_t
  }
  emitc.for %i0 = %lb to %ub step %step {
    emitc.for %i1 = %lb to %ub step %step {
      emitc.for %i2 = %lb to %ub step %step {
        emitc.for %i3 = %lb to %ub step %step {
          emitc.for %i4 = %lb to %ub step %step {
            emitc.for %i5 = %lb to %ub step %step {
              emitc.for %i6 = %lb to %ub step %step {
                emitc.for %i7 = %lb to %ub step %step {
                  emitc.for %i8 = %lb to %ub step %step {
                    emitc.for %i9 = %lb to %ub step %step {
                      emitc.for %i10 = %lb to %ub step %step {
                        emitc.for %i11 = %lb to %ub step %step {
                          emitc.for %i12 = %lb to %ub step %step {
                            emitc.for %i13 = %lb to %ub step %step {
                              emitc.for %i14 = %lb to %ub step %step {
                                emitc.for %i15 = %lb to %ub step %step {
                                  emitc.for %i16 = %lb to %ub step %step {
                                    emitc.for %i17 = %lb to %ub step %step {
                                      emitc.for %i18 = %lb to %ub step %step {
                                        emitc.for %i19 = %lb to %ub step %step {
                                          %0 = emitc.call_opaque "f"() : () -> i32
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return
}
// CPP-DEFAULT: void test_for_nesting(size_t [[V1:[^ ]*]], size_t [[V2:[^ ]*]], size_t [[V3:[^ ]*]]) {
// CPP-DEFAULT-NEXT: size_t [[V4:[^ ]*]] = [[V1]] + [[V2]];
// CPP-DEFAULT-NEXT: size_t [[V5:[^ ]*]] = [[V2]] * [[V3]];
// CPP-DEFAULT-NEXT: size_t [[V6:[^ ]*]] = [[V1]] / [[V3]];
// CPP-DEFAULT-NEXT: for (size_t [[ITERi:i]] = [[V4]]; [[ITERi]] < [[V5]]; [[ITERi]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERj:j]] = [[V4]]; [[ITERj]] < [[V5]]; [[ITERj]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERk:[i-z]]] = [[V4]]; [[ITERk]] < [[V5]]; [[ITERk]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERl:[i-z]]] = [[V4]]; [[ITERl]] < [[V5]]; [[ITERl]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERm:[i-z]]] = [[V4]]; [[ITERm]] < [[V5]]; [[ITERm]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERn:[i-z]]] = [[V4]]; [[ITERn]] < [[V5]]; [[ITERn]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERo:[i-z]]] = [[V4]]; [[ITERo]] < [[V5]]; [[ITERo]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERp:[i-z]]] = [[V4]]; [[ITERp]] < [[V5]]; [[ITERp]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERq:[i-z]]] = [[V4]]; [[ITERq]] < [[V5]]; [[ITERq]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERr:[i-z]]] = [[V4]]; [[ITERr]] < [[V5]]; [[ITERr]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERs:[i-z]]] = [[V4]]; [[ITERs]] < [[V5]]; [[ITERs]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERt:[i-z]]] = [[V4]]; [[ITERt]] < [[V5]]; [[ITERt]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERu:[i-z]]] = [[V4]]; [[ITERu]] < [[V5]]; [[ITERu]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERv:[i-z]]] = [[V4]]; [[ITERv]] < [[V5]]; [[ITERv]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERw:[i-z]]] = [[V4]]; [[ITERw]] < [[V5]]; [[ITERw]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERx:[i-z]]] = [[V4]]; [[ITERx]] < [[V5]]; [[ITERx]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERy:[i-z]]] = [[V4]]; [[ITERy]] < [[V5]]; [[ITERy]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERz:z]] = [[V4]]; [[ITERz]] < [[V5]]; [[ITERz]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERi0:i0]] = [[V4]]; [[ITERi0]] < [[V5]]; [[ITERi0]] += [[V6]]) {
// CPP-DEFAULT-NEXT: for (size_t [[ITERi1:i1]] = [[V4]]; [[ITERi1]] < [[V5]]; [[ITERi1]] += [[V6]]) {
// CPP-DEFAULT-NEXT: int32_t [[V7:[^ ]*]] = f();
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: }
// CPP-DEFAULT-NEXT: return;