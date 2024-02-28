// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s
// RUN: mlir-translate -mlir-to-cpp -declare-variables-at-top %s | FileCheck %s

func.func @load_store(%arg0: !emitc.array<4x8xf32>, %arg1: !emitc.array<3x5xf32>, %arg2: index, %arg3: index) {
  %0 = emitc.subscript %arg0[%arg2, %arg3] : <4x8xf32>
  %1 = emitc.subscript %arg1[%arg2, %arg3] : <3x5xf32>
  emitc.assign %0 : f32 to %1 : f32
  return
}
// CHECK: void load_store(float [[V1:[^ ]*]][4][8], float [[V2:[^ ]*]][3][5],
// CHECK-SAME:            size_t [[V3:[^ ]*]], size_t [[V4:[^ ]*]])
// CHECK-NEXT: [[V2]][[[V3]]][[[V4]]] = [[V1]][[[V3]]][[[V4]]];
