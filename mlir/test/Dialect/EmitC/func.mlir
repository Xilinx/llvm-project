// RUN: mlir-opt %s -split-input-file

// CHECK: emitc.func @f
// CHECK-SAME: %{{[^:]*}}: i32 ref
emitc.func @f(%x: i32 {emitc.reference}) {
    emitc.return
}

// -----

// CHECK: emitc.func @f
// CHECK-SAME: %{{[^:]*}}: i32 ref
emitc.func @f(%x: i32 ref) {
    emitc.return
}
