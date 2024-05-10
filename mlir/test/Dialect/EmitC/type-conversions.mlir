// RUN: mlir-opt -debug -verify-diagnostics -test-emitc-type-conversions %s | FileCheck %s

// CHECK-LABEL: ssize_t
func.func @ssize_t() {
    %0 = "emitc.constant"(){value = 42 : index} : () -> index
    return
}