// RUN: mlir-opt -split-input-file -convert-pdl-to-pdl-interp %s | FileCheck %s

// Ensuse that the dependency between add & less
// causes them to be in the correct order.
// CHECK: apply_constraint "__builtin_add"
// CHECK: apply_constraint "__builtin_less"

module {
  pdl.pattern @test : benefit(1) {
    %0 = attribute
    %1 = types
    %2 = operation "tosa.mul"  {"shift" = %0} -> (%1 : !pdl.range<type>)
    %3 = attribute = 0 : i32
    %4 = attribute = 1 : i32
    %5 = apply_native_constraint "__builtin_add"(%3, %4 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
    apply_native_constraint "__builtin_less"(%0, %5 : !pdl.attribute, !pdl.attribute)
    rewrite %2 {
      replace %2 with %2
    }
  }
}
