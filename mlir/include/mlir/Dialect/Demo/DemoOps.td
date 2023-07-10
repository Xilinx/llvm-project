//===- DemoOps.td - Demo dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEMO_OPS
#define DEMO_OPS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Demo/DemoTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Demo_FooOp : Demo_Op<"foo", [Pure]> {
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, F64Tensor:$out);
    let results = (outs F64Tensor:$res);

    let assemblyFormat = [{
      $lhs `,` $rhs `,` $out attr-dict
        `:` type($lhs) `,` type($rhs) `,` type($out) `to` type($res)
    }];
}

#endif // DEMO_OPS
