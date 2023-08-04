//===- DemoTypes.td - Demo dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEMO_TYPES
#define DEMO_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Demo/DemoDialect.td"

//===----------------------------------------------------------------------===//
// Demo type definitions
//===----------------------------------------------------------------------===//

class Demo_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Demo_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Demo_CustomType : Demo_Type<"Custom", "custom"> {
    let summary = "Demo custom type";
    let description = "Custom type in demo dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // DEMO_TYPES
