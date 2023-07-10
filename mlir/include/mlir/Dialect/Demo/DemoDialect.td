//===- DemoDialect.td - Demo dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEMO_DIALECT
#define DEMO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Demo dialect definition.
//===----------------------------------------------------------------------===//

def Demo_Dialect : Dialect {
    let name = "demo";
    let summary = "A demo MLIR dialect.";
    let description = [{}];
    let cppNamespace = "::mlir::demo";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base demo operation definition.
//===----------------------------------------------------------------------===//

class Demo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Demo_Dialect, mnemonic, traits>;

#endif // DEMO_DIALECT
