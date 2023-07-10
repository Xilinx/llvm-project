//===- DemoTypes.h - Demo dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DEMO_DEMOTYPES_H
#define DEMO_DEMOTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Demo/DemoOpsTypes.h.inc"

#endif // DEMO_DEMOTYPES_H
