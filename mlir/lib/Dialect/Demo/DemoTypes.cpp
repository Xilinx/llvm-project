//===- DemoTypes.cpp - Demo dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Demo/DemoTypes.h"

#include "mlir/Dialect/Demo/DemoDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::demo;

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Demo/DemoOpsTypes.cpp.inc"

void DemoDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Demo/DemoOpsTypes.cpp.inc"
      >();
}
