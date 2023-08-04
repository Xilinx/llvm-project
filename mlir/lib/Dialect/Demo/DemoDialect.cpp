//===- DemoDialect.cpp - Demo dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Demo/DemoDialect.h"
#include "mlir/Dialect/Demo/DemoOps.h"
#include "mlir/Dialect/Demo/DemoTypes.h"

using namespace mlir;
using namespace mlir::demo;

#include "mlir/Dialect/Demo/DemoOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Demo dialect.
//===----------------------------------------------------------------------===//

void DemoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Demo/DemoOps.cpp.inc"
      >();
  registerTypes();
}
