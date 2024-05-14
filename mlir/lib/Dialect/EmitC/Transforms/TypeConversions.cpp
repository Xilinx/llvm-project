//===- TypeConversions.cpp - Convert signless types into C/C++ types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

void populateEmitCSizeTypeConversionPatterns(TypeConverter &converter) {
  converter.addConversion(
      [](IndexType type) { return emitc::SizeTType::get(type.getContext()); });
}
