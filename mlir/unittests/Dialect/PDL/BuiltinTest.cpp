//===- BuiltinTest.cpp - PDL Builtin Tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/Builtins.h"
#include "gmock/gmock.h"
#include <mlir/IR/PatternMatch.h>

using namespace mlir;
using namespace mlir::pdl;

namespace {

class TestPatternRewriter : public PatternRewriter {
public:
  TestPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

class BuiltinTest : public ::testing::Test {
public:
  MLIRContext ctx;
  TestPatternRewriter rewriter{&ctx};
};

TEST_F(BuiltinTest, createDictionaryAttr) {
  auto attr = builtin::createDictionaryAttr(rewriter);
  auto dict = dyn_cast<DictionaryAttr>(attr);
  EXPECT_TRUE(dict);
  EXPECT_TRUE(dict.empty());
}

TEST_F(BuiltinTest, addEntryToDictionaryAttr) {
  auto dictAttr = rewriter.getDictionaryAttr({});

  mlir::Attribute updated = builtin::addEntryToDictionaryAttr(
      rewriter, dictAttr, rewriter.getStringAttr("testAttr"),
      rewriter.getI16IntegerAttr(0));

  EXPECT_TRUE(updated.cast<DictionaryAttr>().contains("testAttr"));

  auto second = builtin::addEntryToDictionaryAttr(
      rewriter, updated, rewriter.getStringAttr("testAttr2"),
      rewriter.getI16IntegerAttr(0));
  EXPECT_TRUE(second.cast<DictionaryAttr>().contains("testAttr"));
  EXPECT_TRUE(second.cast<DictionaryAttr>().contains("testAttr2"));
}

TEST_F(BuiltinTest, createArrayAttr) {
  auto attr = builtin::createArrayAttr(rewriter);
  auto dict = dyn_cast<ArrayAttr>(attr);
  EXPECT_TRUE(dict);
  EXPECT_TRUE(dict.empty());
}

TEST_F(BuiltinTest, addElemToArrayAttr) {
  auto dict = rewriter.getDictionaryAttr(
      rewriter.getNamedAttr("key", rewriter.getStringAttr("value")));
  rewriter.getArrayAttr({});

  auto arrAttr = builtin::createArrayAttr(rewriter);
  mlir::Attribute updatedArrAttr =
      builtin::addElemToArrayAttr(rewriter, arrAttr, dict);

  auto dictInsideArrAttr =
      cast<DictionaryAttr>(*cast<ArrayAttr>(updatedArrAttr).begin());
  EXPECT_EQ(dictInsideArrAttr, dict);
}
} // namespace
