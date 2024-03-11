//===- BuiltinTest.cpp - PDL Builtin Tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/Builtins.h"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Region.h>

using namespace mlir;
using namespace mlir::pdl;

namespace {

class TestPatternRewriter : public PatternRewriter {
public:
  TestPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};

class TestPDLResultList : public PDLResultList {
public:
  TestPDLResultList(unsigned maxNumResults) : PDLResultList(maxNumResults) {}
    /// Return the list of PDL results.
  MutableArrayRef<PDLValue> getResults() {
    return results;
  }
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

TEST_F(BuiltinTest, add) {
  auto onei16 = rewriter.getI16IntegerAttr(1);
  auto onei32 = rewriter.getI32IntegerAttr(1);
  auto onei8 = rewriter.getI8IntegerAttr(1);
  auto largesti8 = rewriter.getI8IntegerAttr(127);

  // check signed integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onei8, largesti8}).failed());
  }

  IntegerType Uint8 = rewriter.getIntegerType(8, false);
  auto oneUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 1, false));
  auto largestUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 255, false));

  // check unsigned integer overflow  
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {oneUint8, largestUint8}).failed());
  }

  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onei16, onei16}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
    cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
    2);
  }
  
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onei16, onei32}).failed());
  }

  auto onef16 = rewriter.getF16FloatAttr(1.0);
  auto onef32 = rewriter.getF32FloatAttr(1.0);
  auto zerof32 = rewriter.getF32FloatAttr(0.0);
  auto negzerof32 = rewriter.getF32FloatAttr(-0.0);
  auto zerof64 = rewriter.getF64FloatAttr(0.0);

  auto maxValF16 = rewriter.getF16FloatAttr(llvm::APFloat::getLargest(llvm::APFloat::IEEEhalf()).convertToFloat());

  // check float overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onef16, maxValF16}).failed());
  }

  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onef32, onef32}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
      cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
      2.0);
  }

  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {zerof32, negzerof32}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
      cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
      0.0);
  }

  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {zerof32, zerof64}).failed());
  }
}
} // namespace
