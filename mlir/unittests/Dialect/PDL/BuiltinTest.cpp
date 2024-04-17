//===- BuiltinTest.cpp - PDL Builtin Tests --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/Builtins.h"
#include "gmock/gmock.h"
#include <cstdint>
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
  MutableArrayRef<PDLValue> getResults() { return results; }
};

class BuiltinTest : public ::testing::Test {
public:
  MLIRContext ctx;
  TestPatternRewriter rewriter{&ctx};
};

TEST_F(BuiltinTest, createDictionaryAttr) {
  TestPDLResultList results(1);
  EXPECT_TRUE(succeeded(builtin::createDictionaryAttr(rewriter, results, {})));
  ASSERT_TRUE(results.getResults().size() == 1);
  auto dict = dyn_cast_or_null<DictionaryAttr>(
      results.getResults().back().cast<Attribute>());
  ASSERT_TRUE(dict);
  EXPECT_TRUE(dict.empty());
}

TEST_F(BuiltinTest, addEntryToDictionaryAttr) {
  TestPDLResultList results(1);

  auto dictAttr = rewriter.getDictionaryAttr({});
  EXPECT_TRUE(succeeded(builtin::addEntryToDictionaryAttr(
      rewriter, results,
      {dictAttr, rewriter.getStringAttr("testAttr"),
       rewriter.getI16IntegerAttr(0)})));
  ASSERT_TRUE(results.getResults().size() == 1);
  mlir::Attribute updated = results.getResults().front().cast<Attribute>();
  EXPECT_TRUE(cast<DictionaryAttr>(updated).contains("testAttr"));

  results = TestPDLResultList(1);
  EXPECT_TRUE(succeeded(builtin::addEntryToDictionaryAttr(
      rewriter, results,
      {updated, rewriter.getStringAttr("testAttr2"),
       rewriter.getI16IntegerAttr(0)})));
  ASSERT_TRUE(results.getResults().size() == 1);
  mlir::Attribute second = results.getResults().front().cast<Attribute>();

  EXPECT_TRUE(cast<DictionaryAttr>(second).contains("testAttr"));
  EXPECT_TRUE(cast<DictionaryAttr>(second).contains("testAttr2"));
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
  auto largesti8 = rewriter.getI8IntegerAttr(-1);

  // check signless integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(onei8.getType().isSignlessInteger());

    EXPECT_TRUE(builtin::add(rewriter, results, {onei8, largesti8}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onei16, onei16}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
        2);
  }

  IntegerType Uint8 = rewriter.getIntegerType(8, false);
  auto oneUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 1, false));
  auto largestUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 255, false));

  // check unsigned integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::add(rewriter, results, {oneUint8, largestUint8}).failed());
  }

  IntegerType SInt8 = rewriter.getIntegerType(8, true);
  auto oneSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 1, true));
  auto largestSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 127, true));

  // check signed integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::add(rewriter, results, {oneSInt8, largestSInt8}).failed());
  }

  // check integer types mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onei16, onei32}).failed());
  }

  auto onef16 = rewriter.getF16FloatAttr(1.0);
  auto onef32 = rewriter.getF32FloatAttr(1.0);
  auto zerof32 = rewriter.getF32FloatAttr(0.0);
  auto negzerof32 = rewriter.getF32FloatAttr(-0.0);
  auto zerof64 = rewriter.getF64FloatAttr(0.0);

  auto maxValF16 = rewriter.getF16FloatAttr(
      llvm::APFloat::getLargest(llvm::APFloat::IEEEhalf()).convertToFloat());

  // check float overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onef16, maxValF16}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {onef32, onef32}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
        2.0);
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::add(rewriter, results, {zerof32, negzerof32}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
        0.0);
  }

  // check float types mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::add(rewriter, results, {zerof32, zerof64}).failed());
  }
}
} // namespace
