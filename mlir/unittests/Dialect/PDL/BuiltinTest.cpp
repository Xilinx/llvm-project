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

TEST_F(BuiltinTest, mul) {
  auto twoi8 = rewriter.getI8IntegerAttr(2);
  auto twoi16 = rewriter.getI16IntegerAttr(2);

  auto largesti8 = rewriter.getI8IntegerAttr(-1);

  // check signless integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(twoi8.getType().isSignlessInteger());

    EXPECT_TRUE(builtin::mul(rewriter, results, {twoi8, largesti8}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mul(rewriter, results, {twoi8, twoi8}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
        4);
  }

  IntegerType Uint8 = rewriter.getIntegerType(8, false);
  auto twoUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 2, false));
  auto largestUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 255, false));

  // check unsigned integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::mul(rewriter, results, {twoUint8, largestUint8}).failed());
  }

  IntegerType SInt8 = rewriter.getIntegerType(8, true);
  auto twoSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 2, true));
  auto largestSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 127, true));

  // check signed integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::mul(rewriter, results, {twoSInt8, largestSInt8}).failed());
  }

  // check integer types mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mul(rewriter, results, {twoi8, twoi16}).failed());
  }

  auto twof16 = rewriter.getF16FloatAttr(2.0);
  auto maxValF16 = rewriter.getF16FloatAttr(
      llvm::APFloat::getLargest(llvm::APFloat::IEEEhalf()).convertToFloat());

  // check float overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mul(rewriter, results, {twof16, maxValF16}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mul(rewriter, results, {twof16, twof16}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
        4.0);
  }
}

TEST_F(BuiltinTest, div) {
  auto sixi8 = rewriter.getI8IntegerAttr(6);
  auto twoi8 = rewriter.getI8IntegerAttr(2);

  // signless integer
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::div(rewriter, results, {sixi8, twoi8}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
        3);
  }

  IntegerType Uint8 = rewriter.getIntegerType(8, false);
  auto oneUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 1, false));
  auto largestUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 255, false));

  // unsigned integer
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::div(rewriter, results, {largestUint8, oneUint8}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getZExtValue(),
        (uint64_t)255);
  }

  IntegerType SInt8 = rewriter.getIntegerType(8, true);
  auto smallestSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, -128, true));
  auto minusOneSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, -1, true));
  auto twoSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 2, true));

  // check signed integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::div(rewriter, results, {smallestSInt8, minusOneSInt8})
                    .failed());
  }

  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::div(rewriter, results, {twoSInt8, minusOneSInt8}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
        -2);
  }

  auto smallF16 = rewriter.getF16FloatAttr(0.0001);
  auto twoF16 = rewriter.getF16FloatAttr(2.0);
  auto maxValF16 = rewriter.getF16FloatAttr(
      llvm::APFloat::getLargest(llvm::APFloat::IEEEhalf()).convertToFloat());

  // check float overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::div(rewriter, results, {maxValF16, smallF16}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::div(rewriter, results, {twoF16, twoF16}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
        1.0);
  }

  // check type mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::div(rewriter, results, {twoF16, twoi8}).failed());
  }
}

TEST_F(BuiltinTest, mod) {
  auto eighti8 = rewriter.getI8IntegerAttr(8);
  auto threei8 = rewriter.getI8IntegerAttr(3);

  // signless integer
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::mod(rewriter, results, {eighti8, threei8}).succeeded());
  }

  IntegerType Uint8 = rewriter.getIntegerType(8, false);
  auto oneUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 1, false));
  auto largestUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 255, false));

  // unsigned integer
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::mod(rewriter, results, {largestUint8, oneUint8}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getZExtValue(),
        (uint64_t)0);
  }

  IntegerType SInt8 = rewriter.getIntegerType(8, true);
  auto minusTenSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, -10, true));
  auto threeSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 3, true));

  // signed integer
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mod(rewriter, results, {minusTenSInt8, threeSInt8})
                    .succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
        -1);
  }

  auto twoF16 = rewriter.getF16FloatAttr(2.0);

  // float
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mod(rewriter, results, {twoF16, twoF16}).succeeded());
  }

  // check type mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::mod(rewriter, results, {twoF16, eighti8}).failed());
  }
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

TEST_F(BuiltinTest, sub) {
  auto onei16 = rewriter.getI16IntegerAttr(1);
  auto onei32 = rewriter.getI32IntegerAttr(1);
  auto onei8 = rewriter.getI8IntegerAttr(1);
  auto largesti8 = rewriter.getI8IntegerAttr(-1);

  // check signless integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(onei8.getType().isSignlessInteger());
    EXPECT_TRUE(builtin::sub(rewriter, results, {onei8, largesti8}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::sub(rewriter, results, {onei16, onei16}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<IntegerAttr>(result.cast<Attribute>()).getValue().getSExtValue(),
        0);
  }

  IntegerType Uint8 = rewriter.getIntegerType(8, false);
  auto oneUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 1, false));
  auto largestUint8 = rewriter.getIntegerAttr(Uint8, APInt(8, 255, false));

  // check unsigned integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::sub(rewriter, results, {oneUint8, largestUint8}).failed());
  }

  IntegerType SInt8 = rewriter.getIntegerType(8, true);
  auto minusOneSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, -1, true));
  auto largestSInt8 = rewriter.getIntegerAttr(SInt8, APInt(8, 127, true));

  // check signed integer overflow
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::sub(rewriter, results, {largestSInt8, minusOneSInt8})
                    .failed());
  }

  // check integer types mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::sub(rewriter, results, {onei16, onei32}).failed());
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
    EXPECT_TRUE(builtin::sub(rewriter, results, {maxValF16, onef16}).failed());
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::sub(rewriter, results, {onef32, onef32}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
        0.0);
  }

  // check correctness of result
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(
        builtin::sub(rewriter, results, {zerof32, negzerof32}).succeeded());

    PDLValue result = results.getResults()[0];
    EXPECT_EQ(
        cast<FloatAttr>(result.cast<Attribute>()).getValue().convertToFloat(),
        0.0);
  }

  // check float types mismatch
  {
    TestPDLResultList results(1);
    EXPECT_TRUE(builtin::sub(rewriter, results, {zerof32, zerof64}).failed());
  }
}
} // namespace
