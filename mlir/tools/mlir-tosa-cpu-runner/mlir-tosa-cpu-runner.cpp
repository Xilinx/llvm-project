//===- mlir-tosa-cpu-runner.cpp - MLIR TOSA Execution on CPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR GPU module and host part to LLVM IR before
// JIT-compiling and executing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"

using namespace mlir;

/// A utility function that builds llvm::Module from two nested MLIR modules.
///
/// module @main {
///   module @kernel {
///     // Some ops
///   }
///   // Some other ops
/// }
///
/// Each of these two modules is translated to LLVM IR module, then they are
/// linked together and returned.
static std::unique_ptr<llvm::Module>
convertMLIRModule(Operation *op, llvm::LLVMContext &context) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("op must be a 'builtin.module"), nullptr;
  // Verify that there is only one nested module.
  auto modules = module.getOps<ModuleOp>();

  // Translate nested module and erase it.
  ModuleOp nested = *modules.begin();
  std::unique_ptr<llvm::Module> kernelModule =
      translateModuleToLLVMIR(nested, context);
  nested.erase();

  std::unique_ptr<llvm::Module> mainModule =
      translateModuleToLLVMIR(module, context);
  llvm::Linker::linkModules(*mainModule, std::move(kernelModule));
  return mainModule;
}

static LogicalResult runMLIRPasses(Operation *module) {
  PassManager passManager(module->getContext(),
                          module->getName().getStringRef());
  applyPassManagerCLOptions(passManager);
  passManager.addPass(tosa::createTosaToArith());
  passManager.addPass(tosa::createTosaToTensor());
  passManager.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());
  passManager.addPass(createConvertElementwiseToLinalgPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(createLinalgDetensorizePass());
  //Bufferization passes
  passManager.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  passManager.addNestedPass<func::FuncOp>(createTensorBufferizePass());
  passManager.addPass(func::createFuncBufferizePass());
  passManager.addPass(arith::createArithBufferizePass());
  bufferization::OneShotBufferizationOptions opts;
  opts.bufferizeFunctionBoundaries = true;
  opts.allowReturnAllocs = true;
  passManager.addPass(bufferization::createOneShotBufferizePass(opts));
  passManager.addNestedPass<func::FuncOp>(bufferization::createFinalizingBufferizePass());
  passManager.addNestedPass<func::FuncOp>(bufferization::createBufferLoopHoistingPass());
  passManager.addPass(bufferization::createDropEquivalentBufferResultsPass());
  passManager.addNestedPass<func::FuncOp>(bufferization::createBufferDeallocationPass());

  passManager.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  passManager.addPass(createConvertSCFToCFPass());
  passManager.addPass(createConvertLinalgToLLVMPass());
  passManager.addPass(createConvertFuncToLLVMPass());
  passManager.addPass(createConvertMathToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(mlir::LLVM::createLegalizeForExportPass());
  return passManager.run(module);
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  mlir::JitRunnerConfig jitRunnerConfig;
  jitRunnerConfig.mlirTransformer = runMLIRPasses;
  //jitRunnerConfig.llvmModuleBuilder = convertMLIRModule;

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect, 
                  mlir::linalg::LinalgDialect, mlir::tosa::TosaDialect>();
  mlir::registerLLVMDialectTranslation(registry);

  return mlir::JitRunnerMain(argc, argv, registry, jitRunnerConfig);
}