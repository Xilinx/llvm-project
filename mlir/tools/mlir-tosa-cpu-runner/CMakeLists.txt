set(LLVM_LINK_COMPONENTS
  Linker
  Core
  Support
  nativecodegen
  native
)

if (MLIR_ENABLE_TOSA_CPU_RUNNER)
  message(STATUS "Building TOSA CPU runner")

  add_mlir_tool(mlir-tosa-cpu-runner
    mlir-tosa-cpu-runner.cpp
  )

  llvm_update_compile_flags(mlir-tosa-cpu-runner)

  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

  target_link_libraries(mlir-tosa-cpu-runner PRIVATE
    ${conversion_libs}
    #FIXME CHANGE LIBS
    MLIRAnalysis
    MLIRArithDialect
    MLIRExecutionEngine
    MLIRFuncDialect
    MLIRTosaDialect
    MLIRLinalgDialect
    MLIRIR
    MLIRJitRunner
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRTranslateLib
    MLIRSupport
  )
endif()
