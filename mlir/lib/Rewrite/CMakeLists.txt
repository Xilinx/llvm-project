add_mlir_library(MLIRRewrite
  ByteCode.cpp
  FrozenRewritePatternSet.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

  DEPENDS
  mlir-generic-headers

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRPDLToPDLInterp
  MLIRSideEffectInterfaces
  )
target_compile_options(obj.MLIRRewrite PRIVATE -g3 -O0)
