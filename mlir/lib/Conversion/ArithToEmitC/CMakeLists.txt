add_mlir_conversion_library(ArithToEmitC
  ArithToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRArithDialect
  MLIRTransforms
)
