add_mlir_conversion_library(MLIRArithToEmitC
  ArithToEmitC.cpp
  ArithToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIREmitCDialect
  MLIREmitCTransforms
  MLIRPass
  MLIRTransformUtils
  )

set_property(SOURCE ArithToEmitC.cpp APPEND PROPERTY COMPILE_FLAGS "-g" )
set_property(SOURCE ArithToEmitCPass.cpp APPEND PROPERTY COMPILE_FLAGS "-g" )