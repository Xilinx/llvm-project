# Exclude tests from libMLIR.so
add_mlir_library(MLIREmitCTransformsTest
  TestTypeConversions.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  )
