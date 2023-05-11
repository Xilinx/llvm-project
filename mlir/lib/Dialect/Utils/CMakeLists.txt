add_mlir_library(MLIRDialectUtils
  FoldUtils.cpp
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithUtils
  MLIRIR
)
