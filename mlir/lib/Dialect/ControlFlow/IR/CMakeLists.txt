add_mlir_dialect_library(MLIRControlFlowDialect
  ControlFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR

  DEPENDS
  MLIRControlFlowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  )

  if(WIN32 AND NOT MINGW)
    target_compile_definitions(MLIRControlFlowDialect PRIVATE
                                 # Don't check the msvcprt iterator debug levels
                                 # as we will define the iterator types; libc++
                                 # uses a different macro to identify the debug
                                 # level.
                                 _ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH)
  endif()