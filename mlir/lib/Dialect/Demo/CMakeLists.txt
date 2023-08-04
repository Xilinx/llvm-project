add_mlir_dialect_library(MLIRDemo
        DemoTypes.cpp
        DemoDialect.cpp
        DemoOps.cpp
        TilingInterfaceImpl.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Demo

        DEPENDS
        MLIRDemoOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLinalgUtils
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
