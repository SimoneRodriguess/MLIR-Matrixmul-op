cmake_minimum_required(VERSION 3.20)
project(mlir-matmul-opt)

find_package(MLIR REQUIRED CONFIG
  HINTS ~/llvm-project/build/lib/cmake/mlir)
find_package(LLVM REQUIRED CONFIG
  HINTS ~/llvm-project/build/lib/cmake/llvm)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

add_executable(matmul-opt
  lib/MatmulTilePass.cpp
  lib/main.cpp
)

target_compile_options(matmul-opt PRIVATE -fno-rtti)

target_include_directories(matmul-opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(matmul-opt PRIVATE
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRPass
  MLIRTransforms
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)
