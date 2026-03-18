#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
  void registerAllDialects(DialectRegistry &registry);
  void registerAllExtensions(DialectRegistry &registry);
  void registerAllPasses();
  void registerMatmulTilePass();
}

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerMatmulTilePass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "matmul-opt\n", registry));
}
