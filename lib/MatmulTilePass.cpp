#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

struct MatmulTilePass
    : public PassWrapper<MatmulTilePass, OperationPass<func::FuncOp>> {

  StringRef getArgument() const override { return "tile-matmul"; }
  StringRef getDescription() const override { return "Tile linalg.matmul ops"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());

    func.walk([&](linalg::MatmulOp matmul) {
      SmallVector<OpFoldResult> tileSizes = {
          rewriter.getIndexAttr(32),
          rewriter.getIndexAttr(32),
          rewriter.getIndexAttr(32)};

      scf::SCFTilingOptions options;
      options.setTileSizes(tileSizes);

      rewriter.setInsertionPoint(matmul);

      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter,
                            cast<TilingInterface>(matmul.getOperation()),
                            options);

      if (failed(tilingResult)) {
        matmul.emitError("Failed to tile matmul");
        signalPassFailure();
        return;
      }

      rewriter.replaceOp(matmul, tilingResult->replacements);
    });
  }
};

namespace mlir {
void registerMatmulTilePass() {
  PassRegistration<MatmulTilePass>();
}
} // namespace mlir
