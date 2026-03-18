# MLIR Matrix Multiply Tiling Pass

A custom MLIR optimization pass that tiles `linalg.matmul` operations for improved cache locality, with a full lowering pipeline from linalg dialect down to LLVM IR.

## What This Does

Takes a naive `linalg.matmul` op and transforms it into a tiled loop nest with 32x32x32 tile sizes. The tiled version keeps working data in L1 cache, reducing cache misses significantly.

## Why It Matters

A naive matrix multiply iterates over entire rows and columns at once. For large matrices, this causes constant cache evictions — data gets loaded from RAM, used once, and thrown out. Loop tiling fixes this by breaking the matrix into small tiles (32x32) that fit entirely in L1 cache. The same arithmetic happens, but now data stays on-chip while it's being used.

## Lowering Pipeline
```
linalg.matmul
    ↓  --tile-matmul          (this pass)
scf.for loops, step 32
    ↓  --convert-linalg-to-loops
6 nested scf.for loops
    ↓  --convert-scf-to-cf
basic blocks + branches
    ↓  --expand-strided-metadata
    ↓  --lower-affine
    ↓  --finalize-memref-to-llvm
    ↓  --convert-arith-to-llvm
    ↓  --convert-cf-to-llvm
    ↓  --convert-func-to-llvm
    ↓  --reconcile-unrealized-casts
LLVM IR (.ll)
    ↓  clang -O2
x86-64 binary
```

## Results

Benchmarked on 512x512 f32 matrices, 5 runs, best time recorded:

| Matrix Size | Naive | Tiled | Speedup |
|-------------|-------|-------|---------|
| 512x512     | 352ms | 171ms | 2.06x   |

Compiled with `-O2`. Speedup comes entirely from improved cache locality — no vectorization, no algorithmic changes.

## Project Structure
```
mlir-matmul-opt/
├── CMakeLists.txt
├── lib/
│   ├── MatmulTilePass.cpp
│   └── main.cpp
├── test/
│   ├── matmul_naive.mlir
│   └── matmul_tiled.mlir
└── benchmark/
    └── bench.c
```

## How to Build
```bash
# Prerequisites: LLVM/MLIR built from source
# Clone: https://github.com/llvm/llvm-project

cmake -G Ninja -B build \
  -DMLIR_DIR=~/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=~/llvm-project/build/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build
```

## How to Run
```bash
# Apply tiling pass
./build/matmul-opt test/matmul_naive.mlir --tile-matmul

# Full lowering to LLVM IR
mlir-opt test/matmul_naive.mlir \
  --tile-matmul \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --finalize-memref-to-llvm \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts | \
mlir-translate --mlir-to-llvmir -o matmul.ll
```

## What I Learned

Writing this pass taught me how MLIR's multi-level lowering works in practice — how a single high-level op like `linalg.matmul` decomposes through multiple dialect layers before reaching machine code. The tiling transformation itself is just loop strip-mining, but seeing it expressed as an IR rewrite made the mechanics concrete.

## Future Work

- Add vectorization pass (AVX2 SIMD) — expected 2-4x additional speedup
- Tune tile sizes per cache size
- Extend to non-square and non-power-of-2 matrices
- CUDA lowering via GPU dialect
