# Kernel auto-selection: measurements

Benchmark: `tools/bench/kernels.R` (median ms per `cpp_assign_accumulate` call, 8,192 rows,
default chunk sizes 4096 x 1024, `bench::mark`, min_time 0.5 s, up to 10 iterations).
Raw data: `results-veclib.csv`, `results-refblas.csv`.

## Machine

- Apple M2 Max, 12 cores, macOS 27.2, R 4.6, clang 21.
- Package built with OpenMP (`-Xclang -fopenmp`, libomp; max threads 12), so threads = 1 vs 2 is meaningful.
  CRAN's macOS R has no OpenMP, so the threads = 1 tables also describe Mac users' serial builds.
- Optimised BLAS: Apple vecLib/Accelerate (via `DYLD_LIBRARY_PATH` override of `libRblas.dylib`).
- Reference BLAS: R's `libRblas.0.dylib`.
- Only vecLib was measured among optimised BLASes. OpenBLAS/MKL were not available here.

## Results

**veclib, threads = 1** (ms per call; ratio = blas / direct, below 1 means blas is faster)

| d | k | direct | blas | ratio |
|---:|---:|---:|---:|---:|
| 2 | 100 | 1.48 | 0.97 | 0.66 |
| 8 | 100 | 1.71 | 1.05 | 0.61 |
| 16 | 100 | 2.78 | 1.21 | 0.44 |
| 32 | 100 | 4.95 | 1.63 | 0.33 |
| 128 | 100 | 20.16 | 4.27 | 0.21 |
| 2 | 1000 | 13.74 | 10.68 | 0.78 |
| 8 | 1000 | 15.71 | 10.36 | 0.66 |
| 16 | 1000 | 24.35 | 10.36 | 0.43 |
| 32 | 1000 | 41.89 | 11.03 | 0.26 |
| 128 | 1000 | 176.12 | 16.54 | 0.09 |
| 2 | 10000 | 139.27 | 107.22 | 0.77 |
| 8 | 10000 | 152.21 | 105.06 | 0.69 |
| 16 | 10000 | 236.35 | 105.22 | 0.45 |
| 32 | 10000 | 412.25 | 110.66 | 0.27 |
| 128 | 10000 | 1780.05 | 144.19 | 0.08 |

**veclib, threads = 2** (ms per call; ratio = blas / direct, below 1 means blas is faster)

| d | k | direct | blas | ratio |
|---:|---:|---:|---:|---:|
| 2 | 100 | 0.79 | 0.65 | 0.82 |
| 8 | 100 | 0.96 | 0.77 | 0.80 |
| 16 | 100 | 1.54 | 0.94 | 0.61 |
| 32 | 100 | 2.68 | 1.37 | 0.51 |
| 128 | 100 | 11.59 | 4.04 | 0.35 |
| 2 | 1000 | 7.14 | 5.99 | 0.84 |
| 8 | 1000 | 8.00 | 5.60 | 0.70 |
| 16 | 1000 | 12.60 | 6.02 | 0.48 |
| 32 | 1000 | 29.01 | 6.70 | 0.23 |
| 128 | 1000 | 93.22 | 11.83 | 0.13 |
| 2 | 10000 | 68.95 | 57.44 | 0.83 |
| 8 | 10000 | 78.42 | 56.15 | 0.72 |
| 16 | 10000 | 119.70 | 58.64 | 0.49 |
| 32 | 10000 | 215.09 | 63.13 | 0.29 |
| 128 | 10000 | 889.03 | 92.05 | 0.10 |

**refblas, threads = 1** (ms per call; ratio = blas / direct, below 1 means blas is faster)

| d | k | direct | blas | ratio |
|---:|---:|---:|---:|---:|
| 2 | 100 | 1.44 | 1.32 | 0.92 |
| 8 | 100 | 1.69 | 2.97 | 1.76 |
| 16 | 100 | 2.68 | 5.15 | 1.92 |
| 32 | 100 | 4.73 | 9.56 | 2.02 |
| 128 | 100 | 20.11 | 36.79 | 1.83 |
| 2 | 1000 | 13.56 | 17.19 | 1.27 |
| 8 | 1000 | 15.63 | 32.19 | 2.06 |
| 16 | 1000 | 24.53 | 53.13 | 2.17 |
| 32 | 1000 | 41.53 | 94.92 | 2.29 |
| 128 | 1000 | 178.08 | 336.12 | 1.89 |
| 2 | 10000 | 136.65 | 171.59 | 1.26 |
| 8 | 10000 | 155.50 | 322.17 | 2.07 |
| 16 | 10000 | 243.40 | 532.78 | 2.19 |
| 32 | 10000 | 405.07 | 925.73 | 2.29 |
| 128 | 10000 | 1815.69 | 4362.75 | 2.40 |

**refblas, threads = 2** (ms per call; ratio = blas / direct, below 1 means blas is faster)

| d | k | direct | blas | ratio |
|---:|---:|---:|---:|---:|
| 2 | 100 | 0.78 | 1.04 | 1.33 |
| 8 | 100 | 0.93 | 2.70 | 2.90 |
| 16 | 100 | 1.50 | 4.82 | 3.21 |
| 32 | 100 | 2.66 | 9.23 | 3.47 |
| 128 | 100 | 12.23 | 36.86 | 3.01 |
| 2 | 1000 | 7.04 | 12.60 | 1.79 |
| 8 | 1000 | 8.05 | 27.60 | 3.43 |
| 16 | 1000 | 12.70 | 48.60 | 3.83 |
| 32 | 1000 | 28.94 | 91.34 | 3.16 |
| 128 | 1000 | 98.27 | 333.51 | 3.39 |
| 2 | 10000 | 69.31 | 126.36 | 1.82 |
| 8 | 10000 | 77.75 | 280.94 | 3.61 |
| 16 | 10000 | 119.85 | 484.51 | 4.04 |
| 32 | 10000 | 204.81 | 882.78 | 4.31 |
| 128 | 10000 | 915.95 | 3740.92 | 4.08 |

## Findings

- Reference BLAS: "blas" is slower than "direct" almost everywhere (1.3x to 4.3x slower; only d = 2, k = 100,
  threads = 1 is near parity at 0.92, within noise of the 10% band). "direct" is always the right choice.
  This supports the allow-list: an unrecognised BLAS must fall back to "direct".
- vecLib: "blas" is faster than "direct" in every cell, including d = 2 (0.66 to 0.84) and d = 8
  (0.61 to 0.80). "direct" is never within 10% of "blas". The gap grows with d: 0.2x at d = 128 serial,
  0.08x at d = 128, k = 10000. Threads = 2 shows the same ordering with smaller ratios at high d,
  because "direct" scales across threads while the vecLib dgemm is already multi-threaded or efficient.
- The earlier rule (blas only for d > 16) left a 1.3x to 2.3x loss on the table at d = 2 to 16 under vecLib.

## Rule

`"auto"` picks "blas" when `d >= 2` and `blas_is_optimised()` (allow-list unchanged), else "direct".

d = 1 was not measured (the grid starts at d = 2) and is left on "direct". The d = 2 ratios are 0.77 to 0.84
at threads = 1 and 0.82 to 0.84 at threads = 2, outside the 10% band, so the threshold is set at 2.
k below 100 was not measured either; small k has little work in either kernel.

Caveat: the threshold rests on one optimised BLAS (vecLib on Apple silicon). A user on OpenBLAS or MKL
may see a different crossover; `options(emeans.kernel = "direct")` overrides it.
