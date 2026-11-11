## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* local: macOS 27.2 (aarch64), R 4.6.2
* GitHub Actions: macOS (R release), Windows (R release), Ubuntu (R release
  and R-devel)

## Notes for the reviewers

* The algorithm is ported from the MIT-licensed e-means code in the 'faiss'
  library. Its copyright holder, Meta Platforms, Inc. and affiliates, is
  listed in Authors@R with the role cph, and the authors of that code are
  listed as contributors. inst/COPYRIGHTS names the derived files and
  reproduces the licence, and inst/AUTHORS describes who wrote what.
* The optional graphics processing unit (GPU) path needs the suggested
  'torch' package and its LibTorch backend; its tests and the GPU vignette's
  code are skipped when they are not available.
* The vignettes are written in Quarto, declared through VignetteBuilder and
  SystemRequirements.
* Examples and tests use at most two threads.
