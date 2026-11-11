# emeans 0.1.0

Initial CRAN release: an R port of the e-means clustering in faiss
(`contrib/e_means`, commit 2b648f0605f2ef3d14c4705350f8a0b214463878).

* `emeans()` fits e-means, a mini-batch alternative to k-means, and returns an
  `"emeans"` object with the same `cluster`, `centers`, `totss`, `withinss`,
  `tot.withinss`, `betweenss` and `size` as a `kmeans()` fit, from a closing
  sweep that labels every training row by its nearest final center. It supports Euclidean and spherical clustering, a
  `max_points_per_centroid` subsample, user-supplied starting centers and
  fractional `nepochs`. Randomness comes from R's RNG, so `set.seed()` makes a
  fit reproducible.
* `print()`, `summary()`, `predict()` and `fitted()` methods for `"emeans"`
  objects.
* Experimental `em_begin()`, `em_accumulate()`, `em_update()`,
  `em_end_epoch()`, `em_finalize()` and `em_centers()` run training one batch
  at a time, for data that does not fit in memory.
* C++ CPU kernels (Rcpp, RcppArmadillo, optional OpenMP) in double precision,
  with a direct kernel and a blocked BLAS kernel. `options(emeans.kernel =)`
  selects between them and `options(emeans.threads =)` sets the default thread
  count.
* Optional GPU path through the {torch} package (float32): `device = "mps"`
  for Apple GPUs, `"torch:cpu"` and an experimental `"cuda"` that has not
  yet been tested on NVIDIA hardware.
* `tidy()`, `glance()` and `augment()` methods for {generics}, following
  broom's columns for `kmeans()`, and an
  `"emeans"` engine for `tidyclust::k_means()`.
* Results were cross-checked pass by pass against the upstream Python
  implementation with replayed random draws; see `tools/crosscheck/`.
* Four vignettes: getting started, large data, tidymodels and GPU.
