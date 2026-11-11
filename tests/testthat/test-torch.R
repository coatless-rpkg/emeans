skip_if_not_installed("torch")
skip_if_not(has_torch(), "LibTorch is not installed")
test_that("R torch index conventions are what the backend assumes", {
  t <- torch::torch_zeros(3, 2); idx <- torch::torch_tensor(c(1L, 3L, 1L), dtype = torch::torch_long())
  t$index_add_(1, idx, torch::torch_ones(3, 2))
  expect_equal(as.matrix(t), rbind(c(2, 2), c(0, 0), c(1, 1)))
  expect_identical(as.integer(torch::torch_tensor(rbind(c(3, 1), c(0, 5)))$min(dim = 2)[[2]]), c(2L, 1L))
})
test_that("torch on CPU agrees with the C++ path, revival included", {
  # One cluster per well-separated blob keeps every final boundary far from
  # any point, so float32 and double cannot disagree on a near-tie; starting
  # all centers inside blob 1 forces revival.
  set.seed(1); mu <- matrix(rnorm(8 * 4, sd = 20), 8)
  x <- mu[rep(1:8, each = 200), ] + matrix(rnorm(1600 * 4), 1600); init <- x[1:8, ]
  set.seed(2); a <- emeans(x, 8, centers = init)
  set.seed(2); b <- emeans(x, 8, centers = init, device = "torch:cpu")
  expect_gt(a$n_splits, 0)
  expect_identical(b$cluster, a$cluster); expect_identical(b$n_splits, a$n_splits)
  expect_equal(b$centers, a$centers, tolerance = 1e-4)
  expect_identical(b$recipe$device, "torch:cpu")
  expect_identical(predict(b, x), predict(a, x))
})
test_that("torch spherical and em_*() work on the torch path", {
  set.seed(3); x <- matrix(rnorm(900), 300, 3); x <- x / sqrt(rowSums(x^2))
  b <- emeans(x, 5, spherical = TRUE, device = "torch:cpu")
  expect_identical(b$cluster, predict(b, x))
  expect_equal(b$tot.withinss, sum((x - b$centers[b$cluster, ])^2), tolerance = 1e-5)
  expect_equal(sqrt(rowSums(b$centers^2)), rep(1, 5), tolerance = 1e-5)
  st <- em_begin(300, x[1:5, ], nepochs = 1, batch_size = 300, device = "torch:cpu")
  em_update(st, em_accumulate(st, x)); em_end_epoch(st)
  expect_s3_class(em_finalize(st), "emeans")
})
test_that("MPS runs when available", {
  skip_if_not(torch::backends_mps_is_available())
  set.seed(4); x <- matrix(rnorm(4000), 1000, 4); fit <- emeans(x, 8, device = "mps")
  expect_identical(fit$recipe$device, "mps"); expect_true(all(is.finite(fit$centers)))
})
test_that("CUDA runs when available", {
  skip_if_not(torch::cuda_is_available())
  set.seed(5); fit <- emeans(matrix(rnorm(4000), 1000, 4), 8, device = "cuda")
  expect_identical(fit$recipe$device, "cuda")
})
