test_that("EMA update matches a hand computation", {
  up <- ema_update_r(rbind(c(0, 0), c(10, 10)), c(1, 1), rbind(c(4, 2), c(0, 0)), c(2, 0), 0.5, FALSE)
  expect_equal(up$centroids, rbind(c(4 / 3, 2 / 3), c(10, 10)))
  expect_equal(up$usage, c(1.5, 0.5))
})
test_that("alpha of one replaces touched usage and leaves untouched usage", {
  up <- ema_update_r(rbind(c(0, 0), c(10, 10)), c(3, 7), rbind(c(4, 2), c(0, 0)), c(2, 0), 1, TRUE)
  expect_identical(up$centroids[1, ], c(2, 1)); expect_identical(up$usage, c(2, 7))
})
test_that("weight one sets the exact mean", {
  up <- ema_update_r(matrix(5, 1, 1), 0, matrix(3, 1, 1), 3, 0.3, FALSE)
  expect_identical(up$centroids, matrix(1, 1, 1))
})
test_that("splits offset alternating coordinates and halve usage", {
  sp <- apply_splits_r(rbind(c(1, 2, 3), c(0, 0, 0)), c(4, 0), source = 1L, target = 2L)
  off <- 1e-4 * c(1, -1, 1)
  expect_equal(sp$centroids[1, ], c(1, 2, 3) * (1 - off))
  expect_equal(sp$centroids[2, ], c(1, 2, 3) * (1 + off))
  expect_equal(sp$usage, c(2, 2))
})
test_that("identical donor and child are separated on coordinate one", {
  child <- separate_splits(matrix(0, 1, 2), matrix(0, 1, 2))
  expect_identical(child[1, 1], cpp_nextafter(0, Inf)); expect_identical(child[1, 2], 0)
})
test_that("normalize_rows keeps zero rows and survives extremes", {
  m <- normalize_rows(rbind(c(3, 4), c(0, 0), c(1e300, 1e300)))
  expect_equal(m[1, ], c(0.6, 0.8)); expect_identical(m[2, ], c(0, 0))
  expect_equal(sqrt(sum(m[3, ]^2)), 1)
})
test_that("cpu backend handles k = 1 and d = 1 without dropping dimensions", {
  be <- cpu_backend("direct", 1L, 4096L, 1024L)
  data <- be$prepare_data(matrix(c(1, 2, 3), 3, 1))
  acc <- list(sums = be$zeros_matrix(1, 1), counts = be$zeros_vector(1), objective = 0)
  res <- be$accumulate(data, 1:3, matrix(0, 1, 1), acc, FALSE, TRUE)
  expect_identical(dim(res$acc$sums), c(1L, 1L)); expect_identical(res$labels, rep(1L, 3))
  up <- be$update(matrix(0, 1, 1), 1, res$acc$sums, res$acc$counts, 0.5, FALSE)
  expect_identical(dim(up$centroids), c(1L, 1L))
})

test_that("backend satisfies the interface check", {
  expect_no_error(check_backend(cpu_backend("direct", 1L, 4096L, 1024L)))
  expect_error(check_backend(list(name = "x")), class = "emeans_error_input")
})
test_that("prepare_data coerces integers and keeps doubles", {
  be <- cpu_backend("direct", 1L, 4096L, 1024L)
  expect_identical(be$prepare_data(matrix(0:3)), matrix(as.double(0:3)))
  x <- matrix(c(1, 2), 2, 1)
  expect_identical(be$prepare_data(x), x)
})
test_that("closing uses means where counted and raw positions elsewhere", {
  be <- cpu_backend("direct", 1L, 4096L, 1024L)
  cen <- rbind(c(1, 1), c(0, 2))
  expect_identical(be$closing(rbind(c(6, 8), c(0, 0)), c(2, 0), cen, FALSE), rbind(c(3, 4), c(0, 2)))
  expect_equal(be$closing(rbind(c(6, 8), c(0, 0)), c(2, 0), cen, TRUE), rbind(c(0.6, 0.8), c(0, 1)))
})
test_that("assign returns 1-based clusters and scores for all rows", {
  be <- cpu_backend("direct", 1L, 4096L, 1024L)
  data <- be$prepare_data(rbind(c(0, 0), c(9, 9), c(1, 0)))
  res <- be$assign(data, rbind(c(0, 0), c(10, 10)), FALSE)
  expect_identical(res$cluster, c(1L, 2L, 1L))
  expect_equal(res$score, c(0, 2, 1))
})
