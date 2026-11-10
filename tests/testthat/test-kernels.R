grid <- expand.grid(d = c(1L, 2L, 17L, 64L), k = c(1L, 3L, 50L))
test_that("both kernels match the brute-force reference", {
  for (g in seq_len(nrow(grid))) {
    set.seed(g); d <- grid$d[g]; k <- grid$k[g]
    x <- matrix(rnorm(300 * d), 300, d); cen <- x[sample.int(300, k), , drop = FALSE]
    rows <- sample.int(300, 200); ref <- ref_assign(x, rows, cen)
    for (kern in c("direct", "blas")) {
      got <- cpp_assign_accumulate(x, rows, cen, FALSE, kern, 64L, 16L, 1L,
                                   matrix(0, k, d), numeric(k), 0)
      expect_identical(got$labels, ref$labels)
      expect_equal(got$sums, ref$sums, tolerance = 1e-12)
      expect_equal(got$counts, ref$counts); expect_equal(got$objective, ref$objective, tolerance = 1e-10)
    }
  }
})
test_that("direct kernel is bit-identical across threads and chunk sizes", {
  set.seed(9); x <- matrix(rnorm(2000 * 5), 2000, 5); cen <- x[1:40, ]; rows <- sample.int(2000)
  run <- function(th, cs, ccs) cpp_assign_accumulate(x, rows, cen, FALSE, "direct", cs, ccs, th,
                                                    matrix(0, 40, 5), numeric(40), 0)
  base <- run(1L, 4096L, 1024L)
  for (args in list(list(2L, 4096L, 1024L), list(1L, 7L, 1024L), list(2L, 1L, 5L)))
    expect_identical(do.call(run, args), base)
})
test_that("blas kernel is bit-identical across threads", {
  set.seed(10); x <- matrix(rnorm(2000 * 32), 2000, 32); cen <- x[1:40, ]; rows <- seq_len(2000)
  run <- function(th) cpp_assign_accumulate(x, rows, cen, FALSE, "blas", 512L, 16L, th,
                                            matrix(0, 40, 32), numeric(40), 0)
  expect_identical(run(2L), run(1L))
})
test_that("continuing an accumulation equals one call", {
  set.seed(11); x <- matrix(rnorm(100 * 3), 100, 3); cen <- x[1:4, ]
  whole <- cpp_assign_accumulate(x, 1:100, cen, FALSE, "direct", 4096L, 1024L, 1L, matrix(0, 4, 3), numeric(4), 0)
  a <- cpp_assign_accumulate(x, 1:50, cen, FALSE, "direct", 4096L, 1024L, 1L, matrix(0, 4, 3), numeric(4), 0)
  b <- cpp_assign_accumulate(x, 51:100, cen, FALSE, "direct", 4096L, 1024L, 1L, a$sums, a$counts, a$objective)
  expect_identical(b$sums, whole$sums); expect_identical(b$counts, whole$counts)
  expect_identical(b$objective, whole$objective)
  expect_identical(c(a$labels, b$labels), whole$labels)
})
test_that("ties go to the lowest index and predict mode skips sums", {
  cen <- rbind(c(1, 0), c(0, 1), c(-1, 0), c(1, 0))
  got <- cpp_assign_accumulate(matrix(0, 1, 2), 1L, cen, FALSE, "blas", 1L, 1L, 1L, NULL, NULL, 0)
  expect_identical(got$labels, 1L); expect_null(got$sums); expect_equal(got$score, 1)
})
test_that("spherical scores are inner products", {
  x <- rbind(c(1, 0), c(0, -1)); cen <- rbind(c(0, 1), c(1, 0))
  got <- cpp_assign_accumulate(x, 1:2, cen, TRUE, "direct", 8L, 8L, 1L, matrix(0, 2, 2), numeric(2), 0)
  expect_identical(got$labels, c(2L, 2L)); expect_equal(got$score, c(1, 0))
  expect_equal(got$objective, (2 - 2 * 1) + (2 - 2 * 0))
})
test_that("thread and kernel resolution respect options and limits", {
  withr::local_options(emeans.threads = 2L)
  expect_identical(resolve_threads(), if (cpp_openmp_max_threads() == 0L) 1L else 2L)
  withr::local_envvar(OMP_THREAD_LIMIT = "1"); expect_identical(resolve_threads(8L), 1L)
  expect_identical(resolve_kernel("direct", 64L), "direct")
  testthat::local_mocked_bindings(blas_name = function() "")
  expect_identical(resolve_kernel("auto", 2L), "direct")
  expect_error(resolve_kernel("gpu", 2L), class = "emeans_error_input")
})

test_that("chunk sizes default once and reject bad values", {
  expect_identical(resolve_chunk_sizes(), list(chunk_size = 4096L, centroid_chunk_size = 1024L))
  expect_identical(resolve_chunk_sizes(10, 5L), list(chunk_size = 10L, centroid_chunk_size = 5L))
  for (bad in list(0, -1, 1.5, NA_real_, Inf, c(1, 2), "a"))
    expect_error(resolve_chunk_sizes(bad), class = "emeans_error_input")
  expect_error(resolve_chunk_sizes(NULL, 0L), class = "emeans_error_input")
})

test_that("auto kernel uses blas only for a named optimised BLAS", {
  choose <- function(name, d) {
    testthat::local_mocked_bindings(blas_name = function() name)
    resolve_kernel("auto", d)
  }
  reference <- c("", "/x/libRblas.0.dylib", "/usr/lib/x86_64-linux-gnu/blas/libblas.so.3")
  optimised <- c("/System/Library/Frameworks/Accelerate.framework/Versions/A/libBLAS.dylib",
                 "/usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3")
  for (nm in reference) {
    expect_identical(choose(nm, 64L), "direct")
    expect_identical(choose(nm, 2L), "direct")
  }
  for (nm in optimised) {
    expect_identical(choose(nm, 64L), "blas")
    expect_identical(choose(nm, 2L), "blas")
    expect_identical(choose(nm, 16L), "blas")
    expect_identical(choose(nm, 1L), "direct")
  }
})
