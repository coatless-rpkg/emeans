test_that("emeans equals an em_*() loop fed the same order", {
  set.seed(123); x <- matrix(rnorm(17 * 3), 17, 3); init <- x[1:4, ]
  set.seed(7); fit <- emeans(x, 4, centers = init, nepochs = 3, batch_size = 4, alpha0 = 0.2, chunk_size = 2L)
  set.seed(7); st <- em_begin(17, init, nepochs = 3, batch_size = 4, alpha0 = 0.2)
  for (e in 1:3) { p <- sample.int(17)
    for (s in seq(1, 17, by = 4)) em_update(st, em_accumulate(st, x[p[s:min(s + 3, 17)], , drop = FALSE]))
    em_end_epoch(st) }
  res <- em_finalize(st)
  expect_identical(res$centers, fit$centers); expect_identical(res$history$obj, fit$history$obj)
  expect_identical(res$n_splits, fit$n_splits)
})
test_that("results are reproducible under set.seed and invariant to threads and chunks", {
  set.seed(1); x <- matrix(rnorm(3000 * 4), 3000, 4)
  withr::local_options(emeans.kernel = "direct")
  set.seed(5); a <- emeans(x, 20, threads = 1L)
  set.seed(5); b <- emeans(x, 20, threads = 2L, chunk_size = 33L, centroid_chunk_size = 3L)
  expect_identical(a$centers, b$centers); expect_identical(a$cluster, b$cluster)
})
test_that("the result has the documented shape", {
  set.seed(2); x <- matrix(rnorm(500 * 2), 500, 2, dimnames = list(NULL, c("u", "v")))
  fit <- emeans(x, 5)
  expect_s3_class(fit, "emeans"); expect_identical(dim(fit$centers), c(5L, 2L))
  expect_identical(colnames(fit$centers), c("u", "v")); expect_length(fit$cluster, 500)
  expect_identical(sum(fit$size), 500L); expect_identical(fit$iter, fit$recipe$total_steps)
  expect_named(fit$history, c("epoch", "rows", "obj", "time", "imbalance", "nsplit"))
  expect_identical(fit$recipe$upstream_commit, "2b648f0605f2ef3d14c4705350f8a0b214463878")
  expect_null(emeans(x, 5, keep_cluster = FALSE)$cluster)
})
test_that("the closing sweep gives kmeans()-style labels and sums of squares", {
  set.seed(11); x <- matrix(rnorm(3000), 1000, 3)
  set.seed(12); fit <- emeans(x, 4)
  expect_identical(names(fit)[1:7], c("cluster", "centers", "totss", "withinss",
                                      "tot.withinss", "betweenss", "size"))
  expect_identical(fit$cluster, predict(fit, x))
  expect_identical(fit$size, tabulate(fit$cluster, 4))
  direct <- vapply(1:4, function(j) {
    sum(sweep(x[fit$cluster == j, , drop = FALSE], 2, fit$centers[j, ])^2)
  }, numeric(1))
  expect_equal(fit$withinss, direct)
  expect_equal(fit$tot.withinss, sum(direct))
  expect_equal(fit$totss, sum(sweep(x, 2, colMeans(x))^2))
  expect_equal(fit$betweenss, fit$totss - fit$tot.withinss)
  set.seed(12); bare <- emeans(x, 4, keep_cluster = FALSE)
  expect_identical(bare$centers, fit$centers)
  for (field in c("cluster", "totss", "withinss", "tot.withinss", "betweenss")) {
    expect_null(bare[[field]])
  }
})
test_that("sums of squares cover the training rows and use Euclidean distance", {
  set.seed(13); x <- matrix(rnorm(2000, mean = 2), 1000, 2)
  set.seed(14); sub <- emeans(x, 4, max_points_per_centroid = 50)
  used <- which(!is.na(sub$cluster)); xs <- x[used, ]
  expect_identical(sum(sub$size), 200L)
  expect_equal(sub$totss, sum(sweep(xs, 2, colMeans(xs))^2))
  expect_equal(sub$tot.withinss, sum((xs - sub$centers[sub$cluster[used], ])^2))
  set.seed(15); sph <- emeans(x, 3, spherical = TRUE)
  expect_identical(sph$cluster, predict(sph, x))
  expect_equal(sph$tot.withinss, sum((x - sph$centers[sph$cluster, ])^2))
})
test_that("max_points_per_centroid trains on a subset and labels the rest NA", {
  set.seed(3); x <- matrix(rnorm(1000 * 2), 1000, 2)
  set.seed(30); fit <- emeans(x, 4, max_points_per_centroid = 50)
  expect_identical(sum(!is.na(fit$cluster)), 200L); expect_equal(fit$recipe$epoch_rows[1], 200)
  set.seed(30); keep <- sort(sample.int(1000, 200)); expect_identical(which(!is.na(fit$cluster)), keep)
})
test_that("input validation", {
  expect_error(emeans(matrix(c(1, NA, 3, 4), 2), 1), class = "emeans_error_input")
  expect_error(emeans(matrix(1:6, 3), 4), class = "emeans_error_input")
  expect_error(emeans(data.frame(a = 1:4, b = letters[1:4]), 2), "b", class = "emeans_error_input")
  set.seed(4); expect_s3_class(emeans(matrix(1:40, 20), 2), "emeans")
  expect_error(emeans(matrix(rnorm(20), 10), 2, centers = matrix(0, 3, 2)), class = "emeans_error_input")
})
test_that("identical rows, k = 1 and d = 1 train without error", {
  set.seed(6); fit <- emeans(matrix(1, 200, 3), 8)
  expect_true(all(is.finite(fit$centers))); expect_identical(dim(fit$centers), c(8L, 3L))
  expect_identical(dim(emeans(matrix(rnorm(100), 100, 1), 3)$centers), c(3L, 1L))
  expect_identical(unique(emeans(matrix(rnorm(100), 50, 2), 1)$cluster), 1L)
})
test_that("spherical mode keeps unit centers and reports similarity objective", {
  set.seed(8); x <- matrix(rnorm(600 * 3), 600, 3); x <- x / sqrt(rowSums(x^2))
  fit <- emeans(x, 6, spherical = TRUE)
  expect_equal(sqrt(rowSums(fit$centers^2)), rep(1, 6)); expect_true(fit$spherical)
})
test_that("emeans finds separated blobs at least as well as kmeans", {
  set.seed(10); mu <- matrix(c(0, 0, 10, 10, -10, 10, 10, -10), 4, byrow = TRUE)
  x <- mu[rep(1:4, each = 500), ] + matrix(rnorm(4000, sd = 0.5), 2000)
  fit <- emeans(x, 4); ss <- function(cen, cl) sum((x - cen[cl, ])^2)
  km <- kmeans(x, 4, nstart = 1)
  expect_lte(ss(fit$centers, predict(fit, x)), 1.05 * km$tot.withinss)
})
test_that("a torch device without {torch} is a device error", {
  local_mocked_bindings(has_torch = function() FALSE)
  expect_error(emeans(matrix(rnorm(20), 10), 2, device = "cuda"), class = "emeans_error_device")
})
test_that("loading {torch} does not move R's random-number stream", {
  # Loading the torch namespace draws from R's RNG; has_torch() must undo
  # that, or the first torch fit in a session gets different permutations.
  local_mocked_bindings(torch_namespace = function() { stats::runif(3); FALSE })
  set.seed(1); expected <- runif(1)
  set.seed(1); expect_false(has_torch()); expect_identical(runif(1), expected)
})
test_that("centers take their column names from x", {
  set.seed(3)
  x <- matrix(rnorm(200), 100, 2, dimnames = list(NULL, c("a", "b")))
  start <- x[1:3, ]
  unnamed <- unname(start)
  fit <- emeans(x, 3, centers = unnamed, nepochs = 2, threads = 1)
  expect_identical(colnames(fit$centers), c("a", "b"))

  set.seed(4); ref <- emeans(x, 3, centers = start, nepochs = 2, threads = 1)
  rev_start <- start[, c("b", "a")]
  set.seed(4); fit_rev <- emeans(x, 3, centers = rev_start, nepochs = 2, threads = 1)
  expect_equal(fit_rev$centers, ref$centers)
  expect_identical(colnames(fit_rev$centers), c("a", "b"))

  bad <- start; colnames(bad) <- c("a", "z")
  expect_error(emeans(x, 3, centers = bad, threads = 1), class = "emeans_error_input")
})
test_that("emeans draws random numbers in the documented order", {
  draw_types <- function(...) {
    rec <- new.env()
    withr::local_options(emeans.rng_recorder = rec)
    emeans(..., threads = 1)
    vapply(rec$draws, function(d) d$type, "")
  }
  set.seed(5); x <- matrix(rnorm(120), 60, 2)
  set.seed(6)
  types <- draw_types(x, 3, nepochs = 3, max_points_per_centroid = 5)
  expect_identical(types[1:2], c("subset", "init"))
  expect_identical(sum(types == "permutation"), 3L)
  expect_identical(match("permutation", types), 3L)
  expect_true(all(types[-(1:2)] %in% c("permutation", "revival")))
  rest <- types[-(1:2)]
  expect_identical(rest[1], "permutation")

  set.seed(7)
  types <- draw_types(x, 3, nepochs = 2, centers = x[1:3, ])
  expect_false("init" %in% types)
  expect_identical(types[types != "revival"], c("permutation", "permutation"))

  set.seed(8)
  types <- draw_types(x[1:3, ], 3, nepochs = 2)
  expect_false("init" %in% types)
  expect_identical(types[types != "revival"], c("permutation", "permutation"))
})
