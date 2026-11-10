run_loop <- function(x, centers, orders, ...) {
  st <- em_begin(nrow(x), centers, ...); B <- st$recipe$batch_size
  for (e in seq_along(st$recipe$epoch_rows)) {
    ord <- orders[[e]][seq_len(st$recipe$epoch_rows[e])]
    for (s in seq(1, length(ord), by = B))
      em_update(st, em_accumulate(st, x[ord[s:min(s + B - 1, length(ord))], , drop = FALSE]))
    em_end_epoch(st)
  }
  em_finalize(st)
}
test_that("each pass keeps the tail batch", {
  x <- matrix(c(0, 0, 0, 0, 10)); o <- list(1:5, 1:5)
  fit <- run_loop(x, matrix(0), o, nepochs = 2, batch_size = 3, alpha0 = 0.5)
  expect_identical(fit$centers, matrix(2)); expect_equal(fit$recipe$steps_per_pass, 2)
  expect_equal(fit$recipe$total_steps, 4)
})
test_that("an untouched centroid keeps its position across passes", {
  x <- matrix(0:3); init <- matrix(c(0, 100))
  expect_equal(run_loop(x, init, list(1:4), nepochs = 1, batch_size = 4, alpha0 = 1)$centers, matrix(c(1.5, 100)))
  expect_equal(run_loop(x, init, list(1:4, 1:4), nepochs = 2, batch_size = 4, alpha0 = 1)$centers, matrix(c(1.5, 100)))
})
test_that("revival starts on the fifth update", {
  x <- matrix(0, 10, 1); init <- matrix(c(0, 10)); o <- rep(list(1:10), 5)
  set.seed(1)
  expect_equal(run_loop(x, init, o, nepochs = 4, batch_size = 10, alpha0 = 0.5, unused_threshold = 0.1)$n_splits, 0)
  expect_equal(run_loop(x, init, o, nepochs = 5, batch_size = 10, alpha0 = 0.5, unused_threshold = 0.1)$n_splits, 1)
})
test_that("revival checks usage after the current batch update", {
  x <- matrix(c(0, 0, 0, 0, 10)); set.seed(1)
  fit <- run_loop(x, matrix(c(0, 10)), list(1:5), nepochs = 1, batch_size = 1, alpha0 = 0.5, unused_threshold = 0.1)
  expect_equal(fit$n_splits, 0)
})
test_that("closing mean, objective and manual recipe", {
  x <- matrix(c(0, 2, 10, 12))
  fit <- run_loop(x, matrix(c(0, 10)), list(1:4), nepochs = 1, batch_size = 4, alpha0 = 0.5)
  expect_identical(fit$centers, matrix(c(1, 11))); expect_equal(tail(fit$history$obj, 1), 8)
  expect_equal(fit$recipe$name, "manual"); expect_null(fit$cluster)
})
test_that("partial pieces share one snapshot and complete together", {
  set.seed(2); x <- matrix(rnorm(40), 20, 2); init <- x[1:3, ]
  st <- em_begin(20, init, nepochs = 1, batch_size = 20, alpha0 = 0.5)
  p <- em_accumulate(st, x[1:7, ]); expect_false(p$complete); expect_equal(p$n, 7)
  expect_error(em_update(st, p), class = "emeans_error_batch")
  q <- em_accumulate(st, x[8:20, ]); expect_true(q$complete)
  em_update(st, q); em_end_epoch(st)
  whole <- run_loop(x, init, list(1:20), nepochs = 1, batch_size = 20, alpha0 = 0.5)
  expect_identical(em_finalize(st)$centers, whole$centers)
})
test_that("sequencing and row-count rules raise classed errors", {
  x <- matrix(rnorm(20), 10, 2); st <- em_begin(10, x[1:2, ], nepochs = 1, batch_size = 4, alpha0 = 0.5)
  expect_error(em_accumulate(st, x[1:5, ]), class = "emeans_error_batch")
  expect_error(em_end_epoch(st), class = "emeans_error_sequence")
  expect_error(em_finalize(st), class = "emeans_error_sequence")
  expect_error(em_accumulate(st, x[, 1, drop = FALSE]), class = "emeans_error_input")
})
test_that("huge n keeps recipe arithmetic in doubles", {
  st <- em_begin(3e9, matrix(rnorm(8), 4, 2))
  expect_equal(st$recipe$epoch_rows, rep(3e9, 5)); expect_true(is.double(st$recipe$total_steps))
})
test_that("verbose header waits for the first batch and lines follow upstream", {
  x <- matrix(c(0, 2, 10, 12))
  expect_silent(st <- em_begin(4, matrix(c(0, 10)), nepochs = 1, batch_size = 4, alpha0 = 0.5, verbose = TRUE))
  msgs <- testthat::capture_messages(em_accumulate(st, x))
  expect_match(msgs[1], "^This is e-means\\. If you use our work")
  expect_match(msgs[2], "^Clustering 4 points in 1D to 2 clusters, 1 epochs \\(batch 4, alpha0 0\\.5, batch given\\)")
  em_update(st, em_accumulate(st, x[0, , drop = FALSE]))
  # Capture first: testthat 3.3.2's expect_match() evaluates `object` twice,
  # and a second em_end_epoch() is (correctly) a sequence error.
  msgs <- testthat::capture_messages(em_end_epoch(st))
  expect_match(msgs, "^  Epoch 1 / 1, iteration 1 / 1 \\([0-9.]+ s\\): objective=8")
  expect_error(em_end_epoch(st), class = "emeans_error_sequence")
})

# Behaviour decided for the port beyond the upstream ports above.
test_that("choose_splits draws nothing without a split and ranks donors by key", {
  expect_null(choose_splits(c(1, 1, 1), 0.1, draw = function(k) stop("drew")))
  expect_null(choose_splits(c(5, 0, 3), 0, draw = function(k) stop("drew")))
  sp <- choose_splits(c(5, 0.001, 3, 0, 1), 0.1, draw = function(k) rep(0.5, k))
  expect_identical(sp, list(source = c(1L, 3L), target = c(2L, 4L)))
})
test_that("revival draws k uniforms only at checks that split", {
  rec <- new.env(); withr::local_options(emeans.rng_recorder = rec); set.seed(1)
  fit <- run_loop(matrix(0, 10, 1), matrix(c(0, 10)), rep(list(1:10), 5), nepochs = 5,
                  batch_size = 10, alpha0 = 0.5, unused_threshold = 0.1)
  expect_identical(vapply(rec$draws, function(d) d$type, ""), "revival")
  expect_length(rec$draws[[1]]$value, 2); expect_equal(fit$history$nsplit, c(0, 0, 0, 0, 1))
})
test_that("em_begin validates its inputs", {
  m <- matrix(rnorm(8), 4, 2)
  expect_error(em_begin(3, m), class = "emeans_error_input")
  expect_error(em_begin(10.5, m), class = "emeans_error_input")
  expect_error(em_begin(10, rbind(m, NA)), class = "emeans_error_input")
  expect_error(em_begin(10, m, nepochs = 0), class = "emeans_error_input")
  expect_error(em_begin(10, m, batch_size = 0), class = "emeans_error_input")
  expect_error(em_begin(10, m, alpha0 = 1.5), class = "emeans_error_input")
  expect_error(em_begin(10, m, spherical = NA), class = "emeans_error_input")
  expect_error(em_begin(10, m, batch_law = "fast"), class = "emeans_error_input")
  expect_error(em_begin(10, m, device = "tpu"), class = "emeans_error_device")
  expect_s3_class(em_begin(10, matrix(1:8, 4)), "em_state")
})
test_that("a rejected call leaves the state usable; a failure part-way does not", {
  x <- matrix(rnorm(20), 10, 2)
  st <- em_begin(10, x[1:2, ], nepochs = 1, batch_size = 10, alpha0 = 0.5)
  expect_error(em_accumulate(st, rbind(x[1:2, ], NA)), class = "emeans_error_input")
  s <- em_accumulate(st, x)
  expect_error(em_accumulate(st, x[1, , drop = FALSE]), class = "emeans_error_batch")
  em_update(st, s); expect_error(em_update(st, s), class = "emeans_error_sequence")
  em_end_epoch(st); expect_identical(em_finalize(st), em_finalize(st))
  st <- em_begin(10, x[1:2, ], nepochs = 1, batch_size = 10, alpha0 = 0.5)
  s <- em_accumulate(st, x)
  local_mocked_bindings(ema_update_r = function(...) stop("boom"))
  expect_error(em_update(st, s), "boom")
  expect_error(em_accumulate(st, x), "create a new state", class = "emeans_error_sequence")
})
test_that("labels come only in the last epoch and describe the rows passed", {
  x <- matrix(c(0, 2, 10, 12))
  st <- em_begin(4, matrix(c(0, 10)), nepochs = 2, batch_size = 4, alpha0 = 0.5)
  a <- em_accumulate(st, x); expect_null(a$labels); em_update(st, a); em_end_epoch(st)
  b <- em_accumulate(st, x[1:2, , drop = FALSE]); expect_identical(b$labels, c(1L, 1L)); expect_null(b$acc)
  b <- em_accumulate(st, x[3:4, , drop = FALSE]); expect_identical(b$labels, c(2L, 2L))
  expect_identical(em_accumulate(st, x[0, , drop = FALSE])$labels, integer(0))
  expect_output(print(st), "epoch 2 / 2, step 1 / 2, batch 4")
  em_update(st, b); em_end_epoch(st); fit <- em_finalize(st)
  expect_identical(fit$size, c(2L, 2L)); expect_identical(fit$iter, 2)
  expect_named(fit$history, c("epoch", "rows", "obj", "time", "imbalance", "nsplit"))
  expect_equal(fit$history$epoch, c(1, 2)); expect_equal(fit$history$imbalance, c(1, 1))
})
test_that("spherical starts from unit centers; recipe records execution", {
  st <- em_begin(10, rbind(c(u = 3, v = 4), c(0, 0)), spherical = TRUE)
  expect_equal(em_centers(st), rbind(c(u = 0.6, v = 0.8), c(0, 0)))
  expect_true(st$recipe$spherical); expect_identical(st$recipe$device, "cpu")
  expect_identical(st$recipe$backend, "cpu"); expect_identical(st$recipe$chunk_size, 4096L)
})
test_that("verbose lines follow upstream formats", {
  st <- em_begin(1000, matrix(rnorm(160), 8, 20), nepochs = 1.5, threads = 1)
  expect_match(capture_messages(report_header(st))[2],
               "1\\.5 epochs \\(batch 32, alpha0 0\\.306, throughput batch law, auto for d > 16\\)")
  expect_match(capture_messages(report_preprocessing(st, 0.0123, "random")),
               "^  Preprocessing in 0\\.01 s \\(random initialization from the data; assignment: C\\+\\+ (direct|blas) kernel on CPU, 1 thread\\)")
  expect_match(capture_messages(report_epoch(st, list(time = 1.234, obj = 1234567.891), 1)),
               "^  Epoch 0\\.5 / 1\\.5 \\(500 of 1000 rows\\), iteration 16 / 48 \\(1\\.23 s\\): objective=1\\.23457e\\+06")
  expect_match(capture_messages(report_subsample(1e10, 4, 200, 50)),
               "^WARNING clustering 10000000000 points to 4 centroids: sampling 200 points \\(max_points_per_centroid=50\\)")
  st <- em_begin(1000, matrix(rnorm(8), 4, 2), batch_law = "throughput", spherical = TRUE)
  expect_match(capture_messages(report_header(st))[2], "throughput batch law, spherical\\)")
})
