# Ports of upstream faiss e-means behavioural tests at the pinned commit
# 2b648f0605f2ef3d14c4705350f8a0b214463878: the NumPy reference contracts in
# tests/e_means_test_utils.py and two tests from tests/test_e_means.py. Each
# test_that() is named after its upstream method, without "test_".
#
# Translation: set.seed() before a call replaces upstream's `seed=` and its
# per-epoch RandomState seeds, so a caller loop replays emeans()'s draws with
# one sample.int(n) per epoch. Labels are 1-based, with NA for upstream's -1.
# assignment_chunk_size is chunk_size. begin, accumulate, update, end_epoch,
# finalize and current_centroids are em_begin(), em_accumulate(),
# em_update(), em_end_epoch(), em_finalize() and em_centers(); train() is
# emeans(); assign() is predict(). Upstream's private fields (`_usage`,
# `_centroids`, `_pass_sums`, ...) are the fields of the em_state
# environment. The CPU path is double precision (see ?emeans, "Differences from upstream"), so
# upstream's FP32 tolerances become exact or 1e-12 checks.
#
# Skipped upstream tests, and why:
# tests/e_means_test_utils.py
# - single_run_guard_preserves_final_result_and_failed_state: Python object
#   ownership (one run per object, retrain guard). The failed-state rule is
#   in test-em-loop.R.
# - spherical_deferred_ema_borrowed_cache_and_chunks: identity and version
#   counter of the borrowed readout cache (Python object ownership).
# - spherical_torch_version_and_inference_invalidation: torch internals.
# - max_points_per_centroid_trains_on_a_seeded_subset: upstream `seed=`
#   handling (default_rng, tensor seeds -1 and 2^40) and an injected failure
#   (object lifetime). The subset contract is in test-emeans.R.
# - pending_execution_failure_requires_new_run: Python object lifetime after
#   an injected failure. The R equivalent is in test-em-loop.R.
# - external_statistics_and_result_ownership: borrowed views, writeable flags
#   and data pointers (Python object ownership).
# - fractional_near_tie_selected_distance: FP32 widening (expanded-L2
#   cancellation at float32 near ties).
# tests/test_e_means.py
# - TestEmeansRecipe (fractional_epoch_rounding, reference_values,
#   batch_size_alone_takes_the_step_law, auto_batch_law,
#   auto_unused_threshold, recipe_records_resolved_defaults, default_nepochs,
#   derived_recipe_metadata, schedule_uses_logical_steps): ported in
#   test-recipe.R.
# - verbose_cites_and_reports_epochs_and_iterations,
#   caller_driven_header_waits_for_the_first_batch: ported in test-em-loop.R.
# - silent_without_verbose: outside this task's port list; verbose output
#   goes through message() only.
# - closed_stdout_never_stops_training: Python stdout handling; the port
#   never writes to stdout.
# - TestAddRowsAt.equals_add_at: NumPy/SciPy helper `_add_rows_at`. The C++
#   kernel accumulates in row order (test-kernels.R).
# - numpy_training_does_not_import_torch: Python optional-import isolation
#   (torch internals).
# - large_batch_sums_and_closing_mean: FP32 widening of sums past 2^24 rows.
# - closing_mean_and_assign, untouched_centroid_retains_its_position,
#   caller_batches_match_matrix_training, revival_starts_on_fifth_update,
#   revival_follows_current_batch_update, each_pass_uses_tail: ported in
#   test-em-loop.R (predict() in test-methods.R).
# - train_tuple_capture_opt_out_and_owned_lifetime: Python object ownership;
#   keep_cluster = FALSE is in test-emeans.R.
# - resident_capture_memory_is_bounded_by_assignment_chunks: Python
#   tracemalloc memory witness.
# - fresh_objects_are_deterministic: set.seed() reproducibility, in
#   test-emeans.R.
# - training_does_not_advance_global_rng: R's RNG drives training
#   (see ?emeans, "Differences from upstream").

# Exact comparisons below assume the direct kernel; tests that cover both
# kernels set the option themselves.
withr::local_options(emeans.kernel = "direct")

kernels <- c("direct", "blas")

# Upstream applies caller-made statistics with update(dict). em_update()
# takes only em_accumulate() results, so complete the batch with placeholder
# rows and swap in the statistics; em_update() applies the batch's pending
# accumulator.
update_with_statistics <- function(state, sums, counts, objective) {
  stats <- em_accumulate(state, matrix(0, local_batch_size(state), state$d))
  state$pending$acc <- list(sums = sums, counts = counts, objective = objective)
  em_update(state, stats)
}

# Feed `x` to a fresh state in emeans()'s order: one sample.int(n) per epoch,
# whose first epoch_rows[e] rows go in batches. Returns the fit and the
# last-epoch labels (NA for rows the last epoch does not visit).
caller_run <- function(x, centers, ...) {
  st <- em_begin(nrow(x), centers, ...)
  batch <- st$recipe$batch_size
  epochs <- length(st$recipe$epoch_rows)
  labels <- rep(NA_integer_, nrow(x))
  for (e in seq_len(epochs)) {
    perm <- sample.int(nrow(x))
    rows <- st$recipe$epoch_rows[e]
    for (start in seq(1, rows, by = batch)) {
      ids <- perm[start:min(rows, start + batch - 1)]
      stats <- em_accumulate(st, x[ids, , drop = FALSE])
      if (e == epochs) {
        labels[ids] <- stats$labels
      }
      em_update(st, stats)
    }
    em_end_epoch(st)
  }
  list(fit = em_finalize(st), labels = labels)
}

# Nearest center by direct squared L2 distance; ties go to the lowest index.
nearest_l2 <- function(x, centers) {
  d2 <- matrix(
    vapply(seq_len(nrow(centers)),
           function(j) rowSums(sweep(x, 2, centers[j, ])^2),
           numeric(nrow(x))),
    nrow(x)
  )
  apply(d2, 1, which.min)
}

test_that("spherical_ip_zero_objective_and_represented_statistics", {
  initial <- rbind(c(0, 0), c(1, 0), c(-1, 0))
  rows <- rbind(c(0, 0), c(0.25, sqrt(15) / 4), c(1, 0), c(-1, 0), c(0, 1),
                c(0, -1))
  expected_labels <- c(1L, 2L, 2L, 3L, 1L, 1L)
  sums <- unname(rowsum(rows, expected_labels))
  for (kernel in kernels) {
    withr::local_options(emeans.kernel = kernel)
    source <- initial
    st <- em_begin(6, source, nepochs = 1, batch_size = 6, alpha0 = 1,
                   spherical = TRUE, chunk_size = 2)
    stats <- em_accumulate(st, rows)
    expect_identical(stats$labels, expected_labels)
    expect_identical(stats$acc$counts, c(3, 2, 1))
    expect_identical(stats$acc$sums, sums)
    expect_equal(stats$acc$objective, 7.5, tolerance = 1e-12)
    em_update(st, stats)
    em_end_epoch(st)
    fit <- em_finalize(st)
    expect_identical(fit$centers[1, ], c(0, 0))
    scores <- rows %*% t(fit$centers)
    expect_identical(predict(fit, rows), apply(scores, 1, which.max))
    expect_equal(predict(fit, rows, type = "distance"), apply(scores, 1, max),
                 tolerance = 1e-12)
    expect_identical(source, initial)
    expect_equal(fit$history$obj[1], 7.5, tolerance = 1e-12)
  }
})

test_that("spherical_projection_extremes_and_wide_closing", {
  expected <- rbind(c(2^-0.5, -(2^-0.5)), c(1, 0), c(0, 0))
  # Upstream's FP32 extremes, then their double counterparts (double precision):
  # squaring 3e300 overflows a double and squaring 1e-300 underflows one.
  extremes <- list(rbind(c(3e38, -3e38), c(1e-38, 0), c(0, 0)),
                   rbind(c(3e300, -3e300), c(1e-300, 0), c(0, 0)))
  for (source in extremes) {
    saved <- source
    output <- normalize_rows(source)
    expect_equal(output, expected, tolerance = 1e-12)
    expect_identical(source, saved)
    st <- em_begin(3, source, nepochs = 1, batch_size = 3, alpha0 = 1,
                   spherical = TRUE)
    expect_equal(em_centers(st), output, tolerance = 1e-12)
  }
  basis <- rbind(c(1, 0), c(0, 1), c(-1, 0))
  st <- em_begin(3, basis, nepochs = 1, batch_size = 3, alpha0 = 1,
                 spherical = TRUE)
  em_update(st, em_accumulate(st, basis))
  em_end_epoch(st)
  # Direct finalize seam, as upstream: a mean of 1e300 must be projected
  # without squaring it (which overflows in double too).
  st$pass_sums <- rbind(c(1e300, -1e300), c(0, 0), c(0, 0))
  st$pass_counts <- c(1, 1, 0)
  expect_equal(em_finalize(st)$centers,
               rbind(c(2^-0.5, -(2^-0.5)), c(0, 0), c(-1, 0)),
               tolerance = 1e-12)
  # Upstream's `_unit_centroids is None` has no analogue: the port keeps no
  # readout cache.
})

test_that("spherical_zero_beats_negative_similarity", {
  for (kernel in kernels) {
    withr::local_options(emeans.kernel = kernel)
    st <- em_begin(2, rbind(c(0, 0), c(1, 0)), nepochs = 1, batch_size = 2,
                   alpha0 = 1, spherical = TRUE)
    stats <- em_accumulate(st, rbind(c(-1, 0), c(0, 0)))
    expect_identical(stats$labels, c(1L, 1L))
    expect_identical(stats$acc$objective, 4)
  }
})

test_that("spherical_fractional_caller_and_matrix", {
  set.seed(73)
  rows <- matrix(rnorm(31 * 5), 31, 5)
  rows <- rows / sqrt(rowSums(rows^2))
  for (budget in c(0.5, 1.5)) {
    args <- list(nepochs = budget, spherical = TRUE, batch_size = 7,
                 alpha0 = 0.2, chunk_size = 3)
    set.seed(7)
    matrix_fit <- do.call(emeans, c(list(rows, 4, centers = rows[1:4, ]), args))
    set.seed(7)
    caller <- do.call(caller_run, c(list(rows, rows[1:4, ]), args))
    expect_identical(matrix_fit$centers, caller$fit$centers)
    # emeans() relabels every row against the final centers.
    expect_identical(matrix_fit$cluster, predict(caller$fit, rows))
    expect_identical(matrix_fit$history$obj, caller$fit$history$obj)
  }
})

test_that("fractional_epochs_preserve_batches_and_closing_labels", {
  cases <- list(list(2, c(31, 31)), list(2.5, c(31, 16, 31)),
                list(1.25, c(8, 31)), list(0.5, 16), list(0.1, 3))
  x <- matrix(0:30 / 4)
  initial <- x[c(1, 11, 21, 31), , drop = FALSE]
  for (case in cases) {
    budget <- case[[1]]
    epoch_rows <- case[[2]]
    args <- list(nepochs = budget, batch_size = 7, alpha0 = 0.2, chunk_size = 3)
    set.seed(7)
    matrix_fit <- do.call(emeans, c(list(x, 4, centers = initial), args))
    set.seed(7)
    st <- do.call(em_begin, c(list(nrow(x), initial), args))
    steps <- ceiling(epoch_rows / 7)
    expect_identical(matrix_fit$recipe$epoch_rows, epoch_rows)
    expect_identical(matrix_fit$recipe$epoch_steps, steps)
    expect_identical(matrix_fit$recipe$total_steps, sum(steps))
    # The last pass's labels come back from em_accumulate(), piece by piece.
    captured <- rep(NA_integer_, nrow(x))
    for (e in seq_along(epoch_rows)) {
      perm <- sample.int(nrow(x))
      for (start in seq(1, epoch_rows[e], by = 7)) {
        idx <- perm[start:min(start + 6, epoch_rows[e])]
        batch <- x[idx, , drop = FALSE]
        frozen <- em_centers(st)
        first <- seq_len(min(2, nrow(batch)))
        stats <- em_accumulate(st, batch[first, , drop = FALSE])
        if (!is.null(stats$labels)) captured[idx[first]] <- stats$labels
        if (nrow(batch) > 2) {
          stats <- em_accumulate(st, batch[-(1:2), , drop = FALSE])
          if (!is.null(stats$labels)) captured[idx[-(1:2)]] <- stats$labels
        }
        expect_identical(em_centers(st), frozen)
        em_update(st, stats)
      }
      expect_identical(st$epoch_steps_done, steps[e])
      em_end_epoch(st)
    }
    # emeans() does not expose its state; its recipe and the caller's step
    # counter stand in for upstream's matrix._step.
    expect_identical(st$step, sum(steps))
    expect_identical(matrix_fit$iter, sum(steps))
    caller_fit <- em_finalize(st)
    expect_identical(caller_fit$centers, matrix_fit$centers)
    expect_identical(caller_fit$history$obj, matrix_fit$history$obj)
    expect_identical(matrix_fit$cluster, predict(matrix_fit, x))
    visited <- perm[seq_len(epoch_rows[length(epoch_rows)])]
    expect_true(all(is.na(captured[-visited])))
    expect_false(anyNA(captured[visited]))
    for (centroid in 1:4) {
      assigned <- visited[captured[visited] == centroid]
      if (length(assigned) > 0L) {
        expect_equal(matrix_fit$centers[centroid, ], mean(x[assigned, ]),
                     tolerance = 1e-12)
      }
    }
    counts <- tabulate(captured[visited], 4)
    expect_identical(matrix_fit$history$imbalance[length(epoch_rows)],
                     4 * sum(counts * counts) / length(visited)^2)
  }
})

test_that("supplied_chunks_share_one_frozen_batch", {
  rows <- rbind(c(0.25, 0), c(10, 10.25), c(0.5, 0.25), c(10.25, 10),
                c(0, 0.25), c(10.5, 10.25), c(0.5, 0), c(10, 10.5),
                c(0.25, 0.5), c(10.25, 10.5), c(0.5, 0.5), c(10.5, 10.5),
                c(0, 0))
  none <- rows[0, , drop = FALSE]
  initial <- rbind(c(0, 0), c(10, 10))
  args <- list(nepochs = 2, batch_size = 4, alpha0 = 0.3, chunk_size = 2)
  whole <- do.call(em_begin, c(list(nrow(rows), initial), args))
  chunked <- do.call(em_begin, c(list(nrow(rows), initial), args))
  empty <- em_accumulate(chunked, none)
  expect_identical(empty$n, 0)
  # Adapted: an em_stats object carries statistics only for a complete
  # batch, so the zero counts of an empty first call are read from the
  # batch accumulator in the state.
  expect_null(empty$acc)
  expect_identical(chunked$pending$acc$counts, c(0, 0))
  for (epoch in 1:2) {
    for (start in seq(1, nrow(rows), by = 4)) {
      batch <- rows[start:min(start + 3, nrow(rows)), , drop = FALSE]
      reference <- em_accumulate(whole, batch)
      frozen <- em_centers(chunked)
      count <- 0
      for (offset in seq(1, nrow(batch), by = 2)) {
        part <- batch[offset:min(offset + 1, nrow(batch)), , drop = FALSE]
        stats <- em_accumulate(chunked, part)
        count <- count + nrow(part)
        expect_identical(stats$n, count)
        expect_identical(em_centers(chunked), frozen)
        expect_identical(chunked$step, whole$step)
        if (count < nrow(batch)) {
          expect_null(stats$acc)
        }
        if (epoch == 2) {
          expect_identical(stats$labels, nearest_l2(part, frozen))
        } else {
          expect_null(stats$labels)
        }
        empty <- em_accumulate(chunked, none)
        expect_identical(empty$n, count)
        if (count < nrow(batch)) {
          expect_null(empty$acc)
        } else {
          expect_identical(empty$acc, stats$acc)
        }
        if (epoch == 2) {
          expect_identical(empty$labels, integer(0))
        }
      }
      expect_identical(stats$acc, reference$acc)
      em_update(whole, reference)
      em_update(chunked, stats)
    }
    em_end_epoch(whole)
    em_end_epoch(chunked)
  }
  chunked_fit <- em_finalize(chunked)
  whole_fit <- em_finalize(whole)
  expect_identical(chunked_fit$centers, whole_fit$centers)
  expect_identical(chunked_fit$history$obj, whole_fit$history$obj)
  expect_identical(chunked_fit$n_splits, whole_fit$n_splits)
  expect_null(chunked_fit$cluster)
})

test_that("last_pass_labels_are_owned_and_not_final_nearest", {
  rows <- matrix(c(0, 1, 49, 51, 52, 53))
  initial <- matrix(c(0, 100))
  # Upstream's contract holds for the loop: the last-pass labels are the
  # caller's own copy and need not match the nearest final center.
  st <- em_begin(6, initial, nepochs = 1, batch_size = 6, alpha0 = 0.5)
  stats <- em_accumulate(st, rows)
  labels <- stats$labels
  em_update(st, stats)
  em_end_epoch(st)
  looped <- em_finalize(st)
  expect_identical(labels, c(1L, 1L, 1L, 2L, 2L, 2L))
  expect_equal(looped$centers, matrix(c(50 / 3, 52)), tolerance = 1e-12)
  expect_identical(predict(looped, rows)[3], 2L)
  # An unfinished run elsewhere does not change the caller's labels.
  other <- em_begin(6, initial, nepochs = 1, batch_size = 6, alpha0 = 0.5)
  em_accumulate(other, rows[1:2, , drop = FALSE])
  expect_identical(labels, c(1L, 1L, 1L, 2L, 2L, 2L))
  # emeans() trains the same way, then relabels every row by its nearest
  # final center, a documented difference from upstream.
  train <- function(keep_cluster) {
    set.seed(7)
    emeans(rows, 2, centers = initial, nepochs = 1, batch_size = 6,
           alpha0 = 0.5, keep_cluster = keep_cluster)
  }
  fit <- train(TRUE)
  expect_equal(fit$centers, looped$centers, tolerance = 1e-12)
  expect_identical(fit$cluster, c(1L, 1L, 2L, 2L, 2L, 2L))
  disabled <- train(FALSE)
  expect_null(disabled$cluster)
  expect_identical(fit$centers, disabled$centers)
  expect_identical(fit$history$obj, disabled$history$obj)
})

test_that("initial_mass_first_update_and_zero_mass_reentry", {
  cases <- list(list(0.5, FALSE, 3), list(1, FALSE, 1e8), list(0.5, TRUE, 1e8))
  for (case in cases) {
    alpha <- case[[1]]
    zero_mass <- case[[2]]
    position <- case[[3]]
    st <- em_begin(2, matrix(c(position, 8.25)), nepochs = 2, batch_size = 2,
                   alpha0 = alpha)
    expect_identical(st$recipe$init_usage, 2^-16)
    # Inject zero mass only to isolate the reentry case, as upstream does.
    if (zero_mass) {
      st$usage[1] <- 0
    }
    mass <- if (zero_mass) 0 else 2^-16
    count <- 2
    weight <- alpha * count / ((1 - alpha) * mass + alpha * count)
    expected <- if (weight == 1) 1 else position + weight * (1 - position)
    update_with_statistics(st, matrix(c(2, 0)), c(2, 0), 7)
    expect_equal(st$centroids, matrix(c(expected, 8.25)), tolerance = 1e-12)
    unused_mass <- if (alpha == 1) 2^-16 else (1 - alpha) * 2^-16
    expect_identical(st$usage[2], unused_mass)
  }
})

test_that("untouched_underflow_and_independent_ema", {
  st <- em_begin(400, matrix(c(0, 8.25)), nepochs = 1, batch_size = 1,
                 alpha0 = 0.9)
  mass <- 2^-16
  position <- 0
  oracle <- got <- untouched <- numeric(400)
  # Disable revival only to isolate decay and the EMA, as upstream does.
  local_mocked_bindings(revive_unused = function(state) 0)
  for (step in 0:399) {
    value <- (step %% 7 - 3) / 8
    alpha <- scheduled_alpha(0.9, step, 400)
    mass <- (1 - alpha) * mass + alpha
    position <- position + alpha / mass * (value - position)
    update_with_statistics(st, matrix(c(value, 0)), c(1, 0), 0)
    oracle[step + 1] <- position
    got[step + 1] <- st$centroids[1, 1]
    untouched[step + 1] <- st$centroids[2, 1]
  }
  # Every step against the double scalar oracle (upstream: FP32 rtol 3e-6).
  expect_lte(max(abs(got - oracle)), 1e-12)
  expect_identical(untouched, rep(8.25, 400))
  # Upstream's bound is float32's smallest normal, 2^-126.
  expect_lte(st$usage[2], 2^-126)
  em_end_epoch(st)
  expect_identical(em_finalize(st)$centers[2, 1], 8.25)
})

test_that("revival_threshold_separation_and_conservation", {
  st <- em_begin(3, matrix(c(10, 20, 30)), unused_threshold = 0.1)
  # Adapted for double precision: upstream's masses put cluster 1 exactly on the
  # FP32-rounded threshold, which a double threshold does not reproduce.
  # These masses put it exactly on the double threshold, so the strict `<`
  # keeps it used. Upstream's FP32 bit patterns of the threshold have no
  # double analogue and are not checked.
  st$usage <- c(27.75, 0.375, 804.375)
  before <- st$usage
  threshold <- st$recipe$unused_threshold * sum(before) / 3
  expect_identical(threshold, before[1])
  expect_identical(before < threshold, c(FALSE, TRUE, FALSE))
  weighted <- colSums(st$centroids * before)
  discarded <- st$centroids[2, ] * before[2]
  set.seed(123)
  expect_equal(revive_unused(st), 1)
  # Infer the sampled donor rather than pinning the RNG.
  after <- st$usage
  source <- which(before != after & seq_len(3) != 2)
  expect_length(source, 1)
  initial_position <- c(10, 20, 30)[source]
  expect_equal(st$centroids[source, ], initial_position * 0.9999,
               tolerance = 1e-12)
  expect_equal(st$centroids[2, ], initial_position * 1.0001, tolerance = 1e-12)
  other <- if (source == 1) 3 else 1
  expect_identical(st$centroids[other, ], c(10, 20, 30)[other])
  expect_equal(sum(st$usage), sum(before) - before[2], tolerance = 1e-12)
  expect_equal(colSums(st$centroids * st$usage), weighted - discarded,
               tolerance = 1e-12)

  donor <- rbind(c(0, 0), c(2, 3), c(1, 0))
  child <- rbind(c(0, 0), c(2.0002, 2.9997), c(1, 0))
  ordinary <- child[2, ]
  result <- separate_splits(donor, child)
  # The split nudge uses the next representable double, not the next float32.
  expect_identical(result[1, 1], 2^-1074)
  expect_identical(result[3, 1], 1 + .Machine$double.eps)
  expect_identical(result[2, ], ordinary)

  # One donor and two unused clusters: one is revived, donor and child
  # separate, and the third is left alone. The RNG stream continues.
  unique <- em_begin(3, rbind(c(0, 0), c(10, 10), c(20, 20)),
                     unused_threshold = 0.1)
  unique$usage <- c(10, 0, 0)
  expect_equal(revive_unused(unique), 1)
  expect_identical(unique$usage, c(5, 5, 0))
  expect_false(identical(unique$centroids[1, ], unique$centroids[2, ]))
  expect_identical(unique$centroids[3, ], c(20, 20))
})

test_that("closing_statistics_survive_revival_and_object_deletion", {
  set.seed(1)
  st <- em_begin(5, matrix(c(0, 100)), nepochs = 1, batch_size = 1,
                 alpha0 = 0.5, unused_threshold = 0.1)
  for (value in c(0, 1, 2, 3, 4)) {
    update_with_statistics(st, matrix(c(value, 0)), c(1, 0), 1)
  }
  expect_equal(st$n_splits, 1)
  revived <- st$centroids[2, 1]
  em_end_epoch(st)
  result <- em_finalize(st)
  expect_identical(result$centers, matrix(c(2, revived)))
  expect_identical(result$history$obj[1], 5)
  expect_identical(result$history$imbalance[1], 2)
  expect_identical(em_finalize(st), result)
  # Upstream checks that the working buffers are released; em_finalize()
  # drops them from the state.
  expect_null(st$centroids)
  expect_null(st$pass_sums)
  before <- result
  rm(st)
  expect_identical(result, before)
})

test_that("fractional_caller_and_matrix_training", {
  x <- matrix(0:50 / 10, 17, 3, byrow = TRUE)
  initial <- x[c(1, 6, 11, 17), ]
  args <- list(nepochs = 3, batch_size = 4, alpha0 = 0.2, chunk_size = 2,
               centroid_chunk_size = 2)
  set.seed(7)
  matrix_fit <- do.call(emeans, c(list(x, 4, centers = initial), args))
  set.seed(7)
  caller <- do.call(caller_run, c(list(x, initial), args))
  expect_identical(caller$fit$centers, matrix_fit$centers)
  expect_identical(caller$fit$history$obj, matrix_fit$history$obj)
  expect_identical(caller$fit$n_splits, matrix_fit$n_splits)
})

test_that("pass_counts_keep_small_fractions", {
  cases <- list(list(25, "25"), list(1.4, "1.4"), list(0.004, "0.004"),
                list(1.001, "1.001"), list(2.999, "2.999"))
  for (case in cases) {
    expect_identical(format_passes(case[[1]]), case[[2]])
  }
})

test_that("assignment_chunking_preserves_logical_updates", {
  set.seed(123)
  x <- matrix(rnorm(41 * 5), 41, 5)
  init <- x[1:4, ]
  train <- function(...) {
    set.seed(7)
    emeans(x, 4, centers = init, nepochs = 3, batch_size = 13, alpha0 = 0.2,
           ...)
  }
  for (kernel in kernels) {
    withr::local_options(emeans.kernel = kernel)
    # chunk_size and centroid_chunk_size both split the assignment; vary both.
    a <- train(chunk_size = 4, centroid_chunk_size = 3)
    b <- train(chunk_size = 64)
    # Only the direct kernel is bit-identical across chunk sizes; a
    # BLAS may round differently by matrix shape, as upstream notes for faiss.
    same <- if (kernel == "direct") expect_identical else expect_equal
    same(a$centers, b$centers)
    same(a$history$obj, b$history$obj)
    expect_identical(a$recipe$steps_per_pass, 4)
    expect_identical(a$recipe$total_steps, 12)
  }
})
