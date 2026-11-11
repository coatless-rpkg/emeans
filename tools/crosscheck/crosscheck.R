# Python cross-check release gate, R side (development only; not shipped).
#
# For each configuration this script generates well-separated blob data,
# records every random draw through the internal `emeans.rng_recorder` hook,
# and runs the caller-driven em_*() loop. It writes the inputs (data, starting
# centroids, per-epoch permutations, revival uniforms) and R's outputs
# (centroids after every epoch, final centers, last-pass labels) under
# tools/crosscheck/runs/<name>/ for crosscheck.py, which replays the same run
# through upstream faiss NumpyEmeans.
#
# Run from the package root:
#   Rscript tools/crosscheck/crosscheck.R
#   uv run --project tools/crosscheck python tools/crosscheck/crosscheck.py
#
# Exchange format: little-endian doubles (writeBin(size = 8)); matrices are
# written row-major (k x d reads back with numpy reshape(k, d)); indices are
# 0-based.

suppressMessages(devtools::load_all(".", quiet = TRUE))

# The gate uses the direct kernel and one thread: results are then
# bit-identical across chunk sizes and threads (see ?emeans, "Performance").
options(emeans.kernel = "direct")
threads <- 1L

run_root <- file.path("tools", "crosscheck", "runs")

# ---- Helpers -------------------------------------------------------------------

write_f64 <- function(x, path) {
  writeBin(as.double(x), path, size = 8, endian = "little")
}

write_matrix <- function(m, path) {
  write_f64(t(m), path) # row-major
}

# Round to the nearest float32, so that R (double) and upstream (which casts
# its input to float32) see exactly the same data and starting centroids.
round_to_f32 <- function(x) {
  v <- readBin(writeBin(as.double(x), raw(), size = 4, endian = "little"),
               "double", n = length(x), size = 4, endian = "little")
  dim(v) <- dim(x)
  v
}

# Blob centres at +/- radius on the coordinate axes: the closest pair is
# radius * sqrt(2) apart. At most 2 * d blobs.
axis_centres <- function(n_blobs, d, radius) {
  stopifnot(n_blobs <= 2 * d)
  m <- matrix(0, n_blobs, d)
  for (b in seq_len(n_blobs)) {
    axis <- (b - 1L) %% d + 1L
    m[b, axis] <- if (b <= d) radius else -radius
  }
  m
}

min_pair_distance <- function(m) {
  min(stats::dist(m))
}

# Euclidean blobs: centres 15 sd from the origin, so the closest pair of
# centres is 15 * sqrt(2) = 21.2 sd apart. Spherical blobs: unit axis
# directions with per-coordinate noise sd 0.05 before projection, so the
# closest pair of directions is sqrt(2) / 0.05 = 28.3 sd apart.
make_blobs <- function(n, d, n_blobs, spherical) {
  if (spherical) {
    sd <- 0.05
    centres <- axis_centres(n_blobs, d, 1)
  } else {
    sd <- 1
    centres <- axis_centres(n_blobs, d, 15)
  }
  blob <- rep_len(seq_len(n_blobs), n)
  x <- centres[blob, , drop = FALSE] + matrix(stats::rnorm(n * d, sd = sd), n, d)
  if (spherical) {
    x <- x / sqrt(rowSums(x * x))
  }
  list(x = round_to_f32(x), separation_sd = min_pair_distance(centres) / sd,
       n_blobs = n_blobs)
}

param_value <- function(v) {
  if (is.null(v) || (length(v) == 1L && is.na(v))) {
    return("None")
  }
  if (is.logical(v)) {
    return(if (v) "True" else "False")
  }
  if (is.numeric(v)) {
    return(paste(sprintf("%.17g", v), collapse = ","))
  }
  paste(v, collapse = ",")
}

write_params <- function(params, path) {
  lines <- vapply(names(params), function(key) {
    paste0(key, "=", param_value(params[[key]]))
  }, character(1))
  writeLines(lines, path)
}

# ---- Configurations ------------------------------------------------------------

configs <- list(
  list(name = "plain", seed = 101, n = 4000, d = 8, k = 16, n_blobs = 16,
       nepochs = 3),
  list(name = "spherical", seed = 202, n = 4000, d = 8, k = 16, n_blobs = 16,
       nepochs = 3, spherical = TRUE),
  list(name = "revival", seed = 303, n = 8000, d = 8, k = 64, n_blobs = 8,
       nepochs = 3),
  list(name = "fractional", seed = 404, n = 4000, d = 8, k = 16, n_blobs = 16,
       nepochs = 2.5),
  list(name = "mppc", seed = 505, n = 6000, d = 8, k = 16, n_blobs = 16,
       nepochs = 3, max_points_per_centroid = 150),
  list(name = "manual", seed = 606, n = 4000, d = 8, k = 16, n_blobs = 16,
       nepochs = 3, batch_size = 64, alpha0 = 0.3),
  list(name = "lowdim", seed = 707, n = 2000, d = 2, k = 4, n_blobs = 4,
       nepochs = 3)
)

# ---- One configuration ---------------------------------------------------------

run_config <- function(cfg) {
  spherical <- isTRUE(cfg$spherical)
  batch_law <- if (is.null(cfg$batch_law)) "auto" else cfg$batch_law
  dir <- file.path(run_root, cfg$name)
  unlink(dir, recursive = TRUE)
  dir.create(dir, recursive = TRUE)

  set.seed(cfg$seed)
  blobs <- make_blobs(cfg$n, cfg$d, cfg$n_blobs, spherical)
  stopifnot(blobs$separation_sd >= 20)
  x_full <- blobs$x
  seed_after_data <- get(".Random.seed", envir = globalenv())

  # Record draws only from here on.
  rec <- new.env()
  rec$draws <- list()
  old <- options(emeans.rng_recorder = rec)
  on.exit(options(old), add = TRUE)

  # Same order as emeans(): subset, init, then one permutation per epoch;
  # revival uniforms are drawn inside em_update().
  keep <- NULL
  x <- x_full
  limit <- if (is.null(cfg$max_points_per_centroid)) Inf else cfg$max_points_per_centroid * cfg$k
  if (nrow(x_full) > limit) {
    keep <- draw_subset(nrow(x_full), limit)
    x <- x_full[keep, , drop = FALSE]
  }
  n <- nrow(x)
  init <- x[draw_init_rows(n, cfg$k), , drop = FALSE]

  st <- em_begin(n, init, nepochs = cfg$nepochs, batch_law = batch_law,
                 batch_size = cfg$batch_size, alpha0 = cfg$alpha0,
                 spherical = spherical,
                 unused_threshold = cfg$unused_threshold, device = "cpu",
                 threads = threads)
  recipe <- st$recipe
  B <- recipe$batch_size
  labels <- rep(NA_integer_, n)
  step_centres <- list()
  epoch_centres <- list()
  for (e in seq_along(recipe$epoch_rows)) {
    perm <- draw_permutation(n)
    write_f64(perm - 1, file.path(dir, sprintf("perm_%d.f64", e - 1L)))
    rows_e <- recipe$epoch_rows[e]
    for (s in seq(1, rows_e, by = B)) {
      rows <- perm[s:min(s + B - 1, rows_e)]
      stats <- em_accumulate(st, x[rows, , drop = FALSE])
      if (!is.null(stats$labels)) {
        labels[rows] <- stats$labels
      }
      em_update(st, stats)
      step_centres[[length(step_centres) + 1L]] <- t(em_centers(st))
    }
    em_end_epoch(st)
    epoch_centres[[e]] <- em_centers(st)
    write_matrix(epoch_centres[[e]], file.path(dir, sprintf("r_centers_epoch_%d.f64", e - 1L)))
  }
  fit <- em_finalize(st)

  types <- vapply(rec$draws, `[[`, character(1), "type")
  revival <- rec$draws[types == "revival"]
  stopifnot(all(lengths(lapply(revival, `[[`, "value")) == cfg$k))

  write_matrix(x_full, file.path(dir, "x.f64"))
  if (!is.null(keep)) {
    write_f64(keep - 1, file.path(dir, "subset.f64"))
  }
  write_matrix(init, file.path(dir, "init.f64"))
  write_f64(unlist(lapply(revival, `[[`, "value")), file.path(dir, "revival.f64"))
  write_matrix(fit$centers, file.path(dir, "r_centers_final.f64"))
  write_f64(labels, file.path(dir, "r_labels.f64"))
  write_f64(unlist(step_centres), file.path(dir, "r_centers_steps.f64"))

  # The emeans() trainer, replayed from the same RNG state, must make the same
  # draws and give the same fit as the loop above.
  assign(".Random.seed", seed_after_data, envir = globalenv())
  rec_trainer <- new.env()
  rec_trainer$draws <- list()
  options(emeans.rng_recorder = rec_trainer)
  fit_trainer <- emeans(x_full, cfg$k, nepochs = cfg$nepochs,
                        batch_law = batch_law, batch_size = cfg$batch_size,
                        alpha0 = cfg$alpha0, spherical = spherical,
                        unused_threshold = cfg$unused_threshold,
                        max_points_per_centroid = cfg$max_points_per_centroid,
                        device = "cpu", threads = threads)
  options(emeans.rng_recorder = rec)
  full_labels <- if (is.null(keep)) labels else replace(rep(NA_integer_, nrow(x_full)), keep, labels)
  trainer_identical <- identical(rec_trainer$draws, rec$draws) &&
    identical(fit_trainer$centers, fit$centers) &&
    identical(fit_trainer$cluster, full_labels) &&
    identical(fit_trainer$n_splits, fit$n_splits)

  write_params(list(
    name = cfg$name,
    seed = cfg$seed,
    n_full = nrow(x_full),
    n = n,
    d = cfg$d,
    k = cfg$k,
    n_blobs = blobs$n_blobs,
    separation_sd = blobs$separation_sd,
    nepochs = cfg$nepochs,
    batch_law = batch_law,
    batch_size = cfg$batch_size,
    alpha0 = cfg$alpha0,
    unused_threshold = cfg$unused_threshold,
    spherical = spherical,
    max_points_per_centroid = cfg$max_points_per_centroid,
    r_recipe_name = recipe$name,
    r_batch_size = recipe$batch_size,
    r_alpha0 = recipe$alpha0,
    r_epoch_rows = recipe$epoch_rows,
    r_total_steps = recipe$total_steps,
    r_unused_threshold = recipe$unused_threshold,
    r_n_splits = fit$n_splits,
    r_nsplit_per_epoch = fit$history$nsplit,
    n_revival_draws = length(revival),
    draw_types = paste(unique(types), collapse = "+"),
    r_trainer_identical = trainer_identical,
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    r_emeans_version = as.character(utils::packageVersion("emeans")),
    r_rng_kind = paste(RNGkind(), collapse = "/"),
    r_kernel = recipe$kernel,
    r_threads = recipe$threads
  ), file.path(dir, "params.txt"))

  message(sprintf(
    "%-10s n=%5.0f d=%d k=%2d B=%4.0f alpha0=%.4g epochs=%s splits=%3.0f revival draws=%3d trainer identical=%s",
    cfg$name, n, cfg$d, cfg$k, B, recipe$alpha0,
    paste(recipe$epoch_rows, collapse = "/"), fit$n_splits, length(revival),
    trainer_identical
  ))
  invisible(trainer_identical)
}

# ---- Main ------------------------------------------------------------------------

dir.create(run_root, recursive = TRUE, showWarnings = FALSE)
ok <- vapply(configs, run_config, logical(1))
# Not fatal here: crosscheck.py reads r_trainer_identical and fails the
# configuration, so the Python side still runs and reports.
if (!all(ok)) {
  warning("emeans() did not reproduce the em_*() loop for: ",
          paste(vapply(configs[!ok], `[[`, character(1), "name"), collapse = ", "))
}
