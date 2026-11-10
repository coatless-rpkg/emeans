# Training state machine: a port of upstream base.py (`_EmeansBase`: update,
# _finish_step, end_epoch) and e_means.py (`NumpyEmeans`: begin, _start_epoch,
# _accumulate, _accumulate_core, _apply_statistics, finalize).
#
# The state touches data and centroids only through the backend interface
# (R/backend.R), so the CPU and torch backends share it. Every call validates
# its inputs and the call order before it changes anything. A rejected call
# leaves the state usable. `state$failed` is set while a call mutates the
# state and cleared when it succeeds, so an error (or an interrupt) part-way
# through leaves the state failed, as upstream's `_failed` does.

# ---- Input validation -------------------------------------------------------

is_count <- function(x, min = 1) {
  is.numeric(x) && length(x) == 1L && is.finite(x) && x == round(x) && x >= min
}

check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    abort_emeans(sprintf("`%s` must be TRUE or FALSE.", name), "emeans_error_input")
  }
  x
}

#' Coerce a numeric matrix or data frame, rejecting non-finite values
#' @noRd
as_finite_matrix <- function(x, name) {
  if (is.data.frame(x)) {
    numeric_cols <- vapply(x, is.numeric, logical(1))
    if (!all(numeric_cols)) {
      abort_emeans(
        sprintf("`%s` has non-numeric column(s): %s.", name,
                paste(names(x)[!numeric_cols], collapse = ", ")),
        "emeans_error_input"
      )
    }
    x <- as.matrix(x)
  }
  if (!is.matrix(x) || !is.numeric(x)) {
    abort_emeans(sprintf("`%s` must be a numeric matrix or data frame.", name),
                 "emeans_error_input")
  }
  # min() and max() propagate NA and NaN and expose infinities without
  # allocating a logical copy of the matrix.
  if (length(x) > 0L && !(is.finite(min(x)) && is.finite(max(x)))) {
    abort_emeans(sprintf("`%s` must not contain NA, NaN or infinite values.", name),
                 "emeans_error_input")
  }
  x
}

check_begin_args <- function(n, k, nepochs, batch_size, alpha0) {
  if (!is_count(n)) {
    abort_emeans("`n` must be a single whole number >= 1.", "emeans_error_input")
  }
  if (n < k) {
    abort_emeans(
      sprintf("`n` (%.0f) must be at least the number of centers (%d).", n, k),
      "emeans_error_input"
    )
  }
  if (!is.numeric(nepochs) || length(nepochs) != 1L || !is.finite(nepochs) ||
      nepochs <= 0) {
    abort_emeans("`nepochs` must be a single finite number > 0.", "emeans_error_input")
  }
  if (!is.null(batch_size) && !is_count(batch_size)) {
    abort_emeans("`batch_size` must be NULL or a single whole number >= 1.",
                 "emeans_error_input")
  }
  if (!is.null(alpha0) && !(is.numeric(alpha0) && length(alpha0) == 1L &&
                            is.finite(alpha0) && alpha0 > 0 && alpha0 <= 1)) {
    abort_emeans("`alpha0` must be NULL or a single number in (0, 1].",
                 "emeans_error_input")
  }
  invisible(NULL)
}

check_batch_law <- function(batch_law) {
  choices <- c("auto", "throughput", "quality")
  if (identical(batch_law, choices)) {
    return("auto")
  }
  if (!is.character(batch_law) || length(batch_law) != 1L ||
      !batch_law %in% choices) {
    abort_emeans(
      "`batch_law` must be one of \"auto\", \"throughput\" or \"quality\".",
      "emeans_error_input"
    )
  }
  batch_law
}

# ---- State helpers ------------------------------------------------------------

elapsed_seconds <- function() {
  proc.time()[["elapsed"]]
}

n_epochs <- function(state) {
  length(state$recipe$epoch_rows)
}

#' Rows the next logical batch must hold (port of `_local_batch_size`)
#' @noRd
local_batch_size <- function(state) {
  min(
    state$recipe$batch_size,
    state$recipe$epoch_rows[state$epoch] - state$epoch_rows_done
  )
}

with_col_names <- function(m, col_names) {
  if (!is.null(col_names)) {
    colnames(m) <- col_names
  }
  m
}

check_state <- function(state) {
  if (!inherits(state, "em_state")) {
    abort_emeans("`state` must be a state created by em_begin().", "emeans_error_input")
  }
  invisible(state)
}

require_running <- function(state) {
  check_state(state)
  if (isTRUE(state$failed)) {
    abort_emeans(
      "This state failed during an earlier call; create a new state with em_begin().",
      "emeans_error_sequence"
    )
  }
  invisible(state)
}

require_open_epoch <- function(state, what) {
  if (state$epoch > n_epochs(state)) {
    abort_emeans(
      sprintf("Every epoch is complete; call em_finalize() instead of %s.", what),
      "emeans_error_sequence"
    )
  }
  invisible(state)
}

#' Reset per-epoch accumulators (port of `_start_epoch`)
#'
#' Per-cluster pass sums exist only in the last epoch, for the closing means.
#' @noRd
start_epoch <- function(state) {
  be <- state$backend
  state$epoch_steps_done <- 0
  state$epoch_rows_done <- 0
  state$splits_before <- state$n_splits
  state$pass_counts <- be$zeros_vector(state$k)
  state$objective <- be$as_native_vector(0)
  state$pass_sums <- if (state$epoch == n_epochs(state)) {
    be$zeros_matrix(state$k, state$d)
  } else {
    NULL
  }
  invisible(state)
}

new_em_stats <- function(pending, labels) {
  structure(
    list(
      n = pending$n,
      complete = pending$complete,
      labels = labels,
      acc = if (pending$complete) pending$acc else NULL
    ),
    batch = pending$token,
    class = "em_stats"
  )
}

check_batch_width <- function(state, width) {
  if (width != state$d) {
    abort_emeans(
      sprintf("The batch has %d column(s); the centers have %d.", width, state$d),
      "emeans_error_input"
    )
  }
  invisible(state)
}

batch_overflow_message <- function(state, pending_n, n, expected) {
  if (expected == 0) {
    return(sprintf(
      "Epoch %d has received all of its %.0f rows; call em_end_epoch() before accumulating more.",
      state$epoch, state$recipe$epoch_rows[state$epoch]
    ))
  }
  if (pending_n == expected) {
    return(sprintf(
      "The current batch is complete (%.0f rows); call em_update() before accumulating more rows.",
      expected
    ))
  }
  sprintf(
    "The current batch takes %.0f rows and already has %.0f; %.0f more would overflow it.",
    expected, pending_n, n
  )
}

# ---- Shared core --------------------------------------------------------------

#' Accumulate rows of native data against the current batch's snapshot
#'
#' Port of upstream `_accumulate`/`_accumulate_core`. The first call of a
#' logical batch fixes the centroid readout that every later piece of the
#' batch is assigned against. `em_accumulate()` and `emeans()` both call this.
#'
#' @param state An `em_state`.
#' @param data Native data from `backend$prepare_data()`.
#' @param rows 1-based row indices into `data`, in batch order.
#' @return An `em_stats` object.
#' @noRd
accumulate_rows <- function(state, data, rows) {
  require_running(state)
  require_open_epoch(state, "em_accumulate()")
  be <- state$backend
  check_batch_width(state, be$n_cols(data))
  n <- length(rows)
  expected <- local_batch_size(state)
  pending <- state$pending
  want_labels <- isTRUE(state$keep_labels) && state$epoch == n_epochs(state)
  # Upstream returns the pending complete batch again for an empty call.
  if (n == 0 && !is.null(pending) && pending$complete) {
    return(new_em_stats(pending, if (want_labels) integer(0)))
  }
  pending_n <- if (is.null(pending)) 0 else pending$n
  if (pending_n + n > expected) {
    abort_emeans(batch_overflow_message(state, pending_n, n, expected),
                 "emeans_error_batch")
  }
  report_header_once(state)

  state$failed <- TRUE
  if (pending_n == 0) {
    pending <- list(
      # A fresh environment is a unique, RNG-free identity for this batch.
      token = new.env(parent = emptyenv()),
      n = 0,
      complete = FALSE,
      readout = be$readout(state$centroids, state$spherical),
      acc = list(
        sums = be$zeros_matrix(state$k, state$d),
        counts = be$zeros_vector(state$k),
        objective = be$as_native_vector(0)
      )
    )
  }
  labels <- if (want_labels) integer(0) else NULL
  if (n > 0) {
    res <- be$accumulate(data, rows, pending$readout, pending$acc,
                         state$spherical, want_labels)
    pending$acc <- res$acc
    if (want_labels) {
      labels <- res$labels
    }
  }
  pending$n <- pending_n + n
  pending$complete <- pending$n == expected
  state$pending <- pending
  state$failed <- FALSE
  new_em_stats(pending, labels)
}

#' Add a batch to the pass totals and apply the EMA step
#'
#' Port of `_apply_statistics` (Algorithm 1, line 8; Equation 2).
#' @noRd
apply_statistics <- function(state, acc, alpha) {
  be <- state$backend
  state$objective <- be$add(state$objective, acc$objective)
  state$pass_counts <- be$add(state$pass_counts, acc$counts)
  if (!is.null(state$pass_sums)) {
    state$pass_sums <- be$add(state$pass_sums, acc$sums)
  }
  up <- be$update(state$centroids, state$usage, acc$sums, acc$counts,
                  alpha, alpha == 1)
  state$centroids <- up$centroids
  state$usage <- up$usage
  invisible(state)
}

#' Split donors into unused clusters (port of `_revive_unused`)
#' @return The number of clusters revived.
#' @noRd
revive_unused <- function(state) {
  be <- state$backend
  splits <- choose_splits(be$as_r_vector(state$usage), state$recipe$unused_threshold)
  if (is.null(splits)) {
    return(0)
  }
  res <- be$split(state$centroids, state$usage, splits$source, splits$target)
  state$centroids <- res$centroids
  state$usage <- res$usage
  nsplit <- length(splits$source)
  state$n_splits <- state$n_splits + nsplit
  nsplit
}

#' Revival check, then advance the counters (port of `_finish_step`)
#' @noRd
finish_step <- function(state, expected) {
  if ((state$step + 1) %% state$recipe$check_unused_every == 0) {
    revive_unused(state)
  }
  state$step <- state$step + 1
  state$epoch_steps_done <- state$epoch_steps_done + 1
  state$epoch_rows_done <- state$epoch_rows_done + expected
  state$pending <- NULL
  invisible(state)
}

history_frame <- function(history) {
  column <- function(name) vapply(history, function(s) as.double(s[[name]]), numeric(1))
  data.frame(
    epoch = column("epoch"),
    rows = column("rows"),
    obj = column("obj"),
    time = column("time"),
    imbalance = column("imbalance"),
    nsplit = column("nsplit")
  )
}

#' Build an `"emeans"` result
#' @noRd
new_emeans <- function(centers, cluster, size, history, recipe, n_splits,
                       spherical, call) {
  structure(
    list(
      cluster = cluster,
      centers = centers,
      size = size,
      iter = recipe$total_steps,
      history = history,
      recipe = recipe,
      n_splits = n_splits,
      spherical = spherical,
      call = call
    ),
    class = "emeans"
  )
}

# ---- Exported loop ------------------------------------------------------------

#' Caller-driven e-means training
#'
#' @description
#' \strong{Lifecycle: experimental.} These functions may change in a later
#' release.
#'
#' They run e-means one batch at a time, for data that will not fit in memory
#' or arrives in pieces, and mirror upstream faiss's
#' `begin`/`accumulate`/`update`/`end_epoch`/`finalize`, with [emeans()]
#' built on the same internals.
#'
#' 1. `em_begin()` creates a training state from the total row count and the
#'    starting centers. Its `$recipe` gives the batch size (`batch_size`) and
#'    the rows of each epoch (`epoch_rows`).
#' 2. For each epoch `e`, the caller shuffles its rows and delivers the first
#'    `recipe$epoch_rows[e]` of them in consecutive batches of
#'    `recipe$batch_size` rows; the last batch of an epoch keeps the
#'    remainder. Each batch goes through `em_accumulate()` and then
#'    `em_update()`, and may reach `em_accumulate()` in several pieces that
#'    are all assigned against the same centers, with only the statistics of
#'    the call that completes the batch accepted by `em_update()`.
#' 3. `em_end_epoch()` closes each epoch.
#' 4. `em_finalize()` returns the fit.
#'
#' Batch rules, as upstream enforces them: every batch has exactly
#' `recipe$batch_size` rows except the last of each pass; each pass delivers
#' exactly `recipe$epoch_rows[i]` rows (the first `epoch_rows[i]` of a fresh
#' shuffle); and calls come in the order above, `em_accumulate()` before
#' `em_update()`, `em_end_epoch()` after the pass's last batch.
#'
#' Batch sizes, epoch row counts and call order are checked. A rejected call
#' changes nothing, so the loop can continue; an error after a call has
#' started changing the state leaves it unusable, and later calls ask for a
#' new state.
#'
#' @param n Total number of rows in the data, which may exceed R's integer
#'   range.
#' @param centers A `k x d` numeric matrix of starting centers. In spherical
#'   mode they are normalised to unit length.
#' @param nepochs Number of passes over the data; may be fractional.
#' @param batch_law Scaling law that picks the batch size when `batch_size`
#'   is `NULL`. `"auto"` uses `"quality"` when `d <= 16` and `"throughput"`
#'   otherwise.
#' @param batch_size Rows per batch, or `NULL` to use `batch_law`.
#' @param alpha0 Initial EMA step in (0, 1], or `NULL` for the step law.
#' @param spherical Cluster unit vectors by inner product, leaving
#'   normalisation of the data to you.
#' @param unused_threshold Revival threshold: clusters whose mass is below
#'   this fraction of the mean are revived. `NULL` means 0.1, or 0 (no
#'   revival) when `d <= 2`.
#' @param chunk_size,centroid_chunk_size Rows and centers searched at once,
#'   or `NULL` for the defaults. They trade memory for speed and, with the
#'   `"direct"` kernel, leave the result untouched.
#' @param device Where to compute: `"cpu"` (the default) runs the C++ kernels
#'   in double precision; `"mps"` (Apple GPUs), `"cuda"` or `"cuda:<index>"`
#'   (NVIDIA GPUs, experimental), `"torch:cpu"` or a `torch_device` run the
#'   optional \pkg{torch} backend in float32, as described under "Devices"
#'   in [emeans()].
#' @param threads Number of threads, or `NULL` for
#'   `getOption("emeans.threads", 2L)`.
#' @param verbose Print progress messages. The header is printed at the
#'   first call after `em_begin()`.
#' @param state A state from `em_begin()`.
#' @param batch A numeric matrix or data frame holding the batch's rows (or
#'   part of them), with `d` columns. On a torch device it may also be a 2-D
#'   `torch_tensor`, which is cast to float32 and not checked for `NA`.
#' @param stats The result of the `em_accumulate()` call that completed the
#'   batch.
#' @param x An `em_state`.
#' @param ... Ignored.
#'
#' @return
#' * `em_begin()`: an `em_state`, an environment whose `$recipe` is the
#'   training recipe.
#' * `em_accumulate()`: an `em_stats` list with `n` (rows of the current
#'   batch so far), `complete` (whether the batch is full) and `labels`
#'   (1-based nearest-center labels of the rows just passed, in the last
#'   epoch only; otherwise `NULL`).
#' * `em_update()`, `em_end_epoch()`: the state, invisibly.
#' * `em_finalize()`: an `"emeans"` object with `cluster = NULL` and no sums
#'   of squares, since the caller keeps the last-epoch labels and the data;
#'   calling it again returns the same object.
#' * `em_centers()`: the current centers as a `k x d` matrix (unit directions
#'   in spherical mode), or the final centers after `em_finalize()`.
#'
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(400), ncol = 2)
#' st <- em_begin(nrow(x), x[1:3, ], nepochs = 2, threads = 1)
#' B <- st$recipe$batch_size
#' for (e in seq_along(st$recipe$epoch_rows)) {
#'   ord <- sample.int(nrow(x))[seq_len(st$recipe$epoch_rows[e])]
#'   for (s in seq(1, length(ord), by = B)) {
#'     rows <- ord[s:min(s + B - 1, length(ord))]
#'     em_update(st, em_accumulate(st, x[rows, , drop = FALSE]))
#'   }
#'   em_end_epoch(st)
#' }
#' fit <- em_finalize(st)
#' fit$centers
#' @name em_loop
NULL

#' @rdname em_loop
#' @export
em_begin <- function(n, centers, nepochs = 5,
                     batch_law = c("auto", "throughput", "quality"),
                     batch_size = NULL, alpha0 = NULL, spherical = FALSE,
                     unused_threshold = NULL, chunk_size = NULL,
                     centroid_chunk_size = NULL, device = "cpu",
                     threads = NULL, verbose = FALSE) {
  call <- match.call()
  centers <- as_finite_matrix(centers, "centers")
  k <- nrow(centers)
  d <- ncol(centers)
  if (k < 1L || d < 1L) {
    abort_emeans("`centers` must have at least one row and one column.",
                 "emeans_error_input")
  }
  check_begin_args(n, k, nepochs, batch_size, alpha0)
  batch_law <- check_batch_law(batch_law)
  check_flag(spherical, "spherical")
  check_flag(verbose, "verbose")
  n <- as.double(n)
  recipe <- build_recipe(n, k, d, nepochs, batch_law, batch_size, alpha0,
                         unused_threshold)
  chunks <- resolve_chunk_sizes(chunk_size, centroid_chunk_size)
  device_info <- resolve_device(device)
  threads <- resolve_threads(threads)
  kernel <- if (identical(device_info$type, "cpu")) resolve_kernel(d = d) else NA_character_
  backend <- make_backend(device_info, kernel, threads, chunks$chunk_size,
                          chunks$centroid_chunk_size)
  recipe <- c(recipe, list(
    backend = backend$name,
    kernel = kernel,
    threads = threads,
    chunk_size = chunks$chunk_size,
    centroid_chunk_size = chunks$centroid_chunk_size,
    device = device_info$device,
    spherical = spherical
  ))

  state <- new.env(parent = emptyenv())
  state$failed <- TRUE
  state$recipe <- recipe
  state$backend <- backend
  state$n <- n
  state$k <- k
  state$d <- d
  state$spherical <- spherical
  state$verbose <- verbose
  state$keep_labels <- TRUE
  state$col_names <- colnames(centers)
  state$call <- call
  dimnames(centers) <- NULL
  centroids <- backend$as_native_matrix(centers)
  if (spherical) {
    centroids <- backend$readout(centroids, TRUE)
  }
  state$centroids <- centroids
  state$usage <- backend$as_native_vector(rep(recipe$init_usage, k))
  state$n_splits <- 0
  state$step <- 0
  state$epoch <- 1L
  state$history <- list()
  state$pending <- NULL
  state$result <- NULL
  start_epoch(state)
  state$start_time <- elapsed_seconds()
  state$header_due <- TRUE
  state$failed <- FALSE
  class(state) <- "em_state"
  state
}

#' @rdname em_loop
#' @export
em_accumulate <- function(state, batch) {
  require_running(state)
  require_open_epoch(state, "em_accumulate()")
  be <- state$backend
  if (is_torch_tensor(batch)) {
    if (!identical(be$name, "torch")) {
      abort_emeans(
        "`batch` is a torch_tensor; use a torch device in em_begin() or pass a matrix.",
        "emeans_error_input"
      )
    }
    batch <- as_training_tensor(batch, "batch")
    width <- batch$size(2L)
  } else {
    batch <- as_finite_matrix(batch, "batch")
    width <- ncol(batch)
  }
  # Check the width before prepare_data() copies the batch to a device.
  check_batch_width(state, width)
  data <- be$prepare_data(batch)
  accumulate_rows(state, data, seq_len(be$n_rows(data)))
}

#' @rdname em_loop
#' @export
em_update <- function(state, stats) {
  require_running(state)
  require_open_epoch(state, "em_update()")
  if (!inherits(stats, "em_stats")) {
    abort_emeans("`stats` must be the result of em_accumulate().", "emeans_error_input")
  }
  expected <- local_batch_size(state)
  if (expected == 0) {
    abort_emeans(
      sprintf("Epoch %d has received all of its %.0f rows; call em_end_epoch() next.",
              state$epoch, state$recipe$epoch_rows[state$epoch]),
      "emeans_error_sequence"
    )
  }
  pending <- state$pending
  if (is.null(pending) || !identical(attr(stats, "batch"), pending$token)) {
    abort_emeans(
      "`stats` do not belong to the current batch; pass the result of the latest em_accumulate().",
      "emeans_error_sequence"
    )
  }
  if (!isTRUE(stats$complete) || stats$n != expected) {
    abort_emeans(
      sprintf("The batch has %.0f of its %.0f rows; accumulate the rest before em_update().",
              stats$n, expected),
      "emeans_error_batch"
    )
  }
  report_header_once(state)
  alpha <- scheduled_alpha(state$recipe$alpha0, state$step, state$recipe$total_steps)

  state$failed <- TRUE
  apply_statistics(state, pending$acc, alpha)
  finish_step(state, expected)
  state$failed <- FALSE
  invisible(state)
}

#' @rdname em_loop
#' @export
em_end_epoch <- function(state) {
  require_running(state)
  require_open_epoch(state, "em_end_epoch()")
  epoch <- state$epoch
  rows <- state$recipe$epoch_rows[epoch]
  if (state$epoch_rows_done != rows) {
    abort_emeans(
      sprintf("Epoch %d has received %.0f of its %.0f rows; finish its batches before em_end_epoch().",
              epoch, state$epoch_rows_done, rows),
      "emeans_error_sequence"
    )
  }
  report_header_once(state)

  state$failed <- TRUE
  be <- state$backend
  counts <- be$as_r_vector(state$pass_counts)
  stat <- list(
    epoch = sum(state$recipe$epoch_rows[seq_len(epoch)]) / state$n,
    rows = rows,
    obj = as.double(be$as_r_vector(state$objective)),
    time = elapsed_seconds() - state$start_time,
    imbalance = state$k * sum(counts * counts) / (rows * rows),
    nsplit = state$n_splits - state$splits_before
  )
  state$history[[length(state$history) + 1L]] <- stat
  state$epoch <- epoch + 1L
  state$pending <- NULL
  if (state$epoch <= n_epochs(state)) {
    start_epoch(state)
  }
  state$failed <- FALSE
  if (state$verbose) {
    report_epoch(state, stat, epoch)
  }
  invisible(state)
}

#' @rdname em_loop
#' @export
em_finalize <- function(state) {
  require_running(state)
  if (!is.null(state$result)) {
    return(state$result)
  }
  if (state$epoch <= n_epochs(state)) {
    abort_emeans(
      sprintf("Epoch %d of %d is not finished; close every epoch with em_end_epoch() before em_finalize().",
              state$epoch, n_epochs(state)),
      "emeans_error_sequence"
    )
  }

  state$failed <- TRUE
  be <- state$backend
  centers <- be$as_r_matrix(
    be$closing(state$pass_sums, state$pass_counts, state$centroids, state$spherical)
  )
  counts <- be$as_r_vector(state$pass_counts)
  size <- if (all(counts <= .Machine$integer.max)) as.integer(counts) else counts
  state$result <- new_emeans(
    centers = with_col_names(centers, state$col_names),
    cluster = NULL,
    size = size,
    history = history_frame(state$history),
    recipe = state$recipe,
    n_splits = state$n_splits,
    spherical = state$spherical,
    call = state$call
  )
  # Release the model, as upstream's finalize does.
  state$centroids <- state$usage <- state$pass_sums <- NULL
  state$pass_counts <- state$objective <- state$pending <- NULL
  state$failed <- FALSE
  state$result
}

#' @rdname em_loop
#' @export
em_centers <- function(state) {
  require_running(state)
  if (!is.null(state$result)) {
    return(state$result$centers)
  }
  be <- state$backend
  with_col_names(be$as_r_matrix(be$readout(state$centroids, state$spherical)),
                 state$col_names)
}

#' @rdname em_loop
#' @export
print.em_state <- function(x, ...) {
  recipe <- x$recipe
  total_epochs <- length(recipe$epoch_rows)
  status <- if (isTRUE(x$failed)) {
    "failed; create a new state"
  } else if (!is.null(x$result)) {
    "finalized"
  } else if (x$epoch > total_epochs) {
    "every epoch done; call em_finalize()"
  } else {
    "running"
  }
  cat("<em_state> e-means caller-driven loop (experimental)\n")
  cat(sprintf("  %.0f points in %dD to %d clusters on %s\n",
              x$n, x$d, x$k, recipe$device))
  cat(sprintf("  epoch %d / %d, step %.0f / %.0f, batch %.0f\n",
              min(x$epoch, total_epochs), total_epochs, x$step,
              recipe$total_steps, recipe$batch_size))
  cat(sprintf("  status: %s\n", status))
  invisible(x)
}
