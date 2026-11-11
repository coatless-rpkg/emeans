#' Validate and coerce training data to a double matrix
#'
#' Accepts a numeric matrix or an all-numeric data frame, keeps column names
#' and rejects `NA`, `NaN` and infinite values.
#' @param x Matrix or data frame.
#' @param name Argument name for messages.
#' @return A double matrix.
#' @noRd
as_training_matrix <- function(x, name = "x") {
  x <- as_finite_matrix(x, name)
  if (!is.double(x)) {
    storage.mode(x) <- "double"
  }
  x
}

#' Fit e-means clustering
#'
#' @description
#' `emeans()` clusters the rows of `x` into `k` groups, much as
#' [stats::kmeans()] does, but learns from small random batches instead of
#' the whole data set at once. Each batch pulls its nearest centers part of
#' the way towards it and the pull weakens over a few passes, until a closing
#' pass puts every center at the mean of its points. The authors of e-means
#' report a lower loss than Lloyd's k-means in about a tenth of the passes
#' (Bessa et al. 2026). A final sweep then labels every row by its nearest
#' center, so the fit carries the same `cluster`, `size` and sums of squares
#' as a [stats::kmeans()] fit.
#'
#' Most fits need only `x` and `k`. The remaining arguments either adjust the
#' training recipe, which `emeans()` otherwise chooses from the size of the
#' data, or decide where and how fast the work runs.
#'
#' @param x The data, one row per observation: a numeric matrix or an
#'   all-numeric data frame without `NA`, `NaN` or infinite values. On a torch
#'   device it may also be a 2-D `torch_tensor` (see "Devices").
#' @param k Number of clusters, at most the number of rows used for training.
#' @param nepochs Number of passes over the data, 5 by default. It may be
#'   fractional, such as `2.5`, but not so small that a pass would hold no
#'   rows.
#' @param centers Optional `k x d` matrix of starting centers. By default
#'   `emeans()` picks `k` distinct rows of `x` at random.
#' @param spherical If `TRUE`, cluster directions rather than points: rows are
#'   compared by inner product and the centers kept at unit length. Normalise
#'   the rows of `x` yourself first.
#' @param batch_size Rows per batch. The default, `NULL`, lets `batch_law`
#'   choose.
#' @param batch_law How the default batch size is chosen. `"quality"` picks
#'   smaller batches for a lower loss and `"throughput"` larger ones for
#'   speed, while `"auto"`, the default, uses `"quality"` for data with at most
#'   16 columns and `"throughput"` beyond that.
#' @param alpha0 Starting step size, between 0 and 1, which sets how far a
#'   center moves towards a batch early on. The default, `NULL`, derives it
#'   from `k`, the batch size, the number of rows and `nepochs`.
#' @param unused_threshold Clusters holding less than this fraction of the
#'   average cluster's weight are revived by splitting a large cluster in two.
#'   The default, `NULL`, means 0.1, or no revival for data with one or two
#'   columns.
#' @param max_points_per_centroid If set, train on at most
#'   `max_points_per_centroid * k` rows chosen at random; the rows left out
#'   get `NA` in `cluster`. The default, `NULL`, trains on every row.
#' @param keep_cluster If `FALSE`, skip the final sweep that labels every row,
#'   which saves roughly the time of one pass and the memory for the labels.
#'   The fit then has no `cluster` or sums of squares, and `size` counts the
#'   rows of the last training pass.
#' @param chunk_size,centroid_chunk_size How many rows and centers are compared
#'   at once. They trade memory for speed and rarely need changing; `NULL` uses
#'   the defaults.
#' @param device Where to compute: `"cpu"` (the default), `"mps"` for Apple
#'   GPUs, `"cuda"` or `"cuda:<index>"` for NVIDIA GPUs (experimental),
#'   `"torch:cpu"` or a `torch_device`. Every device other than `"cpu"` needs
#'   the \pkg{torch} package (see "Devices").
#' @param threads Number of CPU threads. The default, `NULL`, uses
#'   `getOption("emeans.threads", 2L)`.
#' @param verbose If `TRUE`, print the recipe and a progress line for each
#'   pass.
#'
#' @return An object of class `"emeans"`, which is a list with:
#'
#' * `cluster`: the nearest final center for each row of `x`, as an integer
#'   vector matching `predict(fit, x)`. Rows left out by
#'   `max_points_per_centroid` are `NA`.
#' * `centers`: the `k x d` matrix of final centers.
#' * `totss`: the total sum of squares about the overall mean.
#' * `withinss`: for each cluster, the sum of squared distances from its rows
#'   to its center.
#' * `tot.withinss`: `sum(withinss)`.
#' * `betweenss`: `totss - tot.withinss`.
#' * `size`: the number of rows in each cluster.
#' * `iter`: the number of batch updates.
#' * `history`: a data frame with one row per pass, holding `epoch` (passes so
#'   far), `rows` (rows seen in the pass), `obj` (the sum of squared distances
#'   from each row to its center when it was seen), `time` (seconds since the
#'   start), `imbalance` (1 when the clusters are equal in size, larger
#'   otherwise) and `nsplit` (clusters revived).
#' * `recipe`: the settings used, including `nepochs`, `batch_size`, `alpha0`,
#'   `epoch_rows` and `device`.
#' * `n_splits`: the total number of clusters revived.
#' * `spherical`: whether the fit is spherical.
#' * `call`: the matched call.
#'
#' The sums of squares use squared Euclidean distance and cover the rows used
#' for training. With `keep_cluster = FALSE` the fit has no `cluster`,
#' `totss`, `withinss`, `tot.withinss` or `betweenss`, and `size` counts the
#' rows of the last training pass. In spherical mode the centers have unit
#' length, so the sums of squares compare with `totss` only when the rows of
#' `x` do too.
#'
#' @section Devices:
#' On `"cpu"`, emeans runs C++ code in double precision. Every other device
#' runs the same algorithm in float32 through \pkg{torch} and returns an
#' ordinary `"emeans"` object, and [predict()][predict.emeans()] then assigns
#' new data on the same device. [set.seed()] still controls every random draw,
#' but the results differ slightly from the CPU because float32 rounds
#' differently, and on CUDA they can even differ from one run to the next.
#'
#' The CUDA path has not been tested on NVIDIA hardware, so please report what
#' you find at <https://github.com/coatless-rpkg/emeans/issues>. A
#' `torch_tensor` passed as `x` is not checked for `NA`, `NaN` or infinite
#' values, since that would mean copying it back from the device.
#'
#' @section Speed:
#' `threads` sets the number of CPU threads, up to any limit in the
#' `OMP_THREAD_LIMIT` environment variable. R for macOS is built without
#' OpenMP, so on a Mac emeans uses one thread.
#'
#' On the CPU, emeans uses a BLAS-based kernel when R is linked to an
#' optimised BLAS and a simpler direct kernel otherwise. Set
#' `options(emeans.kernel = "direct")` when you need results that are
#' identical to the last bit whatever the chunk sizes.
#'
#' @section Differences from upstream:
#' emeans ports the e-means code in faiss (`contrib/e_means`), with these
#' deliberate differences:
#'
#' 1. R's random number generator and [set.seed()] replace numpy's
#'    `RandomState` and the `seed` argument, so a run does not match Python
#'    for the same seed.
#' 2. The CPU path uses double precision throughout, where upstream uses
#'    float32 for the model and float64 for the objective.
#' 3. Cluster labels start at 1, and rows left out of training get `NA`
#'    rather than `-1`.
#' 4. A tie for the nearest center goes to the lowest index.
#' 5. The split-collision step moves to the next representable double, where
#'    upstream uses the next float32.
#' 6. `emeans()` ends with the extra sweep that labels every row by its
#'    nearest final center and adds the sums of squares. Upstream returns the
#'    labels of the last training pass, which [em_accumulate()] still gives.
#'
#' @references
#' Bessa S, Mazaré P-E, Szilvasy G, Touvron H, Picard D, Douze M, Jégou H
#' (2026). e-means: outperforming k-means (in one-tenth the epochs).
#' Preprint. <https://arxiv.org/abs/2611.09501>
#'
#' @seealso [predict.emeans()] and the other [emeans-methods];
#'   [em_begin()] for data that does not fit in memory; [emeans-broom] for
#'   `tidy()`, `glance()` and `augment()`; `vignette("emeans")` for a tour.
#'
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(200, sd = 0.3), ncol = 2),
#'            matrix(rnorm(200, mean = 3, sd = 0.3), ncol = 2))
#' fit <- emeans(x, 2, threads = 1)
#' fit$centers
#'
#' # Assign new points to the nearest center
#' predict(fit, rbind(c(0, 0), c(3, 3)))
#' @export
emeans <- function(x, k, nepochs = 5,
                   batch_law = c("auto", "throughput", "quality"),
                   batch_size = NULL, alpha0 = NULL, centers = NULL,
                   spherical = FALSE, unused_threshold = NULL,
                   max_points_per_centroid = NULL, keep_cluster = TRUE,
                   chunk_size = NULL, centroid_chunk_size = NULL,
                   device = "cpu", threads = NULL, verbose = FALSE) {
  call <- match.call()
  t0 <- elapsed_seconds()
  x_tensor <- is_torch_tensor(x)
  if (x_tensor) {
    if (!identical(resolve_device(device)$type, "torch")) {
      abort_emeans(
        "`x` is a torch_tensor; use a torch device such as \"mps\", \"cuda\" or \"torch:cpu\", or pass a matrix.",
        "emeans_error_input"
      )
    }
    x <- as_training_tensor(x, "x")
  } else {
    x <- as_training_matrix(x)
  }
  n_rows <- function(x) if (x_tensor) x$size(1L) else nrow(x)
  n_cols <- if (x_tensor) x$size(2L) else ncol(x)
  take_rows <- function(x, rows) {
    if (x_tensor) tensor_rows(x, rows) else x[rows, , drop = FALSE]
  }
  check_flag(keep_cluster, "keep_cluster")
  check_flag(verbose, "verbose")
  if (!is_count(k)) {
    abort_emeans("`k` must be a single whole number >= 1.", "emeans_error_input")
  }
  k <- as.integer(k)
  if (!is.null(max_points_per_centroid) && !is_count(max_points_per_centroid)) {
    abort_emeans("`max_points_per_centroid` must be NULL or a single whole number >= 1.",
                 "emeans_error_input")
  }
  n_full <- n_rows(x)
  if (!is.null(centers)) {
    centers <- as_training_matrix(centers, "centers")
    if (nrow(centers) != k || ncol(centers) != n_cols) {
      abort_emeans(
        sprintf("`centers` must be a %d x %d matrix; it is %d x %d.",
                k, n_cols, nrow(centers), ncol(centers)),
        "emeans_error_input"
      )
    }
    if (!x_tensor && !is.null(colnames(x))) {
      centers <- align_center_names(centers, colnames(x))
    }
  }

  # 1. Training subset.
  keep <- NULL
  limit <- if (is.null(max_points_per_centroid)) Inf else max_points_per_centroid * k
  if (n_full > limit) {
    if (verbose) {
      report_subsample(n_full, k, limit, max_points_per_centroid)
    }
    keep <- draw_subset(n_full, limit)
    x <- take_rows(x, keep)
  }
  n <- n_rows(x)
  if (k > n) {
    abort_emeans(
      sprintf("`k` (%d) must not exceed the number of training rows (%d).", k, n),
      "emeans_error_input"
    )
  }

  # 2. Starting centroids.
  how <- "given"
  if (is.null(centers)) {
    if (n == k) {
      centers <- x
      how <- "every_row"
    } else {
      centers <- take_rows(x, draw_init_rows(n, k))
      how <- "random"
    }
    if (x_tensor) {
      centers <- tensor_to_matrix(centers)
    }
  }

  # 3. State, header, preprocessing line.
  state <- em_begin(n, centers, nepochs = nepochs, batch_law = batch_law,
                    batch_size = batch_size, alpha0 = alpha0,
                    spherical = spherical, unused_threshold = unused_threshold,
                    chunk_size = chunk_size,
                    centroid_chunk_size = centroid_chunk_size, device = device,
                    threads = threads, verbose = verbose)
  # Labels come from the closing assignment pass below, not the last epoch.
  state$keep_labels <- FALSE
  state$call <- call
  data <- state$backend$prepare_data(x)
  if (verbose) {
    report_header_once(state)
    report_preprocessing(state, elapsed_seconds() - t0, how)
  } else {
    state$header_due <- FALSE
  }

  # 4. Epochs.
  recipe <- state$recipe
  batch <- recipe$batch_size
  for (e in seq_along(recipe$epoch_rows)) {
    perm <- draw_permutation(n)
    rows_e <- recipe$epoch_rows[e]
    for (s in seq(1, rows_e, by = batch)) {
      rows <- perm[s:min(s + batch - 1, rows_e)]
      em_update(state, accumulate_rows(state, data, rows))
    }
    em_end_epoch(state)
  }

  # 5. Result, with a closing assignment pass that labels every training row
  # by its nearest final center, as kmeans() does.
  fit <- em_finalize(state)
  fit$call <- call
  if (!keep_cluster) {
    return(fit)
  }
  final <- closing_assignment(state$backend, data, x, x_tensor, fit$centers,
                              isTRUE(fit$spherical))
  cluster <- final$cluster
  if (!is.null(keep)) {
    cluster <- rep(NA_integer_, n_full)
    cluster[keep] <- final$cluster
  }
  fit$cluster <- cluster
  fit$size <- final$size
  add_sums_of_squares(fit, final$withinss, final$totss)
}

#' Closing assignment pass over the training rows
#'
#' Labels every row by its nearest final center and returns the cluster
#' sizes, the within-cluster sums of squared Euclidean distances and the
#' total sum of squares. Spherical fits are scored by inner product, so their
#' squared distances to the unit-length centers are recovered from it.
#'
#' @param backend,data The training backend and its native data.
#' @param x The training rows, as a matrix or a torch tensor.
#' @param x_tensor Whether `x` is a torch tensor.
#' @param centers Final `k x d` centers.
#' @param spherical Whether the fit is spherical.
#' @return A list with `cluster`, `size`, `withinss` and `totss`.
#' @noRd
closing_assignment <- function(backend, data, x, x_tensor, centers, spherical) {
  k <- nrow(centers)
  res <- backend$assign(data, backend$as_native_matrix(centers), spherical)
  cluster <- as.integer(res$cluster)
  d2 <- as.double(res$score)
  if (spherical) {
    d2 <- pmax(row_sq_norms(x, x_tensor) - 2 * d2 + rowSums(centers^2)[cluster], 0)
  }
  withinss <- numeric(k)
  by_cluster <- rowsum(d2, cluster)
  withinss[as.integer(rownames(by_cluster))] <- by_cluster[, 1L]
  list(cluster = cluster, size = tabulate(cluster, k), withinss = withinss,
       totss = total_ss(x, x_tensor))
}

row_sq_norms <- function(x, x_tensor) {
  if (x_tensor) {
    tensor_to_vector(x$square()$sum(dim = 2L))
  } else {
    rowSums(x^2)
  }
}

total_ss <- function(x, x_tensor) {
  if (x_tensor) {
    centred <- x - x$mean(dim = 1L, keepdim = TRUE)
    as.double(centred$square()$sum()$item())
  } else {
    mu <- colMeans(x)
    sum(vapply(seq_len(ncol(x)), function(j) sum((x[, j] - mu[j])^2), numeric(1)))
  }
}

#' Add kmeans()-style sums of squares to a fit, after `centers`
#' @noRd
add_sums_of_squares <- function(fit, withinss, totss) {
  sums <- list(totss = totss, withinss = withinss,
               tot.withinss = sum(withinss), betweenss = totss - sum(withinss))
  first <- c("cluster", "centers")
  structure(c(unclass(fit)[first], sums, unclass(fit)[setdiff(names(fit), first)]),
            class = class(fit))
}

#' Give starting centers the column names of `x`
#'
#' Named centers must carry the same set of names as `x` and are reordered to
#' match; unnamed centers are used by position.
#'
#' @param centers Numeric matrix of starting centers.
#' @param x_names Column names of `x`.
#' @return `centers` with `x_names` as column names.
#' @noRd
align_center_names <- function(centers, x_names) {
  cn <- colnames(centers)
  if (!is.null(cn)) {
    if (!setequal(cn, x_names) || anyDuplicated(cn) > 0L) {
      abort_emeans("`centers` column names must match the column names of `x`.",
                   "emeans_error_input")
    }
    centers <- centers[, match(x_names, cn), drop = FALSE]
  }
  colnames(centers) <- x_names
  centers
}
