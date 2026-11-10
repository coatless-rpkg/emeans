# CPU backend: base R doubles plus the C++ assignment kernel.

#' EMA update of centroids and usage (port of `_apply_statistics`)
#' @noRd
ema_update_r <- function(centroids, usage, sums, counts, alpha, replace_usage) {
  touched <- which(counts > 0)
  if (!replace_usage) {
    usage <- usage * (1 - alpha)
  }
  if (length(touched) == 0L) {
    return(list(centroids = centroids, usage = usage))
  }
  n <- counts[touched]
  retained <- if (replace_usage) 0 else usage[touched]
  mass <- retained + alpha * n
  mean <- sums[touched, , drop = FALSE] / n
  old <- centroids[touched, , drop = FALSE]
  weight <- alpha * n / mass
  position <- old + weight * (mean - old)
  exact <- weight == 1
  position[exact, ] <- mean[exact, , drop = FALSE]
  centroids[touched, ] <- position
  usage[touched] <- mass
  list(centroids = centroids, usage = usage)
}

#' Alternating-sign relative offsets for donor splits
#' @noRd
split_offsets <- function(d) {
  emeans_constants$split_epsilon * rep_len(c(1, -1), d)
}

#' Separate identical donor/child rows on coordinate one
#' @return The adjusted child matrix.
#' @noRd
separate_splits <- function(donor, child) {
  collapsed <- rowSums(donor == child) == ncol(child)
  value <- child[, 1L]
  neighbor <- cpp_nextafter(value, Inf)
  bad <- !is.finite(neighbor)
  if (any(bad)) {
    neighbor[bad] <- cpp_nextafter(value[bad], -Inf)
  }
  child[, 1L] <- ifelse(collapsed, neighbor, value)
  child
}

#' Split donors into targets (port of `_apply_splits`)
#' @noRd
apply_splits_r <- function(centroids, usage, source, target) {
  src <- centroids[source, , drop = FALSE]
  offset <- rep(split_offsets(ncol(centroids)), each = nrow(src))
  donor <- src * (1 - offset)
  child <- separate_splits(donor, src * (1 + offset))
  half <- usage[source] / 2
  centroids[source, ] <- donor
  usage[source] <- half
  centroids[target, ] <- child
  usage[target] <- half
  list(centroids = centroids, usage = usage)
}

#' Project rows to unit length, prescaling by the max absolute value
#' @noRd
normalize_rows <- function(m) {
  if (nrow(m) == 0L) {
    return(m)
  }
  top <- apply(abs(m), 1L, max)
  m <- m / ifelse(top > 0, top, 1)
  norm <- sqrt(rowSums(m * m))
  m / ifelse(norm > 0, norm, 1)
}

#' Build the CPU backend
#'
#' Receives already-resolved kernel, threads and chunk sizes.
#' @noRd
cpu_backend <- function(kernel, threads, chunk_size, centroid_chunk_size) {
  force(kernel); force(threads); force(chunk_size); force(centroid_chunk_size)
  identity_fn <- function(a) a
  be <- list(
    name = "cpu",
    prepare_data = function(x) {
      if (!is.matrix(x)) {
        x <- as.matrix(x)
      }
      if (!is.double(x)) {
        storage.mode(x) <- "double"
      }
      x
    },
    n_rows = function(data) nrow(data),
    n_cols = function(data) ncol(data),
    as_native_matrix = function(m) {
      m <- as.matrix(m)
      storage.mode(m) <- "double"
      m
    },
    as_native_vector = function(v) as.double(v),
    as_r_matrix = identity_fn,
    as_r_vector = identity_fn,
    zeros_matrix = function(k, d) matrix(0, k, d),
    zeros_vector = function(k) numeric(k),
    add = function(a, b) a + b,
    accumulate = function(data, rows, readout, acc, spherical, want_labels) {
      res <- cpp_assign_accumulate(
        data, as.integer(rows), readout, spherical, kernel, chunk_size,
        centroid_chunk_size, threads, acc$sums, acc$counts, acc$objective
      )
      list(
        acc = list(sums = res$sums, counts = res$counts, objective = res$objective),
        labels = if (want_labels) res$labels else NULL
      )
    },
    update = function(centroids, usage, sums, counts, alpha, replace_usage) {
      ema_update_r(centroids, usage, sums, counts, alpha, replace_usage)
    },
    split = function(centroids, usage, source, target) {
      apply_splits_r(centroids, usage, source, target)
    },
    readout = function(centroids, spherical) {
      if (spherical) normalize_rows(centroids) else centroids
    },
    closing = function(pass_sums, pass_counts, centroids, spherical) {
      has <- pass_counts > 0
      mean <- centroids
      mean[has, ] <- pass_sums[has, , drop = FALSE] / pass_counts[has]
      if (spherical) normalize_rows(mean) else mean
    },
    assign = function(data, readout, spherical) {
      res <- cpp_assign_accumulate(
        data, seq_len(nrow(data)), readout, spherical, kernel, chunk_size,
        centroid_chunk_size, threads, NULL, NULL, 0
      )
      list(cluster = res$labels, score = res$score)
    }
  )
  check_backend(be)
  be
}
