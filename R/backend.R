# The backend interface. A backend is a list of functions that is the only
# way the training state machine touches data and centroids. "Native" means
# the backend's own storage type (base R doubles for the CPU backend).
#
#   name                     backend label.
#   prepare_data(x)          user matrix -> native data.
#   n_rows(data), n_cols(data)
#   as_native_matrix(m), as_native_vector(v)   R -> native.
#   as_r_matrix(a), as_r_vector(a)             native -> R.
#   zeros_matrix(k, d), zeros_vector(k), add(a, b)
#   accumulate(data, rows, readout, acc, spherical, want_labels)
#       -> list(acc = list(sums, counts, objective), labels)
#       `rows` are 1-based; labels may be NULL when `want_labels` is FALSE.
#   update(centroids, usage, sums, counts, alpha, replace_usage)
#       -> list(centroids, usage)   EMA step (Equation 2).
#   split(centroids, usage, source, target) -> list(centroids, usage)
#   readout(centroids, spherical)  centroids used for assignment.
#   closing(pass_sums, pass_counts, centroids, spherical)  final centroids.
#   assign(data, readout, spherical) -> list(cluster, score) over all rows.
#
# `acc$objective` is a native scalar.

backend_fields <- c(
  "name", "prepare_data", "n_rows", "n_cols", "as_native_matrix",
  "as_native_vector", "as_r_matrix", "as_r_vector", "zeros_matrix",
  "zeros_vector", "add", "accumulate", "update", "split", "readout",
  "closing", "assign"
)

#' Validate that a backend provides every interface field
#' @param be A backend list.
#' @return `be`, invisibly.
#' @noRd
check_backend <- function(be) {
  missing <- setdiff(backend_fields, names(be))
  if (!is.list(be) || length(missing) > 0L) {
    abort_emeans(
      sprintf("Backend is missing field(s): %s.", paste(missing, collapse = ", ")),
      "emeans_error_input"
    )
  }
  fns <- setdiff(backend_fields, "name")
  bad <- fns[!vapply(be[fns], is.function, logical(1))]
  if (length(bad) > 0L) {
    abort_emeans(
      sprintf("Backend field(s) must be functions: %s.", paste(bad, collapse = ", ")),
      "emeans_error_input"
    )
  }
  invisible(be)
}
