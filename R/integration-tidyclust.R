# tidyclust "emeans" engine for k_means(), registered lazily (tidyclust and
# modelenv are Suggests only).

#' Register the emeans engine for tidyclust's k_means()
#'
#' Idempotent; does nothing unless tidyclust and modelenv are available and
#' tidyclust's own `k_means` model has been registered.
#' @return `NULL`, invisibly.
#' @noRd
register_tidyclust_engine <- function() {
  if (!requireNamespace("modelenv", quietly = TRUE) ||
      !requireNamespace("tidyclust", quietly = TRUE)) {
    return(invisible(NULL))
  }
  env <- modelenv::get_model_env()
  if (!exists("k_means", envir = env, inherits = FALSE)) {
    return(invisible(NULL))
  }
  engines <- env[["k_means"]]
  if ("emeans" %in% engines$engine) {
    return(invisible(NULL))
  }

  modelenv::set_model_engine("k_means", "partition", "emeans")
  modelenv::set_dependency("k_means", "emeans", pkg = "emeans", mode = "partition")
  modelenv::set_dependency("k_means", "emeans", pkg = "tidyclust", mode = "partition")
  modelenv::set_fit(
    model = "k_means", eng = "emeans", mode = "partition",
    value = list(
      interface = "matrix",
      data = c(x = "data"),
      protect = c("data", "k"),
      func = c(pkg = "emeans", fun = ".tidyclust_fit_emeans"),
      defaults = list()
    )
  )
  modelenv::set_encoding(
    model = "k_means", eng = "emeans", mode = "partition",
    options = list(
      predictor_indicators = "traditional",
      compute_intercept = TRUE,
      remove_intercept = TRUE,
      allow_sparse_x = FALSE
    )
  )
  modelenv::set_model_arg(
    model = "k_means", eng = "emeans",
    exposed = "num_clusters", original = "k",
    func = list(pkg = "dials", fun = "num_clusters"),
    has_submodel = FALSE
  )
  modelenv::set_pred(
    model = "k_means", eng = "emeans", mode = "partition", type = "cluster",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(pkg = "emeans", fun = ".tidyclust_predict_emeans"),
      args = list(object = quote(object$fit), new_data = quote(new_data))
    )
  )
  invisible(NULL)
}

#' Fit and predict wrappers used by tidyclust
#'
#' Internal entry points that tidyclust calls as `emeans::<fun>`; not meant
#' for direct use. The fit wrapper calls [emeans()], whose `withinss` and
#' `totss` tidyclust reads.
#' @param data Numeric matrix (or data frame) to cluster.
#' @param k Number of clusters.
#' @param ... Passed on to [emeans()].
#' @return `.tidyclust_fit_emeans()` returns an `"emeans"` object with
#'   `withinss` and `totss` added; `.tidyclust_predict_emeans()` returns a
#'   factor of cluster labels.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(200), ncol = 2)
#' fit <- .tidyclust_fit_emeans(x, 2, threads = 1)
#' fit$withinss
#' .tidyclust_predict_emeans(fit, x[1:5, ])
#' @keywords internal
#' @export
.tidyclust_fit_emeans <- function(data, k, ...) {
  if (is.null(k)) {
    abort_emeans("Please specify `num_clusters` to fit the model.",
                 "emeans_error_input")
  }
  emeans(as.matrix(data), k, ...)
}

make_emeans_labels <- function(n_clusters, prefix, labels) {
  if (is.null(labels)) {
    return(paste0(prefix, seq_len(n_clusters)))
  }
  if (length(labels) < n_clusters) {
    abort_emeans(
      sprintf("`labels` must have length %d, not %d.", n_clusters, length(labels)),
      "emeans_error_input"
    )
  }
  labels <- labels[seq_len(n_clusters)]
  if (anyDuplicated(labels)) {
    abort_emeans("`labels` must not contain duplicate values.",
                 "emeans_error_input")
  }
  labels
}

#' @rdname dot-tidyclust_fit_emeans
#' @param object An `"emeans"` object.
#' @param new_data Observations to label.
#' @param prefix,labels Cluster naming, as in tidyclust.
#' @keywords internal
#' @export
.tidyclust_predict_emeans <- function(object, new_data, prefix = "Cluster_",
                                      labels = NULL) {
  k <- nrow(object$centers)
  factor(
    predict(object, as.matrix(new_data)),
    levels = seq_len(k),
    labels = make_emeans_labels(k, prefix, labels)
  )
}

#' Summary of an emeans fit for tidyclust
#'
#' Method for [tidyclust::extract_fit_summary()].
#' @param object An `"emeans"` object fitted through tidyclust.
#' @param ... Unused.
#' @param prefix Prefix for the default cluster names.
#' @param labels Optional cluster names.
#' @return A list with `cluster_names`, `centroids`, `n_members`,
#'   `sse_within_total_total`, `sse_total`, `orig_labels` and
#'   `cluster_assignments`.
#' @examplesIf requireNamespace("tidyclust", quietly = TRUE)
#' set.seed(1)
#' fit <- emeans(matrix(rnorm(200), ncol = 2), 2, threads = 1)
#' str(tidyclust::extract_fit_summary(fit))
#' @exportS3Method tidyclust::extract_fit_summary
#' @keywords internal
extract_fit_summary.emeans <- function(object, ..., prefix = "Cluster_",
                                       labels = NULL) {
  if (is.null(object$cluster)) {
    abort_emeans("Cluster labels were not kept; refit with `keep_cluster = TRUE`.",
                 "emeans_error_input")
  }
  nms <- factor(make_emeans_labels(nrow(object$centers), prefix, labels))
  list(
    cluster_names = nms,
    centroids = as.data.frame(object$centers),
    n_members = object$size,
    sse_within_total_total = object$withinss,
    sse_total = object$totss,
    orig_labels = unname(object$cluster),
    cluster_assignments = factor(nms[object$cluster], levels = levels(nms))
  )
}

#' Cluster assignments of an emeans fit for tidyclust
#'
#' Method for [tidyclust::extract_cluster_assignment()]. Returns a tibble
#' with one `.cluster` factor column.
#' @inheritParams extract_fit_summary.emeans
#' @return A tibble (built without requiring the tibble package at load).
#' @examplesIf requireNamespace("tidyclust", quietly = TRUE)
#' set.seed(1)
#' fit <- emeans(matrix(rnorm(200), ncol = 2), 2, threads = 1)
#' head(tidyclust::extract_cluster_assignment(fit))
#' @exportS3Method tidyclust::extract_cluster_assignment
#' @keywords internal
extract_cluster_assignment.emeans <- function(object, ..., prefix = "Cluster_",
                                              labels = NULL) {
  if (is.null(object$cluster)) {
    abort_emeans("Cluster labels were not kept; refit with `keep_cluster = TRUE`.",
                 "emeans_error_input")
  }
  nms <- make_emeans_labels(nrow(object$centers), prefix, labels)
  out <- data.frame(.cluster = factor(nms[object$cluster], levels = nms))
  class(out) <- c("tbl_df", "tbl", "data.frame")
  out
}
