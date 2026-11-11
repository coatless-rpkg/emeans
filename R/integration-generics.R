# broom-style methods, registered lazily on {generics} (a Suggests package).

#' Tidy, glance and augment methods for emeans fits
#'
#' Methods for the [generics::tidy()], [generics::glance()] and
#' [generics::augment()] generics, which return plain data frames and need
#' only the \pkg{generics} package to be installed.
#'
#' @param x An `"emeans"` object.
#' @param data The data that was clustered, as a data frame or matrix. Used
#'   when `newdata` is `NULL`; needs a fit made with `keep_cluster = TRUE`.
#' @param newdata Optional new observations, labelled with [predict()].
#' @param ... Unused.
#' @return `tidy()` returns one row per cluster with the center columns,
#'   `size`, `withinss` and `cluster` (a factor). `glance()` returns one row
#'   with `totss`, `tot.withinss`, `betweenss`, `obj` (the last pass's
#'   objective), `iter`, `epochs` and `n_splits`; the sums of squares are `NA`
#'   for a fit made with `keep_cluster = FALSE`, which also has no `withinss`
#'   column in `tidy()`. `augment()` returns the data with a
#'   `.cluster` factor column.
#' @examples
#' if (requireNamespace("generics", quietly = TRUE)) {
#'   set.seed(1)
#'   x <- data.frame(a = c(rnorm(100), rnorm(100, mean = 4)),
#'                   b = c(rnorm(100), rnorm(100, mean = 4)))
#'   fit <- emeans(x, 2, threads = 1)
#'   generics::tidy(fit)
#'   generics::glance(fit)
#'   head(generics::augment(fit, data = x))
#' }
#' @name emeans-broom
NULL

#' @rdname emeans-broom
#' @exportS3Method generics::tidy
tidy.emeans <- function(x, ...) {
  centers <- x$centers
  k <- nrow(centers)
  out <- as.data.frame(centers, stringsAsFactors = FALSE)
  names(out) <- if (is.null(colnames(centers))) {
    paste0("V", seq_len(ncol(centers)))
  } else {
    colnames(centers)
  }
  out$size <- as.integer(x$size)
  if (!is.null(x$withinss)) {
    out$withinss <- x$withinss
  }
  out$cluster <- factor(seq_len(k), levels = seq_len(k))
  rownames(out) <- NULL
  out
}

#' @rdname emeans-broom
#' @exportS3Method generics::glance
glance.emeans <- function(x, ...) {
  h <- x$history
  data.frame(
    totss = x$totss %||% NA_real_,
    tot.withinss = x$tot.withinss %||% NA_real_,
    betweenss = x$betweenss %||% NA_real_,
    obj = h$obj[nrow(h)],
    iter = x$iter,
    epochs = h$epoch[nrow(h)],
    n_splits = x$n_splits
  )
}

#' @rdname emeans-broom
#' @exportS3Method generics::augment
augment.emeans <- function(x, data = NULL, newdata = NULL, ...) {
  k <- nrow(x$centers)
  if (!is.null(newdata)) {
    labels <- predict(x, as.matrix(newdata))
    out <- as.data.frame(newdata, stringsAsFactors = FALSE)
  } else {
    if (is.null(data)) {
      abort_emeans("`data` or `newdata` is required.", "emeans_error_input")
    }
    if (is.null(x$cluster)) {
      abort_emeans(
        "The fit has no stored clusters; refit with `keep_cluster = TRUE` or supply `newdata`.",
        "emeans_error_input"
      )
    }
    out <- as.data.frame(data, stringsAsFactors = FALSE)
    if (nrow(out) != length(x$cluster)) {
      abort_emeans(
        sprintf("`data` has %d rows but the fit has %d cluster labels.",
                nrow(out), length(x$cluster)),
        "emeans_error_input"
      )
    }
    labels <- x$cluster
  }
  out$.cluster <- factor(labels, levels = seq_len(k))
  out
}
