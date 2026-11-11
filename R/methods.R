#' Methods for e-means fits
#'
#' @param object,x An `"emeans"` object (`x` for `print()`).
#' @param newdata Numeric matrix or data frame with the same columns as the
#'   training data. If both `newdata` and the centers have column names, the
#'   columns are matched by name. On a torch device it may also be a 2-D
#'   `torch_tensor`, which is cast to float32 and not checked for `NA`.
#' @param type `"cluster"` for the nearest-center label, or `"distance"` for
#'   the score of that center: squared Euclidean distance, or inner-product
#'   similarity when the fit is spherical.
#' @param device Device used to assign `newdata`; by default the fit's own
#'   (see "Devices" in [emeans()]).
#' @param method `"centers"` for each row's center, or `"classes"` for labels.
#' @param ... Unused.
#' @return `predict()` returns an integer vector of labels or a numeric vector
#'   of scores. `fitted()` returns a matrix or an integer vector. `summary()`
#'   returns an object of class `"summary.emeans"`.
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(200, sd = 0.3), ncol = 2),
#'            matrix(rnorm(200, mean = 3, sd = 0.3), ncol = 2))
#' fit <- emeans(x, 2, threads = 1)
#' fit
#' summary(fit)
#'
#' # Label new points, or score them by their nearest center
#' newx <- rbind(c(0, 0), c(3, 3))
#' predict(fit, newx)
#' predict(fit, newx, type = "distance")
#'
#' # Each training row's center, or its cluster
#' head(fitted(fit))
#' head(fitted(fit, method = "classes"))
#' @name emeans-methods
#' @importFrom stats predict fitted
NULL

#' @rdname emeans-methods
#' @exportS3Method stats::predict
predict.emeans <- function(object, newdata, type = c("cluster", "distance"),
                           device = object$recipe$device, ...) {
  type <- match.arg(type)
  if (missing(newdata)) {
    abort_emeans("`newdata` is required.", "emeans_error_input")
  }
  device_info <- resolve_device(device)
  cen <- object$centers
  if (is_torch_tensor(newdata)) {
    if (!identical(device_info$type, "torch")) {
      abort_emeans(
        "`newdata` is a torch_tensor; use a torch `device` or pass a matrix.",
        "emeans_error_input"
      )
    }
    newdata <- as_training_tensor(newdata, "newdata")
    width <- newdata$size(2L)
  } else {
    newdata <- as_finite_matrix(newdata, "newdata")
    nn <- colnames(newdata)
    cn <- colnames(cen)
    if (!is.null(nn) && !is.null(cn)) {
      if (!setequal(nn, cn) || anyDuplicated(nn) || anyDuplicated(cn)) {
        abort_emeans("Column names of `newdata` do not match the training columns.",
                     "emeans_error_input")
      }
      newdata <- newdata[, cn, drop = FALSE]
    }
    if (!is.double(newdata)) storage.mode(newdata) <- "double"
    width <- ncol(newdata)
  }
  if (width != ncol(cen)) {
    abort_emeans(
      sprintf("`newdata` has %d column(s) but the fit has %d.", width, ncol(cen)),
      "emeans_error_input"
    )
  }
  rec <- object$recipe
  kernel <- rec$kernel
  if (identical(device_info$type, "cpu") && (is.null(kernel) || is.na(kernel))) {
    kernel <- resolve_kernel(d = ncol(cen))
  }
  backend <- make_backend(device_info, kernel, rec$threads,
                          rec$chunk_size, rec$centroid_chunk_size)
  res <- backend$assign(backend$prepare_data(newdata),
                        backend$as_native_matrix(cen), object$spherical)
  if (type == "cluster") res$cluster else res$score
}

#' @rdname emeans-methods
#' @exportS3Method stats::fitted
fitted.emeans <- function(object, method = c("centers", "classes"), ...) {
  method <- match.arg(method)
  if (is.null(object$cluster)) {
    abort_emeans("Cluster labels were not kept; refit with `keep_cluster = TRUE`.",
                 "emeans_error_input")
  }
  if (method == "centers") {
    object$centers[object$cluster, , drop = FALSE]
  } else {
    object$cluster
  }
}

`%||%` <- function(x, y) if (is.null(x)) y else x

# One line on how well the fit does: the kmeans()-style sums of squares when
# the closing pass ran, or else the objective of the last training pass.
fit_quality <- function(object) {
  if (is.null(object$tot.withinss)) {
    obj <- object$objective %||% final_objective(object)
    return(sprintf("last-pass objective %s", format(obj, digits = 6)))
  }
  sprintf("tot.withinss %s (between_SS / total_SS = %.1f %%)",
          format(object$tot.withinss, digits = 6),
          100 * (1 - object$tot.withinss / object$totss))
}

final_objective <- function(object) {
  h <- object$history
  if (is.data.frame(h) && nrow(h) > 0L) h$obj[nrow(h)] else NA_real_
}

#' @rdname emeans-methods
#' @exportS3Method base::print
print.emeans <- function(x, ...) {
  r <- x$recipe
  cat(sprintf("e-means clustering: k = %d, %.0f rows%s\n", nrow(x$centers),
              sum(as.double(x$size)), if (is.null(x$tot.withinss)) " in the last pass" else ""))
  cat(sprintf("  epochs %s, batch %s, alpha0 %s\n", format(r$nepochs), format(r$batch_size),
              sprintf("%.3g", r$alpha0)))
  cat(sprintf("  %s, n_splits %s\n", fit_quality(x), format(x$n_splits)))
  cat(sprintf("  device %s%s\n", r$device, if (isTRUE(x$spherical)) ", spherical" else ""))
  invisible(x)
}

#' @rdname emeans-methods
#' @exportS3Method base::summary
summary.emeans <- function(object, ...) {
  structure(
    list(size = object$size, history = object$history, recipe = object$recipe,
         n_splits = object$n_splits, spherical = object$spherical,
         objective = final_objective(object), totss = object$totss,
         tot.withinss = object$tot.withinss, k = nrow(object$centers),
         call = object$call),
    class = "summary.emeans"
  )
}

#' @rdname emeans-methods
#' @exportS3Method base::print
print.summary.emeans <- function(x, ...) {
  r <- x$recipe
  cat("Summary of e-means fit\n")
  if (!is.null(x$call)) cat("Call: ", paste(deparse(x$call), collapse = " "), "\n", sep = "")
  cat(sprintf("k = %d, epochs %s, batch %s, alpha0 %s, device %s\n", x$k,
              format(r$nepochs), format(r$batch_size), sprintf("%.3g", r$alpha0), r$device))
  cat(sprintf("%s, n_splits %s\n", fit_quality(x), format(x$n_splits)))
  cat("Cluster size:\n")
  print(x$size)
  cat("History (last rows):\n")
  print(x$history[seq.int(max(1L, nrow(x$history) - 4L), nrow(x$history)), , drop = FALSE], row.names = FALSE)
  invisible(x)
}
