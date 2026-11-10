# Verbose output: upstream base.py's faiss-style lines, sent through message().
# Whole-number counts are formatted with "%.0f" because they are doubles that
# may exceed R's integer range.

#' The citation line printed before the header
#' @noRd
citation_line <- function() {
  paste0(
    "This is e-means. If you use our work or find it useful, please cite ",
    "the e-means paper (Bessa et al., see citation(\"emeans\"))."
  )
}

#' Format a number of passes: 3 -> "3", 1.4 -> "1.4"
#' @noRd
format_passes <- function(passes) {
  sprintf("%.4g", passes)
}

#' Describe how batches are assigned, for the preprocessing line
#' @noRd
describe_assignment <- function(state) {
  recipe <- state$recipe
  if (identical(recipe$backend, "cpu")) {
    threads <- recipe$threads
    return(sprintf(
      "C++ %s kernel on CPU, %d %s", recipe$kernel, threads,
      if (threads == 1L) "thread" else "threads"
    ))
  }
  sprintf("%s on %s", recipe$backend, recipe$device)
}

#' Print the citation and header lines (port of `_report_header`)
#' @noRd
report_header <- function(state) {
  recipe <- state$recipe
  law <- recipe$batch_law
  extras <- if (is.null(law)) {
    "batch given"
  } else if (identical(recipe$batch_law_requested, "auto")) {
    sprintf("%s batch law, auto for %s", law, recipe$batch_law_reason)
  } else {
    sprintf("%s batch law", law)
  }
  if (isTRUE(state$spherical)) {
    extras <- c(extras, "spherical")
  }
  message(citation_line())
  message(sprintf(
    "Clustering %.0f points in %.0fD to %.0f clusters, %s epochs (batch %.0f, alpha0 %.3g, %s)",
    state$n, as.double(state$d), as.double(state$k),
    format_passes(sum(recipe$epoch_rows) / state$n),
    recipe$batch_size, recipe$alpha0, paste(extras, collapse = ", ")
  ))
  invisible(NULL)
}

#' Print the header once, at the first call that really starts the run
#'
#' Clears `state$header_due` whether or not the state is verbose.
#' @noRd
report_header_once <- function(state) {
  if (isTRUE(state$header_due)) {
    state$header_due <- FALSE
    if (isTRUE(state$verbose)) {
      report_header(state)
    }
  }
  invisible(NULL)
}

#' Print the preprocessing line (port of `_report_preprocessing`)
#'
#' @param seconds Time from the start of training to the start of the epoch
#'   clock.
#' @param how `"given"` (centers supplied), `"every_row"` (`n == k`) or
#'   `"random"` (rows sampled from the data).
#' @noRd
report_preprocessing <- function(state, seconds, how) {
  init <- switch(how,
    given = "initial centroids given",
    every_row = "every row used as an initial centroid",
    random = "random initialization from the data",
    abort_emeans(sprintf("Unknown preprocessing `how`: \"%s\".", how),
                 "emeans_error_input")
  )
  message(sprintf(
    "  Preprocessing in %.2f s (%s; assignment: %s)",
    seconds, init, describe_assignment(state)
  ))
  invisible(NULL)
}

#' Print one epoch's line (port of `_report_epoch`)
#'
#' @param stat One epoch's statistics, with `obj` and `time`.
#' @param epoch 1-based index of the epoch just finished.
#' @noRd
report_epoch <- function(state, stat, epoch) {
  recipe <- state$recipe
  rows <- recipe$epoch_rows[epoch]
  partial <- if (rows == state$n) {
    ""
  } else {
    sprintf(" (%.0f of %.0f rows)", rows, state$n)
  }
  message(sprintf(
    "  Epoch %s / %s%s, iteration %.0f / %.0f (%.2f s): objective=%g",
    format_passes(sum(recipe$epoch_rows[seq_len(epoch)]) / state$n),
    format_passes(sum(recipe$epoch_rows) / state$n), partial,
    sum(recipe$epoch_steps[seq_len(epoch)]), recipe$total_steps,
    stat$time, stat$obj
  ))
  invisible(NULL)
}

#' Print the `max_points_per_centroid` subsampling warning line
#' @noRd
report_subsample <- function(n, k, limit, mppc) {
  message(sprintf(
    "WARNING clustering %.0f points to %.0f centroids: sampling %.0f points (max_points_per_centroid=%.0f)",
    as.double(n), as.double(k), as.double(limit), as.double(mppc)
  ))
  invisible(NULL)
}
