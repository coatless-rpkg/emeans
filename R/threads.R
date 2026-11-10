#' Resolve the number of threads to use
#'
#' `threads` defaults to `getOption("emeans.threads", 2L)`, is capped by
#' `OMP_THREAD_LIMIT`, and is 1 when the build has no OpenMP.
#'
#' @param threads Requested threads, or `NULL` for the option default.
#' @return An integer of at least 1.
#' @noRd
resolve_threads <- function(threads = NULL) {
  if (is.null(threads)) {
    threads <- getOption("emeans.threads", 2L)
  }
  if (!is.numeric(threads) || length(threads) != 1L || is.na(threads) || threads < 1) {
    abort_emeans("`threads` must be a single positive number.", "emeans_error_input")
  }
  threads <- as.integer(threads)
  if (cpp_openmp_max_threads() == 0L) {
    return(1L)
  }
  limit <- suppressWarnings(as.integer(Sys.getenv("OMP_THREAD_LIMIT", "")))
  if (!is.na(limit) && limit >= 1L) {
    threads <- min(threads, limit)
  }
  max(1L, threads)
}
