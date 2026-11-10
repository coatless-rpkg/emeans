# Default chunk sizes. The BLAS kernel's score block is at most
# chunk_size x centroid_chunk_size doubles: 4096 * 1024 * 8 bytes = 32 MB.
default_chunk_size <- 4096L
default_centroid_chunk_size <- 1024L

#' Resolve the row and centroid chunk sizes
#'
#' @param chunk_size,centroid_chunk_size A single finite whole number >= 1,
#'   or `NULL` for the default.
#' @return A list with integer `chunk_size` and `centroid_chunk_size`.
#' @noRd
resolve_chunk_sizes <- function(chunk_size = NULL, centroid_chunk_size = NULL) {
  check_one <- function(value, default, name) {
    if (is.null(value)) {
      return(default)
    }
    ok <- is.numeric(value) && length(value) == 1L && is.finite(value) &&
      value >= 1 && value == round(value) && value <= .Machine$integer.max
    if (!ok) {
      abort_emeans(
        sprintf("`%s` must be a single whole number >= 1.", name),
        "emeans_error_input"
      )
    }
    as.integer(value)
  }
  list(
    chunk_size = check_one(chunk_size, default_chunk_size, "chunk_size"),
    centroid_chunk_size = check_one(
      centroid_chunk_size, default_centroid_chunk_size, "centroid_chunk_size"
    )
  )
}

# Thin wrapper so tests can mock the BLAS name.
blas_name <- function() {
  extSoftVersion()[["BLAS"]]
}

#' Is R linked against an optimised BLAS?
#'
#' R cannot always name its BLAS (it may show as "" or "libR", and detection
#' does not work on Windows), so this is an allow-list: only a BLAS whose name
#' matches a known optimised library counts. Anything else is treated as the
#' reference BLAS.
#'
#' @return `TRUE` or `FALSE`.
#' @noRd
blas_is_optimised <- function() {
  grepl(
    "vecLib|Accelerate|openblas|mkl|blis|atlas|flexiblas|armpl",
    blas_name(),
    ignore.case = TRUE
  )
}

#' Choose the CPU assignment kernel
#'
#' The `"auto"` rule picks `"blas"` when `d >= 2` and the BLAS is optimised,
#' else `"direct"`. Measured in `tools/bench/results.md`.
#'
#' @param kernel `"auto"`, `"direct"` or `"blas"`.
#' @param d Number of columns.
#' @return `"direct"` or `"blas"`.
#' @noRd
resolve_kernel <- function(kernel = getOption("emeans.kernel", "auto"), d) {
  if (!is.character(kernel) || length(kernel) != 1L || !kernel %in% c("auto", "direct", "blas")) {
    abort_emeans(
      "`kernel` must be one of \"auto\", \"direct\" or \"blas\".",
      "emeans_error_input"
    )
  }
  if (kernel != "auto") {
    return(kernel)
  }
  if (d >= 2 && blas_is_optimised()) "blas" else "direct"
}
