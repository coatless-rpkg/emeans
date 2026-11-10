#' @importFrom stats runif
#' @keywords internal
#' @noRd
NULL

#' Evaluate an expression without moving R's random-number stream
#'
#' Restores `.Random.seed` (or its absence) afterwards, for code outside the
#' package that draws random numbers as a side effect, such as loading the
#' {torch} namespace.
#' @param expr Expression to evaluate.
#' @return The value of `expr`.
#' @noRd
with_preserved_seed <- function(expr) {
  env <- globalenv()
  if (exists(".Random.seed", envir = env, inherits = FALSE)) {
    seed <- get(".Random.seed", envir = env, inherits = FALSE)
    on.exit(assign(".Random.seed", seed, envir = env))
  } else {
    on.exit(if (exists(".Random.seed", envir = env, inherits = FALSE)) {
      rm(".Random.seed", envir = env)
    })
  }
  expr
}

#' Record a random draw
#'
#' Appends `list(type = type, value = value)` to the `draws` list of the
#' environment in `getOption("emeans.rng_recorder")`. Does nothing when the
#' option is unset or is not an environment.
#'
#' @param type Character label for the draw.
#' @param value The drawn values.
#' @noRd
rng_record <- function(type, value) {
  rec <- getOption("emeans.rng_recorder")
  if (!is.environment(rec)) {
    return(invisible(NULL))
  }
  if (is.null(rec$draws)) {
    rec$draws <- list()
  }
  rec$draws[[length(rec$draws) + 1L]] <- list(type = type, value = value)
  invisible(NULL)
}

#' Draw the training subset
#'
#' @param n Number of rows to choose from.
#' @param limit Number of rows to keep.
#' @return Sorted integer indices of length `limit`.
#' @noRd
draw_subset <- function(n, limit) {
  value <- sort(sample.int(n, limit))
  rng_record("subset", value)
  value
}

#' Draw initial centroid rows
#'
#' @param n Number of rows to choose from.
#' @param k Number of rows to draw.
#' @return Integer indices of length `k`.
#' @noRd
draw_init_rows <- function(n, k) {
  value <- sample.int(n, k)
  rng_record("init", value)
  value
}

#' Draw a permutation of rows
#'
#' @param n Number of rows to permute.
#' @return Integer permutation of `seq_len(n)`.
#' @noRd
draw_permutation <- function(n) {
  value <- sample.int(n)
  rng_record("permutation", value)
  value
}

#' Draw revival uniforms
#'
#' @param k Number of uniforms to draw.
#' @return Double vector of length `k` on the unit interval.
#' @noRd
draw_revival_uniforms <- function(k) {
  value <- runif(k)
  rng_record("revival", value)
  value
}
