#' Signal an emeans error
#'
#' @param message Error message.
#' @param class Specific condition class, added in front of `"emeans_error"`.
#' @param call Call to report, `NULL` for none.
#' @noRd
abort_emeans <- function(message, class, call = NULL) {
  stop(structure(
    class = c(class, "emeans_error", "error", "condition"),
    list(message = message, call = call)
  ))
}
