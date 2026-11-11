# Device resolution and backend construction.

#' Load the {torch} namespace if it is installed
#' @return `TRUE` when the namespace is loaded.
#' @noRd
torch_namespace <- function() {
  isNamespaceLoaded("torch") || requireNamespace("torch", quietly = TRUE)
}

#' Is {torch} usable?
#'
#' True when {torch} is installed with its LibTorch libraries. A wrapper so
#' tests can mock it. Loading the torch namespace draws from R's RNG, so the
#' stream is restored afterwards: otherwise the first torch fit in a session
#' would get different permutations from later fits with the same seed.
#' @noRd
has_torch <- function() {
  with_preserved_seed(torch_namespace()) && torch::torch_is_installed()
}

#' Resolve a `device` argument
#'
#' `"cpu"` selects the C++ kernels. `"cuda"` (or `"cuda:<index>"`), `"mps"`,
#' `"torch:cpu"` and `torch_device` objects select the {torch} backend.
#'
#' @param device A device specification.
#' @return A list with `type` (the backend family, `"cpu"` or `"torch"`) and
#'   `device` (the label recorded as `recipe$device`); torch devices also
#'   carry `torch_device`.
#' @noRd
resolve_device <- function(device) {
  is_label <- is.character(device) && length(device) == 1L && !is.na(device)
  if (is_label && identical(device, "cpu")) {
    return(list(type = "cpu", device = "cpu"))
  }
  is_object <- inherits(device, "torch_device")
  if (!is_object && !(is_label && grepl("^(cuda(:[0-9]+)?|mps|torch:cpu)$", device))) {
    abort_emeans(
      "`device` must be \"cpu\", \"cuda\", \"mps\", \"torch:cpu\" or a torch_device.",
      "emeans_error_device"
    )
  }
  shown <- if (is_label) sprintf("\"%s\"", device) else "a torch_device"
  if (!has_torch()) {
    abort_emeans(
      sprintf(
        "`device = %s` needs the {torch} package; install it with install.packages(\"torch\") and then torch::install_torch().",
        shown
      ),
      "emeans_error_device"
    )
  }
  if (is_object) {
    type <- device$type
    index <- device$index
  } else if (identical(device, "torch:cpu")) {
    type <- "cpu"
    index <- NULL
  } else {
    type <- sub(":.*$", "", device)
    index <- if (grepl(":", device)) as.integer(sub("^.*:", "", device)) else NULL
  }
  if (identical(type, "cpu")) {
    label <- "torch:cpu"
    available <- TRUE
    index <- NULL
  } else if (identical(type, "mps")) {
    label <- "mps"
    available <- torch::backends_mps_is_available()
    index <- NULL
  } else if (identical(type, "cuda")) {
    label <- if (is.null(index)) "cuda" else sprintf("cuda:%d", index)
    available <- torch::cuda_is_available() &&
      (is.null(index) || index < torch::cuda_device_count())
  } else {
    abort_emeans(
      sprintf("torch device type \"%s\" is not supported; use \"cuda\", \"mps\" or \"cpu\".", type),
      "emeans_error_device"
    )
  }
  if (!isTRUE(available)) {
    abort_emeans(
      sprintf("Device \"%s\" is not available on this machine.", label),
      "emeans_error_device"
    )
  }
  torch_device <- if (is.null(index)) torch::torch_device(type) else torch::torch_device(type, index)
  list(type = "torch", device = label, torch_device = torch_device)
}

#' Build the backend for a resolved device
#'
#' @param device_info Result of `resolve_device()`.
#' @param kernel,threads,chunk_size,centroid_chunk_size Resolved execution
#'   settings. The torch backend ignores `kernel` and `threads`.
#' @return A backend (see `R/backend.R`).
#' @noRd
make_backend <- function(device_info, kernel, threads, chunk_size,
                         centroid_chunk_size) {
  if (identical(device_info$type, "cpu")) {
    return(cpu_backend(kernel, threads, chunk_size, centroid_chunk_size))
  }
  if (identical(device_info$type, "torch")) {
    return(torch_backend(device_info$torch_device, chunk_size, centroid_chunk_size))
  }
  abort_emeans(
    sprintf("No backend for device type \"%s\".", device_info$type),
    "emeans_error_device"
  )
}
