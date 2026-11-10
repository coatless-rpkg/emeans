# Recipe: scaling laws, epoch rows and cosine schedule (port of upstream
# base.py). All row and step counts are doubles.

#' Round a positive value to the nearest power of two
#' @noRd
round_power_of_two <- function(value) {
  if (value <= 1) {
    return(1)
  }
  low <- 2^floor(log2(value))
  high <- 2 * low
  if (value / low < high / value) low else high
}

#' Initial EMA step coupled to a batch size by the fitted step law
#' @noRd
step_law <- function(n, k, nepochs, batch_size) {
  a <- emeans_constants$alpha_law
  alpha0 <- a[1] * k^a[2] * batch_size^a[3] * n^a[4] * nepochs^a[5]
  min(max(alpha0, 1e-4), 1)
}

#' Resolve `batch_law = "auto"` from the dimension
#' @noRd
resolve_batch_law <- function(batch_law, d) {
  if (!identical(batch_law, "auto")) {
    return(list(law = batch_law, reason = "requested"))
  }
  cutoff <- emeans_constants$auto_quality_max_d
  if (d <= cutoff) {
    list(law = "quality", reason = paste0("d <= ", cutoff))
  } else {
    list(law = "throughput", reason = paste0("d > ", cutoff))
  }
}

#' Revival threshold for a run, picked from `d` if `NULL`
#' @noRd
resolve_unused_threshold <- function(unused_threshold, d) {
  if (is.null(unused_threshold)) {
    return(if (d <= emeans_constants$no_revival_max_d) {
      0
    } else {
      emeans_constants$unused_threshold
    })
  }
  if (!is.numeric(unused_threshold) || length(unused_threshold) != 1L ||
      !isTRUE(unused_threshold >= 0)) {
    abort_emeans(
      "`unused_threshold` must be a single number >= 0 (0 disables revival).",
      "emeans_error_input"
    )
  }
  as.double(unused_threshold)
}

#' Resolve the statistical batch size and initial EMA step
#' @noRd
resolve_recipe <- function(n, k, nepochs, batch_law, batch_size = NULL,
                           alpha0 = NULL) {
  n <- as.double(n)
  if (is.null(batch_size)) {
    laws <- emeans_constants$batch_laws
    if (!is.character(batch_law) || length(batch_law) != 1L ||
        !batch_law %in% names(laws)) {
      abort_emeans(
        paste0("`batch_law` must be one of ",
               paste0('"', c("auto", names(laws)), '"', collapse = ", "), "."),
        "emeans_error_input"
      )
    }
    cf <- laws[[batch_law]]
    raw_batch <- cf[1] * k^cf[2] * n^cf[3] * nepochs^cf[4]
    batch_size <- round_power_of_two(raw_batch)
    # These intentionally differ: they reproduce the two fitted recipes.
    batch_size <- if (batch_law == "throughput") {
      min(batch_size, round_power_of_two(n / 2))
    } else {
      min(batch_size, n %/% 2)
    }
    batch_size <- max(1, min(batch_size, n))
  } else {
    batch_size <- min(trunc(as.double(batch_size)), n)
  }
  if (is.null(alpha0)) {
    alpha0 <- step_law(n, k, nepochs, batch_size)
  }
  list(batch_size = as.double(batch_size), alpha0 = as.double(alpha0))
}

#' Rows per epoch: whole passes, a rounded fractional prefix, a closing pass
#' @noRd
epoch_rows <- function(n, nepochs) {
  n <- as.double(n)
  whole <- floor(nepochs)
  remainder <- round((nepochs - whole) * n)
  rows <- if (whole == 0) {
    remainder
  } else {
    c(rep(n, whole - 1), if (remainder > 0) remainder, n)
  }
  if (any(rows < 1)) {
    abort_emeans(
      "`nepochs` is too small: an epoch would have zero rows.",
      "emeans_error_input"
    )
  }
  as.double(rows)
}

#' Cosine-anneal an EMA step towards one tenth of its initial value
#' @noRd
scheduled_alpha <- function(alpha0, step, total_steps) {
  fraction <- min(step, total_steps) / max(1, total_steps)
  cosine <- 0.5 * (1 + cos(pi * fraction))
  f <- emeans_constants$alpha_min_factor
  alpha0 * (f + (1 - f) * cosine)
}

#' Algorithmic part of the training recipe
#'
#' Execution fields (device, kernel, threads, chunks, backend) are added by
#' `em_begin()`.
#' @noRd
build_recipe <- function(n, k, d, nepochs = 5, batch_law = "auto",
                         batch_size = NULL, alpha0 = NULL,
                         unused_threshold = NULL) {
  n <- as.double(n)
  law <- resolve_batch_law(batch_law, d)
  res <- resolve_recipe(n, k, nepochs, law$law, batch_size, alpha0)
  rows <- epoch_rows(n, nepochs)
  steps <- ceiling(rows / res$batch_size)
  batch_given <- !is.null(batch_size)
  alpha_given <- !is.null(alpha0)
  list(
    name = if (batch_given) {
      if (alpha_given) "manual" else "manual_batch"
    } else {
      "canonical_v1"
    },
    batch_law = if (batch_given) NULL else law$law,
    batch_law_requested = if (batch_given) NULL else batch_law,
    batch_law_reason = if (batch_given) NULL else law$reason,
    batch_size = res$batch_size,
    batch_size_requested = if (batch_given) as.double(batch_size) else NULL,
    alpha0 = res$alpha0,
    alpha0_source = if (alpha_given) "given" else "step_law",
    nepochs = nepochs,
    steps_per_pass = ceiling(n / res$batch_size),
    total_steps = sum(steps),
    epoch_rows = rows,
    epoch_steps = steps,
    tail_policy = "keep",
    alpha_min_factor = emeans_constants$alpha_min_factor,
    check_unused_every = emeans_constants$check_unused_every,
    unused_threshold = resolve_unused_threshold(unused_threshold, d),
    split_epsilon = emeans_constants$split_epsilon,
    init_usage = emeans_constants$init_usage,
    upstream_commit = upstream_commit
  )
}
