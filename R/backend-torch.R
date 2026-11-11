# Torch backend: float32 tensors on a {torch} device (CUDA, MPS or CPU).
#
# Port of upstream torch_backend.py (`TorchBackend`: assign, accumulate,
# update, split; `_separate_splits`) and of the readout and closing in
# e_means_torch.py (`_readout`, `_spherical_closing`, `finalize`).
#
# {torch} is only in Suggests. Every call is `torch::`-qualified, and none
# runs before resolve_device() has checked that {torch} is usable. R torch
# takes 1-based dimensions and 1-based index tensors (index_select,
# index_add_, index_copy_) and returns 1-based indices from min(dim =), so
# labels stay 1-based throughout. The backend draws no random numbers: rows,
# donors and targets arrive as R integers and become `torch_long` tensors.
#
# Precision: data, centroids, usage and sums are float32, as upstream.
# Counts and objectives are float64 where the device has it (upstream keeps
# pass counts and the objective in float64); MPS has no float64, so there
# they are float32 too, exact up to 2^24 rows per cluster.

#' Is `x` a torch tensor?
#'
#' Needs no {torch}: a tensor can only exist when {torch} is installed.
#' @noRd
is_torch_tensor <- function(x) {
  inherits(x, "torch_tensor")
}

#' Validate a torch tensor holding a data matrix
#'
#' Checks the shape and dtype only. `NA`, `NaN` and infinite values are not
#' checked: that would read the whole tensor back from the device.
#' @param x A `torch_tensor`.
#' @param name Argument name for messages.
#' @return `x` as float32 on its own device.
#' @noRd
as_training_tensor <- function(x, name) {
  if (length(x$shape) != 2L) {
    abort_emeans(sprintf("`%s` must be a 2-dimensional torch_tensor.", name),
                 "emeans_error_input")
  }
  if (x$dtype == torch::torch_bool() || x$is_complex()) {
    abort_emeans(sprintf("`%s` must be a real-valued torch_tensor.", name),
                 "emeans_error_input")
  }
  x$detach()$to(dtype = torch::torch_float32())
}

#' Gather rows of a tensor by 1-based R indices
#' @noRd
tensor_rows <- function(x, rows) {
  x$index_select(1L, torch::torch_tensor(rows, dtype = torch::torch_long(),
                                         device = x$device))
}

#' Copy a tensor to an R double matrix
#'
#' Moves to the CPU before any dtype change: MPS has no float64.
#' @noRd
tensor_to_matrix <- function(x) {
  m <- as.matrix(x$cpu())
  storage.mode(m) <- "double"
  dim(m) <- x$shape
  m
}

#' Copy a tensor to an R double vector
#' @noRd
tensor_to_vector <- function(x) {
  as.numeric(x$cpu())
}

#' Project rows to unit length, prescaling by the max absolute value
#'
#' Port of upstream `normalize_into`. Zero rows stay zero. Returns a new
#' tensor.
#' @noRd
torch_normalize_rows <- function(m) {
  top <- torch::linalg_vector_norm(m, ord = Inf, dim = 2L, keepdim = TRUE)
  m <- m / torch::torch_where(top > 0, top, 1)
  norm <- torch::linalg_vector_norm(m, dim = 2L, keepdim = TRUE)
  m / torch::torch_where(norm > 0, norm, 1)
}

#' `nextafter()` toward `direction`, on the CPU if the device lacks it
#' @noRd
torch_nextafter_toward <- function(value, direction) {
  to <- torch::torch_full_like(value, direction)
  tryCatch(
    torch::torch_nextafter(value, to),
    error = function(e) {
      torch::torch_nextafter(value$cpu(), to$cpu())$to(device = value$device)
    }
  )
}

#' Separate identical donor/child rows on coordinate one
#'
#' Port of `_separate_splits` for float32. Modifies and returns `child`.
#' @noRd
torch_separate_splits <- function(donor, child) {
  collapsed <- (donor == child)$all(dim = 2L)
  value <- child[, 1]
  neighbor <- torch_nextafter_toward(value, Inf)
  inward <- torch_nextafter_toward(value, -Inf)
  neighbor <- torch::torch_where(neighbor$abs() < Inf, neighbor, inward)
  child[, 1] <- torch::torch_where(collapsed, neighbor, value)
  child
}

#' Build the torch backend
#'
#' @param device A `torch_device`.
#' @param chunk_size,centroid_chunk_size Rows and centroids per score block.
#' @return A backend (see `R/backend.R`).
#' @noRd
torch_backend <- function(device, chunk_size, centroid_chunk_size) {
  force(device); force(chunk_size); force(centroid_chunk_size)
  f32 <- torch::torch_float32()
  long <- torch::torch_long()
  wide <- if (identical(device$type, "mps")) f32 else torch::torch_float64()

  index_tensor <- function(i) torch::torch_tensor(i, dtype = long, device = device)
  r_labels <- function(labels) {
    as.integer(labels$to(dtype = torch::torch_int())$cpu())
  }
  centroid_norm <- function(centroids, spherical) {
    if (spherical) {
      torch::torch_zeros(centroids$size(1L), dtype = f32, device = device)
    } else {
      centroids$square()$sum(dim = 2L)
    }
  }

  # Port of `TorchBackend.assign`: scores |c|^2 - 2 x.c over centroid chunks,
  # keeping a strict-less running minimum so ties go to the lowest index.
  # `score` is the squared distance, or -2 x.c in spherical mode.
  nearest <- function(x, centroids, norm, spherical) {
    m <- x$size(1L)
    k <- centroids$size(1L)
    best <- torch::torch_full(m, Inf, dtype = f32, device = device)
    label <- torch::torch_ones(m, dtype = long, device = device)
    for (begin in seq.int(1L, k, by = centroid_chunk_size)) {
      len <- min(centroid_chunk_size, k - begin + 1L)
      block <- centroids$narrow(1L, begin, len)
      score <- torch::torch_addmm(norm$narrow(1L, begin, len)$unsqueeze(1L),
                                  x, block$t(), alpha = -2)
      top <- score$min(dim = 2L)
      better <- top[[1L]] < best
      best <- torch::torch_where(better, top[[1L]], best)
      label <- torch::torch_where(better, top[[2L]] + as.integer(begin - 1L), label)
    }
    score <- if (spherical) best else (best + x$square()$sum(dim = 2L))$clamp(min = 0)
    list(score = score, label = label)
  }

  be <- list(
    name = "torch",
    prepare_data = function(x) {
      if (is_torch_tensor(x)) {
        return(x$detach()$to(device = device, dtype = f32))
      }
      if (!is.matrix(x)) {
        x <- as.matrix(x)
      }
      if (!is.double(x)) {
        storage.mode(x) <- "double"
      }
      torch::torch_tensor(x, dtype = f32, device = device)
    },
    n_rows = function(data) data$size(1L),
    n_cols = function(data) data$size(2L),
    as_native_matrix = function(m) {
      m <- as.matrix(m)
      storage.mode(m) <- "double"
      torch::torch_tensor(m, dtype = f32, device = device)
    },
    as_native_vector = function(v) {
      torch::torch_tensor(as.double(v), dtype = f32, device = device)
    },
    as_r_matrix = tensor_to_matrix,
    as_r_vector = tensor_to_vector,
    zeros_matrix = function(k, d) torch::torch_zeros(k, d, dtype = f32, device = device),
    zeros_vector = function(k) torch::torch_zeros(k, dtype = wide, device = device),
    add = function(a, b) a + b,
    accumulate = function(data, rows, readout, acc, spherical, want_labels) {
      norm <- centroid_norm(readout, spherical)
      sums <- acc$sums$clone()
      counts <- acc$counts$clone()
      objective <- acc$objective$to(dtype = wide)
      labels <- list()
      m <- length(rows)
      for (begin in seq(1, m, by = chunk_size)) {
        end <- min(begin + chunk_size - 1, m)
        x <- data$index_select(1L, index_tensor(rows[begin:end]))
        res <- nearest(x, readout, norm, spherical)
        if (spherical) {
          objective <- objective + 2 * (end - begin + 1)
        }
        objective <- objective + res$score$sum(dtype = wide)
        sums$index_add_(1L, res$label, x)
        counts$index_add_(1L, res$label,
                          torch::torch_ones(end - begin + 1, dtype = wide, device = device))
        if (want_labels) {
          labels[[length(labels) + 1L]] <- res$label
        }
      }
      list(
        acc = list(sums = sums, counts = counts, objective = objective),
        labels = if (want_labels) r_labels(torch::torch_cat(labels)) else NULL
      )
    },
    # Port of `TorchBackend.update`: the masked EMA step in float32, with
    # fixed-shape masks instead of row selection.
    update = function(centroids, usage, sums, counts, alpha, replace_usage) {
      counts <- counts$to(dtype = f32)
      touched <- counts > 0
      safe_counts <- torch::torch_where(touched, counts, 1)
      mean <- sums$masked_fill(touched$logical_not()$unsqueeze(2L), 0) /
        safe_counts$unsqueeze(2L)
      if (!replace_usage) {
        usage <- usage * (1 - alpha)
      }
      retained <- if (replace_usage) torch::torch_zeros_like(usage) else usage
      mass <- retained + alpha * counts
      weight <- alpha * counts / torch::torch_where(touched, mass, 1)
      position <- (mean - centroids) * weight$unsqueeze(2L) + centroids
      position <- torch::torch_where((weight == 1)$unsqueeze(2L), mean, position)
      list(
        centroids = torch::torch_where(touched$unsqueeze(2L), position, centroids),
        usage = torch::torch_where(touched, mass, usage)
      )
    },
    # Port of `TorchBackend.split`, with the float32 collision nudge.
    split = function(centroids, usage, source, target) {
      src <- index_tensor(source)
      tgt <- index_tensor(target)
      donors <- centroids$index_select(1L, src)
      sign <- torch::torch_tensor(rep_len(c(1, -1), donors$size(2L)),
                                  dtype = f32, device = device)
      offset <- emeans_constants$split_epsilon * sign
      donor <- donors * (1 - offset)
      child <- torch_separate_splits(donor, donors * (1 + offset))
      half <- usage$index_select(1L, src) / 2
      centroids <- centroids$clone()
      usage <- usage$clone()
      centroids$index_copy_(1L, src, donor)
      usage$index_copy_(1L, src, half)
      centroids$index_copy_(1L, tgt, child)
      usage$index_copy_(1L, tgt, half)
      list(centroids = centroids, usage = usage)
    },
    # Always a new tensor, never an alias of the stored centroids.
    readout = function(centroids, spherical) {
      if (spherical) torch_normalize_rows(centroids) else centroids$clone()
    },
    closing = function(pass_sums, pass_counts, centroids, spherical) {
      has <- (pass_counts > 0)$unsqueeze(2L)
      mean <- pass_sums$to(dtype = f32) /
        pass_counts$clamp(min = 1)$to(dtype = f32)$unsqueeze(2L)
      mean <- torch::torch_where(has, mean, centroids)
      if (spherical) torch_normalize_rows(mean) else mean
    },
    assign = function(data, readout, spherical) {
      n <- data$size(1L)
      if (n == 0L) {
        return(list(cluster = integer(0), score = numeric(0)))
      }
      norm <- centroid_norm(readout, spherical)
      scores <- labels <- list()
      for (begin in seq(1, n, by = chunk_size)) {
        len <- min(chunk_size, n - begin + 1)
        res <- nearest(data$narrow(1L, begin, len), readout, norm, spherical)
        scores[[length(scores) + 1L]] <- res$score
        labels[[length(labels) + 1L]] <- res$label
      }
      score <- torch::torch_cat(scores)
      if (spherical) {
        score <- score * -0.5
      }
      list(cluster = r_labels(torch::torch_cat(labels)),
           score = tensor_to_vector(score))
    }
  )
  check_backend(be)
  be
}
