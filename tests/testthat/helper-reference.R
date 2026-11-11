# Brute-force reference for the C++ assignment kernels, in plain R.
ref_assign <- function(x, rows, centers, spherical = FALSE) {
  xs <- x[rows, , drop = FALSE]
  k <- nrow(centers)
  d <- ncol(x)
  if (spherical) {
    sim <- xs %*% t(centers)
    labels <- max.col(sim, ties.method = "first")
    score <- sim[cbind(seq_along(labels), labels)]
    objective <- sum(2 - 2 * score)
  } else {
    d2 <- outer(rowSums(xs^2), rowSums(centers^2), "+") - 2 * xs %*% t(centers)
    labels <- max.col(-d2, ties.method = "first")
    score <- pmax(d2[cbind(seq_along(labels), labels)], 0)
    objective <- sum(score)
  }
  sums <- matrix(0, k, d)
  counts <- numeric(k)
  for (i in seq_along(labels)) {
    sums[labels[i], ] <- sums[labels[i], ] + xs[i, ]
    counts[labels[i]] <- counts[labels[i]] + 1
  }
  list(labels = as.integer(labels), score = score, sums = sums,
       counts = counts, objective = objective)
}
