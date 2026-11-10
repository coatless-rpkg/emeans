# Revival of unused clusters (port of upstream e_means.py `_choose_splits`,
# Algorithm 1, line 9).

#' Choose donor and target clusters for revival
#'
#' Clusters whose EMA mass is below `tau` times the mean mass are unused.
#' Donors (used clusters with positive mass) are drawn without replacement in
#' proportion to their mass, by Gumbel top-k. Targets are the first unused
#' clusters in index order. Draws no random numbers when nothing is split.
#'
#' @param usage Double vector of EMA masses, one per cluster.
#' @param tau Revival threshold (`recipe$unused_threshold`).
#' @param draw Function returning `k` uniforms.
#' @return `NULL` when nothing is split, otherwise a list of 1-based integer
#'   `source` (donors, by decreasing key) and `target` indices.
#' @noRd
choose_splits <- function(usage, tau, draw = draw_revival_uniforms) {
  k <- length(usage)
  threshold <- tau * sum(usage) / k
  unused <- usage < threshold
  donors <- !unused & usage > 0
  nsplit <- min(sum(unused), sum(donors))
  if (nsplit == 0L) {
    return(NULL)
  }
  uniform <- pmax(draw(k), .Machine$double.xmin)
  keys <- rep(-Inf, k)
  keys[donors] <- log(usage[donors]) - log(-log(uniform[donors]))
  # Upstream takes argsort(keys)[-nsplit:][::-1]: the largest keys, largest
  # first.
  ascending <- order(keys)
  source <- rev(ascending[seq.int(k - nsplit + 1L, k)])
  target <- which(unused)[seq_len(nsplit)]
  list(source = source, target = target)
}
