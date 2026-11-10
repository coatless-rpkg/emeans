# Algorithmic constants ported from upstream faiss e-means (base.py).

upstream_commit <- "2b648f0605f2ef3d14c4705350f8a0b214463878"

emeans_constants <- list(
  # Epsilon in Equation 5: cosine schedule minimum step / initial step.
  alpha_min_factor = 0.1,
  # Logical batch updates between revival checks.
  check_unused_every = 5L,
  # Initial EMA mass N_j.
  init_usage = 2^-16,
  # Relative coordinate perturbation for donor splits.
  split_epsilon = 1e-4,
  # Revival threshold tau: revive N_j < tau * mean(N).
  unused_threshold = 0.1,
  # unused_threshold = NULL turns revival off at or below this dimension.
  no_revival_max_d = 2L,
  # batch_law = "auto" picks the quality law at or below this dimension.
  auto_quality_max_d = 16L,
  # Batch law B = C_B * k^b_k * n^b_n * T^b_T, coefficients (C_B, b_k, b_n, b_T).
  batch_laws = list(
    throughput = c(0.035134, 0.134749, 0.876473, 0.813344),
    quality = c(0.00439175, 0.134749, 0.876473, 0.813344)
  ),
  # Step law alpha0 = C * k^a_k * B^a_B * n^a_n * T^a_T.
  alpha_law = c(0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
)
