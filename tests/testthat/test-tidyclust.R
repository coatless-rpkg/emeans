skip_if_not_installed("tidyclust"); skip_if_not_installed("modelenv")
test_that("k_means() fits with the emeans engine", {
  set.seed(1); df <- as.data.frame(matrix(rnorm(600), 300, 2))
  spec <- tidyclust::k_means(num_clusters = 3) |> tidyclust::set_engine("emeans", nepochs = 3)
  fit <- generics::fit(spec, ~., data = df)
  cen <- tidyclust::extract_centroids(fit); expect_identical(nrow(cen), 3L)
  asg <- tidyclust::extract_cluster_assignment(fit); expect_identical(nrow(asg), 300L)
  pr <- predict(fit, df[1:5, ]); expect_identical(levels(pr$.pred_cluster), paste0("Cluster_", 1:3))
  expect_true(is.finite(tidyclust::sse_within_total(fit)$.estimate))
})
test_that("registration is idempotent", {
  expect_silent(register_tidyclust_engine()); expect_silent(register_tidyclust_engine())
})
test_that(".tidyclust_fit_emeans gives true SSE for Euclidean and spherical fits", {
  set.seed(2); x <- matrix(rnorm(600, mean = 3), 300, 2)
  for (sph in c(FALSE, TRUE)) {
    fit <- .tidyclust_fit_emeans(x, 3, spherical = sph, nepochs = 3)
    expect_length(fit$withinss, 3L)
    expect_true(all(fit$withinss >= 0))
    # Spherical centers are unit-norm, so SSE can exceed totss on unnormalised data.
    if (!sph) expect_lte(sum(fit$withinss), fit$totss)
    expect_equal(fit$totss, sum(sweep(x, 2, colMeans(x))^2))
    direct <- sum(rowSums((x - fit$centers[predict(fit, x), ])^2))
    expect_equal(sum(fit$withinss), direct, tolerance = 1e-8)
  }
})
