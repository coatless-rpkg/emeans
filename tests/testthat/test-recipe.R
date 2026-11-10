test_that("resolve_recipe matches upstream reference values", {
  cases <- list(
    list(1e6, 16384, 25, "throughput", 262144, 1.0),
    list(1e6, 16384, 25, "quality", 32768, 0.1545388281304627),
    list(1e8, 262144, 25, "throughput", 33554432, 1.0),
    list(1e8, 262144, 25, "quality", 4194304, 0.42718555053293517))
  for (cs in cases) {
    r <- resolve_recipe(cs[[1]], cs[[2]], cs[[3]], cs[[4]])
    expect_equal(r$batch_size, cs[[5]])
    expect_equal(r$alpha0, cs[[6]], tolerance = 1e-12)
  }
})
test_that("a batch_size alone takes the step law", {
  r <- resolve_recipe(1e6, 16384, 3, "throughput", batch_size = 8192)
  expect_equal(r$batch_size, 8192)
  expect_equal(r$alpha0, step_law(1e6, 16384, 3, 8192))
})
test_that("epoch_rows matches upstream rounding", {
  expect_equal(epoch_rows(1e10, 1.9999999995), c(9999999995, 1e10))
  expect_equal(epoch_rows(3, 1.5), c(2, 3))
  expect_equal(epoch_rows(5, 1.5), c(2, 5))
  expect_equal(epoch_rows(3, 1.01), 3)
  expect_equal(epoch_rows(32, 5), rep(32, 5))
})
test_that("auto batch law and revival threshold follow d", {
  expect_equal(resolve_batch_law("auto", 2), list(law = "quality", reason = "d <= 16"))
  expect_equal(resolve_batch_law("auto", 16), list(law = "quality", reason = "d <= 16"))
  expect_equal(resolve_batch_law("auto", 17), list(law = "throughput", reason = "d > 16"))
  expect_equal(resolve_batch_law("quality", 128), list(law = "quality", reason = "requested"))
  expect_equal(resolve_unused_threshold(NULL, 2), 0)
  expect_equal(resolve_unused_threshold(NULL, 3), 0.1)
  expect_equal(resolve_unused_threshold(0.05, 2), 0.05)
  expect_error(resolve_unused_threshold(-0.1, 8), class = "emeans_error_input")
})
test_that("scheduled_alpha follows the cosine to one tenth", {
  expect_equal(scheduled_alpha(0.4, 0, 10), 0.4)
  expect_equal(scheduled_alpha(0.4, 10, 10), 0.04)
  expect_equal(scheduled_alpha(0.4, 9, 10), 0.04880982706687237, tolerance = 1e-14)
})
test_that("round_power_of_two rounds to the nearest power", {
  expect_equal(round_power_of_two(0.3), 1)
  expect_equal(round_power_of_two(3), 4)
  expect_equal(round_power_of_two(5), 4)
  expect_equal(round_power_of_two(6), 8)
  expect_equal(round_power_of_two(1024), 1024)
})
test_that("build_recipe records resolved defaults and derived metadata", {
  r <- build_recipe(2048, 4, 2, nepochs = 2)
  expect_equal(r$batch_law, "quality"); expect_equal(r$batch_law_reason, "d <= 16")
  expect_equal(r$unused_threshold, 0)
  expect_equal(r$batch_size, resolve_recipe(2048, 4, 2, "quality")$batch_size)
  r <- build_recipe(2048, 4, 32, nepochs = 2, batch_law = "throughput")
  expect_equal(r$batch_law_reason, "requested"); expect_equal(r$unused_threshold, 0.1)
  r <- build_recipe(31, 3, 4, nepochs = 3)
  expect_equal(r$name, "canonical_v1")
  expect_equal(r$steps_per_pass, ceiling(31 / r$batch_size))
  expect_equal(r$total_steps, 3 * r$steps_per_pass)
  expect_equal(r$tail_policy, "keep")
  r <- build_recipe(32, 4, 2, nepochs = 1, batch_size = 8)
  expect_equal(r$name, "manual_batch"); expect_null(r$batch_law)
  expect_equal(r$alpha0_source, "step_law"); expect_equal(r$alpha0, step_law(32, 4, 1, 8))
  r <- build_recipe(32, 4, 2, nepochs = 1, batch_size = 8, alpha0 = 0.5)
  expect_equal(r$name, "manual"); expect_equal(r$alpha0_source, "given")
  expect_equal(build_recipe(32, 4, 2)$nepochs, 5)
})
test_that("an epoch budget that rounds to zero rows is an error", {
  expect_error(build_recipe(3, 1, 2, nepochs = 0.1), "nepochs", class = "emeans_error_input")
})
