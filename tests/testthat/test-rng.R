test_that("draws use R's RNG and are recorded in order", {
  rec <- new.env(); rec$draws <- list()
  withr::local_options(emeans.rng_recorder = rec)
  set.seed(1)
  a <- draw_subset(100, 10); b <- draw_init_rows(10, 3)
  p <- draw_permutation(10); u <- draw_revival_uniforms(4)
  set.seed(1)
  expect_identical(a, sort(sample.int(100, 10))); expect_identical(b, sample.int(10, 3))
  expect_identical(p, sample.int(10)); expect_identical(u, runif(4))
  expect_identical(vapply(rec$draws, `[[`, "", "type"),
                   c("subset", "init", "permutation", "revival"))
  expect_identical(rec$draws[[3]]$value, p)
})
test_that("nothing is recorded without a recorder", {
  withr::local_options(emeans.rng_recorder = NULL)
  expect_silent(draw_permutation(5))
})
test_that("a recorder option that is not an environment is ignored", {
  withr::local_options(emeans.rng_recorder = list(draws = list()))
  set.seed(2)
  expect_silent(p <- draw_permutation(5))
  set.seed(2)
  expect_identical(p, sample.int(5))
})
