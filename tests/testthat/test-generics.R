skip_if_not_installed("generics")
test_that("tidy, glance and augment follow broom's kmeans shapes", {
  set.seed(1); x <- matrix(rnorm(400), 200, 2, dimnames = list(NULL, c("a", "b"))); fit <- emeans(x, 3)
  td <- generics::tidy(fit); expect_identical(nrow(td), 3L)
  expect_named(td, c("a", "b", "size", "withinss", "cluster"))
  expect_equal(td$withinss, fit$withinss)
  gl <- generics::glance(fit); expect_identical(nrow(gl), 1L)
  expect_named(gl, c("totss", "tot.withinss", "betweenss", "obj", "iter",
                     "epochs", "n_splits"))
  expect_equal(gl$tot.withinss, fit$tot.withinss)
  bare <- emeans(x, 3, keep_cluster = FALSE)
  expect_true(is.na(generics::glance(bare)$tot.withinss))
  expect_named(generics::tidy(bare), c("a", "b", "size", "cluster"))
  au <- generics::augment(fit, data = as.data.frame(x))
  expect_identical(as.integer(au$.cluster), fit$cluster)
  expect_identical(as.integer(generics::augment(fit, newdata = as.data.frame(x[1:5, ]))$.cluster),
                   predict(fit, x[1:5, ]))
})

test_that("augment validates its inputs", {
  set.seed(1); x <- matrix(rnorm(400), 200, 2)
  fit <- emeans(x, 3)
  expect_error(generics::augment(fit), class = "emeans_error_input")
  expect_error(generics::augment(fit, data = as.data.frame(x[1:5, ])),
               class = "emeans_error_input")
  fit2 <- emeans(x, 3, keep_cluster = FALSE)
  expect_error(generics::augment(fit2, data = as.data.frame(x)),
               class = "emeans_error_input")
  expect_named(generics::tidy(fit), c("V1", "V2", "size", "withinss", "cluster"))
})
