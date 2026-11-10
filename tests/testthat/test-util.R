test_that("OpenMP probe and nextafter helpers work", {
  expect_true(is.integer(cpp_openmp_max_threads()))
  expect_gte(cpp_openmp_max_threads(), 0L)
  expect_identical(cpp_nextafter(0, Inf), 4.9406564584124654e-324)
  expect_identical(cpp_nextafter(1, -Inf), 1 - .Machine$double.eps / 2)
})
test_that("abort_emeans signals classed conditions", {
  expect_error(abort_emeans("boom", "emeans_error_input"),
               class = "emeans_error_input")
  expect_error(abort_emeans("boom", "emeans_error_input"),
               class = "emeans_error")
})
