# Benchmark the "direct" and "blas" assignment kernels.
# Usage: R --vanilla -f tools/bench/kernels.R --args <label> [lib_loc]
# Writes tools/bench/results-<label>.csv (median ms per call).
# The package must be installed (with OpenMP for a meaningful threads = 2).
args <- commandArgs(trailingOnly = TRUE)
label <- args[[1]]
lib_loc <- if (length(args) >= 2) args[[2]] else NULL
library(emeans, lib.loc = lib_loc)
assign_fn <- emeans:::cpp_assign_accumulate

cat("BLAS:", sessionInfo()$BLAS, "\n")
cat("OpenMP max threads:", emeans:::cpp_openmp_max_threads(), "\n")

m <- 8192L
rows <- seq_len(m)
grid <- expand.grid(d = c(2L, 8L, 16L, 32L, 128L), k = c(100L, 1000L, 10000L))
set.seed(1)
out <- list()
for (g in seq_len(nrow(grid))) {
  d <- grid$d[g]; k <- grid$k[g]
  x <- matrix(rnorm(m * d), m, d)
  cen <- x[sample.int(m, k, replace = k > m), , drop = FALSE] + rnorm(k * d, sd = 1e-3)
  run <- function(kernel, threads) {
    assign_fn(x, rows, cen, FALSE, kernel, 4096L, 1024L, threads,
              matrix(0, k, d), numeric(k), 0)
  }
  b <- bench::mark(
    direct_1 = run("direct", 1L), blas_1 = run("blas", 1L),
    direct_2 = run("direct", 2L), blas_2 = run("blas", 2L),
    min_time = 0.5, max_iterations = 10, check = FALSE, filter_gc = FALSE
  )
  parts <- strsplit(as.character(b$expression), "_")
  out[[g]] <- data.frame(
    blas = label, d = d, k = k,
    kernel = vapply(parts, `[`, "", 1), threads = as.integer(vapply(parts, `[`, "", 2)),
    ms = round(as.numeric(b$median) * 1000, 2)
  )
  print(out[[g]], row.names = FALSE)
}
res <- do.call(rbind, out)
write.csv(res, file.path("tools", "bench", paste0("results-", label, ".csv")), row.names = FALSE)
