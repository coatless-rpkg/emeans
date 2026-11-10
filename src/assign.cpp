// Nearest-centroid assignment plus per-cluster accumulation.
//
// Two strategies share one contract. Labels come from a parallel argmin over
// a score (x.c - 0.5|c|^2, or x.c when spherical) with strict `>` so the
// lowest centroid index wins ties. Sums, counts and the objective are then
// accumulated serially in the order of `rows`, so results do not depend on
// the number of threads or on chunk sizes.
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Fixed 4-accumulator dot product; the order depends only on d.
inline double dot4(const double* a, const double* b, arma::uword d) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  arma::uword t = 0;
  for (; t + 4 <= d; t += 4) {
    s0 += a[t] * b[t];
    s1 += a[t + 1] * b[t + 1];
    s2 += a[t + 2] * b[t + 2];
    s3 += a[t + 3] * b[t + 3];
  }
  for (; t < d; ++t) s0 += a[t] * b[t];
  return (s0 + s1) + (s2 + s3);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_assign_accumulate(const arma::mat& x, const Rcpp::IntegerVector& rows,
                                 const arma::mat& centroids, bool spherical,
                                 std::string kernel, int chunk_size,
                                 int centroid_chunk_size, int threads, SEXP acc_sums,
                                 SEXP acc_counts, double acc_objective) {
  const arma::uword n = x.n_rows, d = x.n_cols, k = centroids.n_rows;
  const arma::uword m = static_cast<arma::uword>(rows.size());
  if (centroids.n_cols != d) Rcpp::stop("`centroids` must have the same number of columns as `x`.");
  if (k == 0) Rcpp::stop("`centroids` must have at least one row.");
  if (kernel != "direct" && kernel != "blas") Rcpp::stop("`kernel` must be \"direct\" or \"blas\".");
  if (chunk_size < 1 || centroid_chunk_size < 1) Rcpp::stop("Chunk sizes must be positive.");
  if (threads < 1) threads = 1;
  const bool predict = Rf_isNull(acc_sums);
  if (predict != Rf_isNull(acc_counts)) Rcpp::stop("`acc_sums` and `acc_counts` must both be NULL or both set.");

  std::vector<arma::uword> idx(m);
  for (arma::uword i = 0; i < m; ++i) {
    const int r = rows[i];
    if (r == NA_INTEGER || r < 1 || static_cast<arma::uword>(r) > n)
      Rcpp::stop("`rows` contains an index outside `x`.");
    idx[i] = static_cast<arma::uword>(r - 1);
  }

  // Half squared norms of centroids (zero in spherical mode).
  arma::vec half(k, arma::fill::zeros);
  if (!spherical) {
    for (arma::uword j = 0; j < k; ++j) {
      double s = 0;
      for (arma::uword t = 0; t < d; ++t) s += centroids(j, t) * centroids(j, t);
      half[j] = 0.5 * s;
    }
  }
  const double* h = half.memptr();
  // Armadillo uses 32-bit uword unless ARMA_64BIT_WORD is set, so x(i, t)
  // can overflow its offset arithmetic for n * d >= 2^32. Index x through a
  // raw pointer with size_t offsets in every loop that touches it.
  const double* xp = x.memptr();
  const std::size_t nx = static_cast<std::size_t>(n);

  std::vector<int> lab(m, 0);
  std::vector<double> best(m, -std::numeric_limits<double>::infinity());

  if (kernel == "direct") {
    const arma::mat ct = centroids.t();  // d x k, each centroid contiguous
#pragma omp parallel num_threads(threads)
    {
      std::vector<double> xb(d);
#pragma omp for schedule(static)
      for (long long ii = 0; ii < static_cast<long long>(m); ++ii) {
        const arma::uword i = static_cast<arma::uword>(ii);
        for (arma::uword t = 0; t < d; ++t) xb[t] = xp[static_cast<std::size_t>(idx[i]) + static_cast<std::size_t>(t) * nx];
        double bv = -std::numeric_limits<double>::infinity();
        int bj = 0;
        for (arma::uword j = 0; j < k; ++j) {
          const double v = dot4(xb.data(), ct.colptr(j), d) - h[j];
          if (v > bv) { bv = v; bj = static_cast<int>(j); }
        }
        lab[i] = bj;
        best[i] = bv;
      }
    }
  } else {
    const arma::uword cs = static_cast<arma::uword>(chunk_size);
    const arma::uword ccs = static_cast<arma::uword>(centroid_chunk_size);
    for (arma::uword a = 0; a < m; a += cs) {
      const arma::uword b = std::min(m, a + cs), mc = b - a;
      arma::mat xt(d, mc);  // gathered chunk, transposed
      for (arma::uword i = 0; i < mc; ++i)
        for (arma::uword t = 0; t < d; ++t) xt(t, i) = xp[static_cast<std::size_t>(idx[a + i]) + static_cast<std::size_t>(t) * nx];
      for (arma::uword c0 = 0; c0 < k; c0 += ccs) {
        const arma::uword c1 = std::min(k, c0 + ccs);
        const arma::mat cb = centroids.rows(c0, c1 - 1);
        const arma::mat g = cb * xt;  // (c1 - c0) x mc
#pragma omp parallel for num_threads(threads) schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(mc); ++ii) {
          const arma::uword i = static_cast<arma::uword>(ii);
          const double* gi = g.colptr(i);
          double bv = best[a + i];
          int bj = lab[a + i];
          for (arma::uword j = c0; j < c1; ++j) {
            const double v = gi[j - c0] - h[j];
            if (v > bv) { bv = v; bj = static_cast<int>(j); }
          }
          lab[a + i] = bj;
          best[a + i] = bv;
        }
      }
    }
  }

  // Serial, row-ordered accumulation.
  arma::mat sums;
  arma::vec counts;
  if (!predict) {
    sums = Rcpp::as<arma::mat>(acc_sums);
    counts = Rcpp::as<arma::vec>(acc_counts);
    if (sums.n_rows != k || sums.n_cols != d || counts.n_elem != k)
      Rcpp::stop("Accumulators have the wrong dimensions.");
  }
  Rcpp::IntegerVector labels(m);
  Rcpp::NumericVector score(m);
  double objective = acc_objective;
  for (arma::uword i = 0; i < m; ++i) {
    const arma::uword r = idx[i];
    const int j = lab[i];
    labels[i] = j + 1;
    double s;
    if (spherical) {
      s = best[i];
      objective += 2.0 - 2.0 * s;
    } else {
      double xx = 0;
      for (arma::uword t = 0; t < d; ++t) {
        const double v = xp[static_cast<std::size_t>(r) + static_cast<std::size_t>(t) * nx];
        xx += v * v;
      }
      s = xx - 2.0 * best[i];
      if (s < 0) s = 0;
      objective += s;
    }
    score[i] = s;
    if (!predict) {
      counts[j] += 1.0;
      for (arma::uword t = 0; t < d; ++t)
        sums(j, t) += xp[static_cast<std::size_t>(r) + static_cast<std::size_t>(t) * nx];
    }
  }

  Rcpp::List out = Rcpp::List::create(
      Rcpp::_["labels"] = labels, Rcpp::_["score"] = score,
      Rcpp::_["sums"] = R_NilValue, Rcpp::_["counts"] = R_NilValue,
      Rcpp::_["objective"] = objective);
  if (!predict) {
    out["sums"] = sums;
    out["counts"] = Rcpp::NumericVector(counts.begin(), counts.end());
  }
  return out;
}
