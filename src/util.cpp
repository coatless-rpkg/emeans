#include <Rcpp.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::export]]
int cpp_openmp_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 0;
#endif
}

// [[Rcpp::export]]
Rcpp::NumericVector cpp_nextafter(Rcpp::NumericVector x, double toward) {
  Rcpp::NumericVector out(x.size());
  for (R_xlen_t i = 0; i < x.size(); ++i) {
    out[i] = std::nextafter(x[i], toward);
  }
  return out;
}
