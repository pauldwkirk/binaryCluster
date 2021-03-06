// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// binaryCluster
List binaryCluster(const arma::umat X, const int initialNumberOfComponents, const int numIterations, unsigned int nDataLevels, unsigned int nViews, unsigned int hasNull, std::string const& fileLocation);
RcppExport SEXP binaryCluster_binaryCluster(SEXP XSEXP, SEXP initialNumberOfComponentsSEXP, SEXP numIterationsSEXP, SEXP nDataLevelsSEXP, SEXP nViewsSEXP, SEXP hasNullSEXP, SEXP fileLocationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::umat >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type initialNumberOfComponents(initialNumberOfComponentsSEXP);
    Rcpp::traits::input_parameter< const int >::type numIterations(numIterationsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nDataLevels(nDataLevelsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type nViews(nViewsSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type hasNull(hasNullSEXP);
    Rcpp::traits::input_parameter< std::string const& >::type fileLocation(fileLocationSEXP);
    rcpp_result_gen = Rcpp::wrap(binaryCluster(X, initialNumberOfComponents, numIterations, nDataLevels, nViews, hasNull, fileLocation));
    return rcpp_result_gen;
END_RCPP
}
