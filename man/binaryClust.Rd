\name{binaryClust}
\alias{binaryClust}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to perform Dirchlet process mixture modelling of binary data.
}
\description{
Performs Dirchlet process mixture modelling of binary data, with optional variable selection.
}
\usage{
binaryClust(X, numberOfIterations = 100, pathToFile = getwd(), featureSelection = FALSE, seed = NULL, initialNumberOfComponents = ceiling(nrow(X)/20))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{Data matrix to cluster. Rows are items to cluster, columns are covariates.}
  \item{numberOfIterations}{Number of Gibbs sampling iterations. Defaults to 100.}
  \item{pathToFile}{Where to write the MCMC output. Creates the specified directory if it does not already exist. Defaults to current working directory.}
  \item{featureSelection}{Whether or not to perform variable selection. Defaults to FALSE.}
  \item{seed}{Seed for random number generation, for reproducibility. Defaults to NULL (no seed set).}
  \item{initialNumberOfComponents}{Initial number of mixture components.  Defaults to ceiling(nrow(X)/20).}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{A list with a single element, successfulCompletion, which takes the value 1 if everything completed successfully.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{Paul DW Kirk}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
data(sampleData)
output <- binaryClust(sampleData, numberOfIterations = 10, seed = 3, featureSelection = FALSE)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
% \keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
