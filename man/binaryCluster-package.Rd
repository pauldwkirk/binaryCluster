\name{binaryCluster-package}
\alias{binaryCluster-package}
\alias{binaryCluster}
\docType{package}
\title{
\packageTitle{binaryCluster}
}
\description{
\packageDescription{binaryCluster}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{binaryCluster}
\packageIndices{binaryCluster}
Performs clustering of binary data using Dirichlet process mixture models, with variable selection.
}
\author{
\packageAuthor{binaryCluster}

Maintainer: \packageMaintainer{binaryCluster}
}
\references{
}
\keyword{ package }
\seealso{
}
\examples{
data(sampleData)
output <- binaryClust(sampleData, numberOfIterations = 10, seed = 3, featureSelection = FALSE)
}
