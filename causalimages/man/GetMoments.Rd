% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CausalImage_GetMoments.R
\name{GetMoments}
\alias{GetMoments}
\title{Get moments for normalization}
\usage{
GetMoments(iterator, dataType, image_dtype, momentCalIters = 34L)
}
\arguments{
\item{iterator}{An iterator}

\item{dataType}{A string denoting data type}

\item{momentCalIters}{Number of minibatches with which to estimate moments}
}
\value{
Returns  mean/sd normalization arrays.
}
\description{
A function obtaining moments for normalization
}
\examples{
# (Not run)

}
