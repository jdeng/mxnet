% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ndarray.R
\name{mx.nd.save}
\alias{mx.nd.save}
\title{Save an mx.nd.array object}
\usage{
mx.nd.save(ndarray, filename)
}
\arguments{
\item{ndarray}{the \code{mx.nd.array} object}

\item{filename}{the filename (including the path)}
}
\description{
Save an mx.nd.array object
}
\examples{
mat = mx.nd.array(1:3)
mx.nd.save(mat, 'temp.mat')
mat2 = mx.nd.load('temp.mat')
as.array(mat)
as.array(mat2)

}

