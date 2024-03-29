% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/CausalImage_Simulation.R
\name{SimulateImageSystem}
\alias{SimulateImageSystem}
\title{Simulate causal systems involving images}
\usage{
SimulateImageSystem(...)
}
\arguments{
\item{dag}{\emph{(character string)} An input DAG specifying causal structure.
This input should be of the form \url{`i->t,i->y,t->y,....'}
Currently, only one node in a DAG can be an image (this should be labeled ``\url{i}'').
The non-image nodes can have arbitrary string labels.
The image can be a confounder, effect moderator, effect mediator.
If the image is to be used as a moderator, use the notation, \url{t-i>y}.}

\item{...}{\emph{(optional)} In estimation mode, users input the data matrices associated with the
non-image nodes of \url{DAG} and image node \url{i}. For example, if \url{x} is a DAG node,
users must, in estimation mode, supply data to \url{x} in a form that can be coerced to a tensor.}

\item{treatment}{\emph{(character string, optional)} In estimation mode, users specify the treatment variable here.
If \url{treatment} is specified, users must provide other data inputs to the DAG (see \url{...}).}

\item{image_pool}{\emph{(character string, optional)} The path to where analysis specific
images are located. This can be specified both in simulation and estimation mode.
If not specified, the simulation uses a pool of Landsat images from Nigeria.}

\item{analysis_level}{\emph{(character string, default is \url{`scene'})} Defines the unit
of analysis used in the simulation framework. This is ignored in estimation mode,
where the unit of analysis is inferred from the data dimensions.}

\item{control}{\emph{(list)} A list containing control parameters in the data generating process.}
}
\value{
A list:
\itemize{
\item In \emph{simulation mode}, the function returns a list with as many elements as
unique nodes in \verb{DAG}. Each element represents the simulated data.
\item In \emph{estimation mode},the function returns an estimated treatment effect with 95\\% confidence intervals.
}
}
\description{
This function generates simulated causal structures using images. It is currently under construction.
}
\section{References}{

\itemize{
\item Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Image-based Treatment Effect Heterogeneity. Forthcoming in \emph{Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)}, 2023.
}
}

\examples{
#set seed
set.seed(1)

# Simulation mode
#simulatedData <- causalimage('r->i, i->t, t->y, r->y')
#print(names(simulatedData))

# Estimation mode
#estimatedResults <- causalimage('r->i, i->t, t->y, r->y', y=y, r=r, y=y', treatment='t')
#print( estimatedResults )

}
