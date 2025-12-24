#' CausalImages Tutorial Data
#'
#' A tutorial dataset containing satellite imagery and associated variables for
#' demonstrating causal inference with earth observation data. The data is loaded
#' via `data(CausalImagesTutorialData)` and provides the following objects.
#'
#' @format The dataset contains the following objects:
#' \describe{
#'   \item{FullImageArray}{A 4-dimensional array of satellite images with dimensions
#'     (n_images, height, width, channels). Each image is a 35x35 pixel RGB image.}
#'   \item{KeysOfImages}{A character vector of unique identifiers for each image in
#'     \code{FullImageArray}. Used to link images to observations.}
#'   \item{KeysOfObservations}{A character vector of length n_observations specifying
#'     which image key corresponds to each observation. Multiple observations may
#'     share the same image key.}
#'   \item{LongLat}{A data frame with columns \code{geo_long} and \code{geo_lat}
#'     containing the longitude and latitude coordinates for each observation.}
#'   \item{obsW}{A binary numeric vector indicating treatment assignment (0 = control,
#'     1 = treated) for each observation.}
#'   \item{obsY}{A numeric vector of observed outcomes for each observation.}
#'   \item{X}{A numeric matrix of tabular covariates for each observation. The first
#'     column is typically an intercept and may be dropped in analysis.}
#' }
#'
#' @details
#' This dataset is designed for tutorial purposes to demonstrate the key
#' functionalities of the causalimages package, including:
#' \itemize{
#'   \item Writing image data to TFRecord format via \code{\link{WriteTfRecord}}
#'   \item Extracting image representations via \code{\link{GetImageRepresentations}}
#'   \item Performing image-based confounding analysis via \code{\link{AnalyzeImageConfounding}}
#'   \item Analyzing treatment effect heterogeneity via \code{\link{AnalyzeImageHeterogeneity}}
#' }
#'
#' @examples
#' # Load the tutorial data
#' data(CausalImagesTutorialData)
#'
#' # View dimensions of the image array
#' dim(FullImageArray)
#'
#' # Check the number of observations
#' length(obsY)
#'
#' # View treatment distribution
#' table(obsW)
#'
#' @source Simulated data for package tutorials.
#' @name CausalImagesTutorialData
#' @docType data
#' @keywords datasets
NULL

#' @rdname CausalImagesTutorialData
#' @format NULL
"FullImageArray"

#' @rdname CausalImagesTutorialData
#' @format NULL
"KeysOfImages"

#' @rdname CausalImagesTutorialData
#' @format NULL
"KeysOfObservations"

#' @rdname CausalImagesTutorialData
#' @format NULL
"LongLat"

#' @rdname CausalImagesTutorialData
#' @format NULL
"obsW"

#' @rdname CausalImagesTutorialData
#' @format NULL
"obsY"

#' @rdname CausalImagesTutorialData
#' @format NULL
"X"
