#!/usr/bin/env Rscript
#' AnalyzeImageMediation
#'
#' *Under construction.*
#'
#' @usage
#'
#' AnalyzeImageMediation()
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param obsY A numeric vector containing observed outcomes.
#' @param acquireImageRepFxn A function specifying how to load images representations associated with `imageKeysOfUnits` into memory. For example, if observation `3` has a value  of `"a34f"` in `imageKeysOfUnits`, `acquireImageFxn` should extract the image associated with the unique key `"a34f"`.
#' First argument should be image key values and second argument have be `training` (in case behavior in training/)
#' @param acquireImageFxn (default = `acquireImageRepFxn`) Similar to `acquireImageRepFxn`; this is a function specifying how to load images associated with `imageKeysOfUnits` into memory.
#' @param transportabilityMat (optional) A matrix with a column named `keys` specifying keys to be used by `acquireImageRepFxn` for generating treatment effect predictions for out-of-sample points.
#' @param imageKeysOfUnits (default = `1:length(obsY)`) A vector of length `length(obsY)` specifying the unique image ID associated with each unit. Samples of `imageKeysOfUnits` are fed into `acquireImageFxn` to call images into memory.
#' @param long,lat (optional) Vectors specifying longitude and latitude coordinates for units. Used only for describing highest and lowest probability neighorhood units if specified.
#' @param X (optional) A numeric matrix containing tabular information used if `orthogonalize = T`.
#' @param conda_env (default = `NULL`) A string specifying a conda environment wherein `tensorflow`, `tensorflow_probability`, and `gc` are installed.
#' @param orthogonalize (default = `F`) A Boolean specifying whether to perform the image decomposition after orthogonalizing with respect to tabular covariates specified in `X`.
#' @param nMonte_variational (default = `5L`) An integer specifying how many Monte Carlo iterations to use in the
#' calculation of the expected likelihood in each training step.
#' @param nMonte_predictive (default = `20L`) An integer specifying how many Monte Carlo iterations to use in the calculation
#' of posterior means (e.g., mean cluster probabilities).
#' @param nMonte_salience (default = `100L`) An integer specifying how many Monte Carlo iterations to use in the calculation
#' of the salience maps (e.g., image gradients of expected cluster probabilities).
#' @param reparameterizationType (default = `"Flipout"`) Either `"Flipout"`, or `"Reparameterization"`. Specifies the estimator used in the Bayesian neural components. With `"Flipout"`, convolutions are performed via CPU; with `"Reparameterization", they are performed by GPU if available.
#' @param figuresKey (default = `""`) A string specifying an identifier that is appended to all figure names.
#' @param figuresPath (default = `"./"`) A string specifying file path for saved figures made in the analysis.
#' @param kernelSize (default = `5L`) Dimensions used in convolution kernels.
#' @param nSGD (default = `400L`) Number of stochastic gradient descent (SGD) iterations.
#' @param batchSize (default = `25L`) Batch size used in SGD optimization.
#' @param doConvLowerDimProj (default = `T`) Should we project the `nFilters` convolutional feature dimensions down to `nDimLowerDimConv` to reduce the number of required parameters.
#' @param nDimLowerDimConv (default = `3L`) If `doConvLowerDimProj = T`, then, in each convolutional layer, we project the `nFilters` feature dimensions down to `nDimLowerDimConv` to reduce the number of parameters needed.
#' @param nFilters (default = `32L`) Integer specifying the number of convolutional filters used.
#' @param nDenseWidth (default = `32L`) Width of dense projection layers post-convolutions.
#' @param nDepthHidden_conv (default = `3L`) Hidden depth of convolutional layer.
#' @param nDepthHidden_dense (default = `0L`) Hidden depth of dense layers. Default of `0L` means a single projection layer is performed after the convolutional layer (i.e., no hidden layers are used).
#' @param quiet (default = `F`) Should we suppress information about progress?
#' @param yDensity (default = `normal`) Specifies the density for the outcome. Current options include `normal` and `lognormal`.
#' @param maxPoolSize (default = `2L`) Integer specifying the max pooling size used in the convolutional layers.
#' @param strides (default = `2L`) Integer specifying the strides used in the convolutional layers.=
#' @param simMode (default = `F`) Should the analysis be performed in comparison with ground truth from simulation?
#' @param plotResults (default = `T`) Should analysis results be plotted?
#' @param channelNormalize (default = `T`) Should channelwise image feature normalization be attempted? Default is `T`, as this improves training.
#'
#' @return A list consiting of \itemize{
#'   \item `ATE_est` ATE estimate.
#'   \item `ATE_se` Standard error estimate for the ATE.
#' }
#'
#' @section References:
#' \itemize{
#' \item  TBA.
#' }
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/causalimages-software/
#'
#' @export
#' @md

AnalyzeImageMediation <- function(
                                   obsW,
                                   obsY,
                                   X = NULL,
                                   file,
                                   nDepth = 3L,
                                   doConvLowerDimProj = T,
                                   nDimLowerDimConv = 3L,
                                   nFilters = 32L,

                                   normalizationType = "none",
                                   doHiddenDim = T,
                                   HiddenDim  = 32L,
                                   DenseActivation = "linear",

                                   orthogonalize = F,
                                   imageKeysOfUnits = 1:length(obsY),
                                   acquireImageRepFxn = NULL ,
                                   acquireImageFxn = NULL ,
                                   transportabilityMat = NULL ,
                                   lat = NULL,
                                   long = NULL,
                                   conda_env = NULL,

                                   figuresKey = "",
                                   figuresPath = "./",
                                   modelType = "variational_minimal",
                                   simMode = F,
                                   plotResults = F,

                                   nDepthHidden_conv = 1L,
                                   nDepthHidden_dense = 0L,
                                   maxPoolSize = 2L,
                                   strides = 1L,
                                   yDensity = "normal",
                                   compile = T,
                                   nMonte_variational = 5L,
                                   nMonte_predictive = 20L,
                                   nMonte_salience = 100L,
                                   batchSize = 25L,
                                   kernelSize = 5L,
                                   nSGD  = 400L,
                                   nDenseWidth = 32L,
                                   reparameterizationType = "Reparameterization",
                                   channelNormalize = T,
                                   printDiagnostics = F,
                                   quiet = F){
# Under construction
}
