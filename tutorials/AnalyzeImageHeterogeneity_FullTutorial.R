#!/usr/bin/env Rscript

################################
# Image confounding tutorial using causalimages
################################

# remote install latest version of the package if needed
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# local install for development team
# install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

# load in package
library( causalimages  )

# load in tutorial data
data(  CausalImagesTutorialData )

# example acquire image function (loading from memory)
# in general, you'll want to write a function that returns images
# that saved disk associated with keys
acquireImageFromMemory <- function(keys, training = F){
  # here, the function input keys
  # refers to the unit-associated image keys
  m_ <- FullImageArray[match(keys, KeysOfImages),,,]

  # if keys == 1, add the batch dimension so output dims are always consistent
  # (here in image case, dims are batch by height by width by channel)
  if(length(keys) == 1){
    m_ <- array(m_,dim = c(1L,dim(m_)[1],dim(m_)[2],dim(m_)[3]))
  }

  return( m_ )
}

# drop first column of X
X <- X[,-1]

# mean imputation for simplicity
X <- apply(X,2,function(zer){
  zer[is.na(zer)] <- mean( zer,na.rm = T )
  return( zer )
})

# perform image-based treatment effect heterogeneity decomposition
ImageHeterogeneityResults <- AnalyzeImageHeterogeneity(
  # data inputs
  obsW =  obsW,
  obsY = obsY,
  imageKeysOfUnits =  KeysOfObservations,
  acquireImageFxn = acquireImageFromMemory,
  conda_env = "tensorflow_m1", # change "tensorflow_m1" to the location of your conda environment containing tensorflow v2 and tensorflow_probability,
  conda_env_required = T,
  X = X,
  plotBands = 1L:3L,

  # inputs to control where visual results are saved as PDF or PNGs
  # (these image grids are large and difficult to display in RStudio's interactive mode)
  plotResults = T,
  figuresPath = "~/Downloads",
  printDiagnostics = T,
  figuresTag = "CausalImagesTutorial",

  # optional arguments for generating transportability maps
  # here, we leave those NULL
  transportabilityMat = NULL, #
  lat =  NULL, # required only if transportabilityMat specified
  long =  NULL, # # required only if transportabilityMat specified

  # other modeling options
  orthogonalize = F,
  modelType = "variational_minimal",
  kClust_est = 2,
  nMonte_variational = 3L, # make this larger for real application (e.g., 10)
  nSGD = 4L, # make this larger for real applications (e.g., 2000L)
  batchSize = 10L, # make this larger for real application (e.g., 50L)
  channelNormalize = T,
  compile = T,
  yDensity = "normal",
  kernelSize = 3L, maxPoolSize = 2L, strides = 1L,
  nDepthHidden_conv = 2L, # in practice, nDepthHidden_conv would be more like 4L
  nFilters = 64L,
  nDepthHidden_dense = 0L, nDenseWidth = 32L,
  nDimLowerDimConv = 3L,
  reparameterizationType = "Flipout"
)



# now, let's go do an analysis using video data

# example video function (this here just appends two identical images for illustration only)
# in practice, image sequence / video data will be read from disk
acquireVideoRepFromMemory <- function(keys, training = F){
  tmp <- FullImageArray[match(keys, KeysOfImages),,,]
  tmp <- abind::abind(tmp, tmp, along = 0)
  tmp <- aperm(tmp, c(2, 1, 3, 4, 5))
  return(  tmp  )
}

