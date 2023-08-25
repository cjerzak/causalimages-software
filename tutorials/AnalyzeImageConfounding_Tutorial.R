#!/usr/bin/env Rscript

################################
# Image confounding tutorial using causalimages
################################

# install latest version of the package if needed
devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# load in package
library( causalimages  )

# load in tutorial data
data(  CausalImagesTutorialData )

# example acquire image function (loading from memory)
# in general, you'll want to write a function that returns images
# that saved disk associated with keys
acquireImageRepFromMemory <- function(keys, training = F){
  # here, the function input keys
  # refers to the unit-associated image keys
  return( FullImageArray[match(keys, KeysOfImages),,,] )
}

# exampole video function (this here just appends two identical images for illustration only)
# in practice, image sequence / video data will be read from disk
acquireVideoRepFromMemory <- function(keys, training = F){
  tmp <- FullImageArray[match(keys, KeysOfImages),,,]
  tmp <- abind::abind(tmp, tmp, along = 0)
  tmp <- aperm(tmp, c(2, 1, 3, 4, 5))
  return(  tmp  )
}

# drop first column
X <- X[,-1]

# mean imputation for simplicity
X <- apply(X,2,function(zer){
  zer[is.na(zer)] <- mean( zer,na.rm = T )
  return( zer )
})

# select observation subset to make tutorial analyses run faster
take_indices <- unlist( tapply(1:length(obsW),obsW,function(zer){sample(zer, 50)}) )

# uncomment for a larger n analysis
#take_indices <- 1:length( obsY )

# obtain image embeddings following Rolf et al. https://www.nature.com/articles/s41467-021-24638-z
MyImageEmbeddings <- GetRandomizedImageEmbeddings(
  imageKeysOfUnits = KeysOfObservations[ take_indices ],
  acquireImageFxn = acquireImageRepFromMemory,
  nFeatures = 100,
  kernelSize = 3L,
  conda_env = "tensorflow_m1",
  conda_env_required = T
)

# each row corresponds to an observation
# each column represents an embedding dimension associated with the imagery for that location
MyImageEmbeddings$embeddings

# embeddings_fxn is the randomized embedding function written in tf (used for other package functions)
#MyImageEmbeddings$embeddings_fxn

# obtain video embeddings generalizing Rolf et al.
# each column represents an embedding dimension associated with the image sequence for that location
MyVideoEmbeddings <- GetRandomizedImageEmbeddings(
  imageKeysOfUnits = KeysOfObservations[ take_indices ],
  acquireImageFxn = acquireVideoRepFromMemory,
  temporalKernelSize = 2L,
  kernelSize = 3L,
  nFeatures = 100,
  conda_env = "tensorflow_m1",
  conda_env_required = T
)
MyVideoEmbeddings$embeddings

# perform causal inference with image and tabular confounding
ImageConfoundingAnalysis <- AnalyzeImageConfounding(
  obsW = obsW[ take_indices ],
  obsY = obsY[ take_indices ],
  X = X[ take_indices,apply(X[ take_indices,],2,sd)>0],
  long = LongLat$geo_long[ take_indices ],
  lat = LongLat$geo_lat[ take_indices ],
  imageKeysOfUnits = KeysOfObservations[ take_indices ],
  batchSize = 4, samplingType = "balancedTrain",

  acquireImageFxn = acquireImageRepFromMemory,
  #modelClass = "cnn", # uses convolutional network (richer model class)
  modelClass = "randomizedEmbeds", # uses randomized image embeddings (faster)
  file = NULL,
  plotBands = c(1,2,3),
  dropoutRate = 0.1,
  tagInFigures = T, figuresTag = "TutorialExample",
  nBoot = 10,
  nSGD = 10, # this should be more like 1000 in full analysis
  figuresPath = "~/Downloads/", # figures saved here
  conda_env = "tensorflow_m1", # conda env to activate where a version of tensorflow lives
  conda_env_required = T
)

# ATE estimate (image  confounder adjusted)
ImageConfoundingAnalysis$tauHat_propensityHajek

# ATE se estimate (image confounder adjusted)
ImageConfoundingAnalysis$tauHat_propensityHajek_se