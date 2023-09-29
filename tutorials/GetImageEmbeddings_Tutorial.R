#!/usr/bin/env Rscript

################################
# Image and image-sequence embeddings tutorial using causalimages
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

# example video function (this here just appends two images for illustration only)
# in practice, image sequence / video data will be read from disk
acquireVideoRepFromMemory <- function(keys, training = F){
  tmp <- acquireImageFromMemory(keys, training = training)

  if(length(keys) == 1){
    tmp <- array(tmp,dim = c(1L,dim(tmp)[1],dim(tmp)[2],dim(tmp)[3]))
  }

  tmp <- array(tmp,dim = c(dim(tmp)[1],
                           2,
                           dim(tmp)[3],
                           dim(tmp)[4],
                           1L))
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
MyImageEmbeddings <- GetImageEmbeddings(
  imageKeysOfUnits = KeysOfObservations[ take_indices ],
  acquireImageFxn = acquireImageFromMemory,
  nEmbedDim = 100,
  kernelSize = 3L,
  conda_env = "tensorflow_m1",
  conda_env_required = T
)

# each row corresponds to an observation
# each column represents an embedding dimension associated with the imagery for that location
MyImageEmbeddings$embeddings

# embeddings_fxn is the embedding function written in tf (used for other package functions)
#MyImageEmbeddings$embeddings_fxn

# obtain video embeddings
# each column represents an embedding dimension associated with the image sequence for that location
MyVideoEmbeddings <- GetImageEmbeddings(
  imageKeysOfUnits = KeysOfObservations[ take_indices ],
  acquireImageFxn = acquireVideoRepFromMemory,
  temporalKernelSize = 2L,
  kernelSize = 3L,
  nEmbedDim = 100,
  conda_env = "tensorflow_m1",
  conda_env_required = T
)
MyVideoEmbeddings$embeddings
