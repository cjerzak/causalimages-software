#!/usr/bin/env Rscript

################################
# Full image heterogeneity tutorial using causalimages
# **In process**
# Note: In future builds of causalimages, we will
# add easier ways to download and store satellite imagery.
# Stay tuned! Contributions & bug fixes are welcome!
################################

# remote install latest version of the package if needed
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# local install for development team
# install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

# load in package
library( causalimages  )

# specify uganda data URL
uganda_data_url <- "https://dl.dropboxusercontent.com/s/xy8xvva4i46di9d/Public%20Replication%20Data%2C%20YOP%20Experiment.zip?dl=0"

# download into new directory
download.file( uganda_data_url,  destfile = (download_folder <- "~/Downloads/UgandaAnalysis.zip" ))

# unzip and list files
unzip(download_folder, exdir = "~/Downloads/UgandaAnalysis")

# set new wd
setwd(sprintf('%s/Public Replication Data, YOP Experiment/',
              gsub(download_folder,pattern="\\.zip",replace="")))

# see directory contents
list.files()

# images saved here
list.files(  "./Uganda2000_processed"  )

# individual-level data
UgandaDataProcessed <- read.csv(  "./UgandaDataProcessed.csv"  )

# approximate longitude + latitude for units
UgandaDataProcessed$geo_long
UgandaDataProcessed$geo_lat

# image keys of units (use for referencing satellite images)
UgandaDataProcessed$geo_long_lat_key

# an experimental outcome
UgandaDataProcessed$Yobs

# treatment variable
UgandaDataProcessed$Wobs

# information on keys linking to satellite images for all of Uganda
# (not just experimental context, use for constructing transportability maps)
UgandaGeoKeyMat <- read.csv(  "./UgandaGeoKeyMat.csv"  )

# load in tensorflow (edit for your computer)
library(tensorflow); library(keras)
tensorflow::use_condaenv("tensorflow_m1", required = T)
try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]],T),T)
try( tf$config$set_soft_device_placement( T ) , T)

# write a function that reads in images as saved and process them into an array
NBANDS <- 3L
imageHeight <- imageWidth <- 351L #  pixel height/width
acquireImageFromDisk <- function(keys, training = F){
  # initialize an array shell to hold image slices
  array_shell <- array(NA,dim = c(1L,imageHeight,imageWidth,NBANDS))

  # iterate over keys:
  # -- images are referenced to keys
  # -- keys are referenced to units (to allow for duplicate images uses)
  array_ <- sapply(keys,function(key_){
    # iterate over all image bands (NBANDS = 3 for RBG images)
    for(band_ in 1:NBANDS){
      # place the image in the correct place in the array
      array_shell[,,,band_] <-
        (as.matrix(data.table::fread( # note the use of data.table::fread to speed up reading in image to memory
          input = sprintf("./Uganda2000_processed/GeoKey%s_BAND%s.csv",
                          key_,
                          band_),header = F)[-1,] ))
    }
    return( array_shell )
  },
  simplify="array")  #using simplify = "array" combines images slices together

  # convert images to tensorflow array for further processing
  # note: your acquireImageFxn need not return tensorflow arrays.
  # R arrays are fine (with dimensions c(nBatch, imageWidth, imageHeight,nChannels)
  # (R arrays will be detected converted and converted internally)
  array_ <- tf$squeeze(tf$constant(array_,dtype=tf$float32),0L)
  array_ <- tf$transpose(array_,c(3L,0L,1L,2L))
  return( array_ )
}


# try out the function
# note: some units are co-located in same area (hence, multiple observations per image key)
check_indices <- c(1,20,50)
ImageBatch <- acquireImageFromDisk(
    keys = UgandaDataProcessed$geo_long_lat_key[check_indices],
    training = F
)

# sanity checks in the analysis of earth observation data are essential
# check that images are centered around correct location
causalimages::image2(  as.array(ImageBatch)[1,,,1] )
UgandaDataProcessed$geo_long[check_indices[1]]
UgandaDataProcessed$geo_lat[check_indices[1]]
# check against google maps
# https://www.google.com/maps/place/1%C2%B018'16.4%22N+34%C2%B005'15.1%22E/@1.3111951,34.0518834,10145m/data=!3m1!1e3!4m4!3m3!8m2!3d1.3045556!4d34.0875278?entry=ttu
# checks out okay, we're good to move on




# !!!!! IN PROCESS AFTER THIS POINT !!!!!

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
  plotBands = 1L,

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
  nMonte_variational = 2L, # make this larger for real application (e.g., 10)
  nSGD = 400L, # make this larger for real applications (e.g., 2000L)
  batchSize = 50L, # make this larger for real application (e.g., 50L)
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


