#!/usr/bin/env Rscript
{
################################
# Full image heterogeneity tutorial using causalimages
# Note: In future builds of causalimages, we will
# add easier ways to download and store satellite imagery.
# Stay tuned! Contributions & bug fixes are welcome!
################################

# run code if downloading data for the first time
download_folder <- "~/Downloads/UgandaAnalysis.zip"
if(T == F){
  # specify uganda data URL
  uganda_data_url <- "https://dl.dropboxusercontent.com/s/xy8xvva4i46di9d/Public%20Replication%20Data%2C%20YOP%20Experiment.zip?dl=0"
  download_folder <- "~/Downloads/UgandaAnalysis.zip"

  # download into new directory
  download.file( uganda_data_url,  destfile = download_folder)

  # unzip and list files
  unzip(download_folder, exdir = "~/Downloads/UgandaAnalysis")
}

# remote install latest version of the package
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# local install for development team
# install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

# load in package
library( causalimages  )

# set new wd
setwd(sprintf('%s/Public Replication Data, YOP Experiment/',
              gsub(download_folder,pattern="\\.zip",replace="")))

# see directory contents
list.files()

# images saved here
list.files(  "./Uganda2000_processed"  )

# individual-level data
UgandaDataProcessed <- read.csv(  "./UgandaDataProcessed.csv"  )

# unit-level covariates (many covariates are subject to missingness!)
dim( UgandaDataProcessed )
table( UgandaDataProcessed$female )
table( UgandaDataProcessed$age )

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
tail( UgandaGeoKeyMat )

# set outcome to an income index
UgandaDataProcessed$Yobs <- UgandaDataProcessed$income_index_e_RECREATED

# drop observations with NAs in key variables
# (you can also use a multiple imputation strategy)
UgandaDataProcessed <- UgandaDataProcessed[!is.na(UgandaDataProcessed$Yobs) &
                                             !is.na(UgandaDataProcessed$Wobs) &
                                             !is.na(UgandaDataProcessed$geo_lat) , ]

# sanity checks
{
  # load in tensorflow (edit for your computer)
  library(tensorflow); library(keras)
  tensorflow::use_condaenv("tensorflow_m1", required = T)
  #try( tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]],T),T)
  #try( tf$config$set_soft_device_placement( T ) , T)

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
  check_indices <- c(1, 20, 50, 101)
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
}

# image-based heterogeneity analysis
if(T == F){
ImageHeterogeneityResults <- AnalyzeImageHeterogeneity(
  # data inputs
  obsW =  UgandaDataProcessed$Wobs,
  obsY = UgandaDataProcessed$Yobs,
  imageKeysOfUnits =  UgandaDataProcessed$geo_long_lat_key,
  acquireImageFxn = acquireImageFromDisk,
  conda_env = "tensorflow_m1", # change "tensorflow_m1" to the location of your conda environment containing tensorflow v2 and tensorflow_probability,
  conda_env_required = T,
  X = X,
  plotBands = 1L,
  lat =  UgandaDataProcessed$geo_lat, # not required but helpful for dealing with redundant locations in EO data
  long =  UgandaDataProcessed$geo_long, # not required but helpful for dealing with redundant locations in EO data

  # inputs to control where visual results are saved as PDF or PNGs
  # (these image grids are large and difficult to display in RStudio's interactive mode)
  plotResults = T,
  figuresPath = "~/Downloads/",
  printDiagnostics = T,
  figuresTag = "causalimagesTutorial",

  # optional arguments for generating transportability maps
  # here, we leave those NULL for simplicity
  transportabilityMat = NULL, #

  # other modeling options
  #modelClass = "cnn", # uses CNN image modeling class
  modelClass = "embeddings", # uses image/video embeddings model class
  orthogonalize = F,
  heterogeneityModelType = "variational_minimal",
  kClust_est = 2, # vary depending on problem. Usually < 5
  nMonte_variational = 2L, # make this larger for real application (e.g., 10)
  nSGD = 4L, # make this larger for real applications (e.g., 2000L)
  batchSize = 34L, # make this larger for real application (e.g., 50L)
  compile = T,
  channelNormalize = T,
  yDensity = "normal",
  kernelSize = 3L, maxPoolSize = 2L, strides = 2L,
  nDepthHidden_conv = 2L, # in practice, nDepthHidden_conv would be more like 4L
  nFilters = 64L, # vary the following depending on image type and GPU memory
  nDepthHidden_dense = 0L,
  nDenseWidth = 32L,
  nDimLowerDimConv = 3L,
  reparameterizationType = "Flipout"
)
}

# video heterogeneity example
# example video function (this here just appends two identical images for illustration only)
# in practice, actual image sequence / video data will be read from disk
acquireVideoRepFromDisk <- function(keys, training = F){
  tmp <- acquireImageFromDisk(keys, training = training)
  tmp <- tf$expand_dims(tmp,0L)
  tmp <- tf$transpose(tmp,c(1L,0L,2L,3L,4L))
  tmp_ <- tf$transpose(tmp,c(0L,1L,3L,2L,4L)) # swap image dims too to see variability across time
  tmp <- tf$concat(list(tmp,tmp_),axis = 1L)
  return(  tmp  )
}
if(T == F){
# image2( as.array(acquireVideoRepFromDisk(UgandaDataProcessed$geo_long_lat_key[1:5]))[1,1,,,1] )
# image2( as.array(acquireVideoRepFromDisk(UgandaDataProcessed$geo_long_lat_key[1:5]))[1,2,,,1] )
dim( acquireVideoRepFromDisk(UgandaDataProcessed$geo_long_lat_key[1:5]) )
dim( acquireVideoRepFromDisk(UgandaDataProcessed$geo_long_lat_key[1]) )
VideoHeterogeneityResults <- AnalyzeImageHeterogeneity(
  # data inputs
  obsW =  UgandaDataProcessed$Wobs,
  obsY = UgandaDataProcessed$Yobs,
  imageKeysOfUnits =  UgandaDataProcessed$geo_long_lat_key,
  acquireImageFxn = acquireVideoRepFromDisk, # this is new!
  conda_env = "tensorflow_m1", # change "tensorflow_m1" to the location of your conda environment containing tensorflow v2 and tensorflow_probability,
  conda_env_required = T,
  X = X,
  plotBands = 1L:3L, # select 3 bands to plot
  #plotBands = 1L, # or select 1 bands to plot
  lat =  UgandaDataProcessed$geo_lat, # not required but helpful for dealing with redundant locations in EO data
  long =  UgandaDataProcessed$geo_long, # not required but helpful for dealing with redundant locations in EO data

  # inputs to control where visual results are saved as PDF or PNGs
  # (these image grids are large and difficult to display in RStudio's interactive mode)
  plotResults = T,
  figuresPath = "~/Downloads/",
  printDiagnostics = T,
  figuresTag = "causalimagesTutorial",

  # optional arguments for generating transportability maps
  # here, we leave those NULL for simplicity
  transportabilityMat = NULL, #

  # other modeling options
  #modelClass = "cnn", kernelSize = 3L, # CNN image modeling class
  modelClass = "embeddings", nEmbedDim = 64L, kernelSize = 9L, temporalKernelSize = 2L, # image/video embeddings model class
  orthogonalize = F,
  heterogeneityModelType = "variational_minimal",
  kClust_est = 2, # vary depending on problem. Usually < 5
  nMonte_variational = 10L, # make this larger for real application (e.g., 10)
  nSGD = 50L, # make this larger for real applications (e.g., 2000L)
  batchSize = 34L, # make this larger for real application (e.g., 50L)
  compile = T,
  channelNormalize = T,
  yDensity = "normal",
  maxPoolSize = 2L, strides = 2L,
  nDepthHidden_conv = 2L, # in practice, nDepthHidden_conv would be more like 4L
  nFilters = 64L, # vary the following depending on image type and GPU memory
  nDepthHidden_dense = 0L,
  nDenseWidth = 50L,
  nDimLowerDimConv = 3L,
  reparameterizationType = "Flipout"
)
}

# Image heterogeneity example with tfrecords (faster)
if(T == F){
  # scramble data so no patterning in the tf record sequence
  # make sure to set seed so you can re-use the saved tfrecord
  tfrecord_loc <- "~/Downloads/UgandaExample.tfrecord"
  set.seed(144L); UgandaDataProcessed_ <- UgandaDataProcessed[sample(1:nrow(UgandaDataProcessed)),]

  if(T == F){
    # run code to (re)create tfrecord
    # write a tf records repository
    # whenever changes are made to the
    # input data to AnalyzeImageHeterogeneity,
    #  WriteTfRecord() should be re-run
    WriteTfRecord(  file = tfrecord_loc,
                    imageKeys = UgandaDataProcessed_$geo_long_lat_key,
                    acquireImageFxn = acquireImageFromDisk,
                    conda_env = "tensorflow_m1"  )
  }

  ImageHeterogeneityResults <- AnalyzeImageHeterogeneity(
    # data inputs
    obsW =  UgandaDataProcessed_$Wobs,
    obsY = UgandaDataProcessed_$Yobs,
    imageKeysOfUnits =  UgandaDataProcessed_$geo_long_lat_key,
    file = tfrecord_loc, # location of tf record (absolute paths are safest)
    acquireImageFxn = NULL,
    conda_env = "tensorflow_m1", # change "tensorflow_m1" to the location of your conda environment containing tensorflow v2 and tensorflow_probability,
    conda_env_required = T,
    X = X,
    plotBands = 1L:3L,
    lat =  UgandaDataProcessed_$geo_lat, # not required but helpful for dealing with redundant locations in EO data
    long =  UgandaDataProcessed_$geo_long, # not required but helpful for dealing with redundant locations in EO data

    # inputs to control where visual results are saved as PDF or PNGs
    # (these image grids are large and difficult to display in RStudio's interactive mode)
    plotResults = T,
    figuresPath = "~/Downloads/",
    printDiagnostics = T,
    figuresTag = "causalimagesTutorial",

    # optional arguments for generating transportability maps
    # here, we leave those NULL for simplicity
    transportabilityMat = NULL, #

    # other modeling options
    #modelClass = "cnn",  kernelSize = 3L, # CNN image modeling class
    modelClass = "embeddings", nEmbedDim = 128L, kernelSize = 9L,# image embeddings model class
    nSGD = 5L, # make this larger for real applications (e.g., 2000L)
    LEARNING_RATE_BASE = 0.005,
    orthogonalize = F,
    heterogeneityModelType = "variational_minimal",
    kClust_est = 2, # vary depending on problem. Usually < 5
    nMonte_variational = 10L, # make this larger for real application (e.g., 10)
    batchSize = 32L, # make this larger for real application (e.g., 50L)
    compile = T,
    channelNormalize = T,
    yDensity = "normal",
    maxPoolSize = 2L, strides = 3L,
    nDepthHidden_conv = 2L, # in practice, nDepthHidden_conv would be more like 4L
    nFilters = 50L, # vary the following depending on image type and GPU memory
    nDepthHidden_dense = 0L,
    nDenseWidth = 50L,
    nDimLowerDimConv = 3L,
    reparameterizationType = "Flipout"
  )
}

# video heterogeneity example with tfrecords (faster)
if(T == T){
  # scramble data so no patterning in the tf record sequence
  # make sure to set seed so you can re-use the saved tfrecord
  tfrecord_loc <- "~/Downloads/UgandaExampleVideo.tfrecord"
  set.seed(144L); UgandaDataProcessed_ <- UgandaDataProcessed[sample(1:nrow(UgandaDataProcessed)),]
  # we keep a copy of the original data but
  # will *always* use the shuffled copy for *all future analyzes*
  # where tf records are involved to avoid bugs!

  if(T == F){
    # run code to (re)create tfrecord
    # write a tf records repository
    WriteTfRecord(  file = tfrecord_loc,
                    imageKeys = UgandaDataProcessed_$geo_long_lat_key,
                    acquireImageFxn = acquireVideoRepFromDisk,
                    writeVideo = T,
                    conda_env = "tensorflow_m1"  )
  }

  VideoHeterogeneityResults <- AnalyzeImageHeterogeneity(
    # data inputs
    obsW =  UgandaDataProcessed_$Wobs,
    obsY = UgandaDataProcessed_$Yobs,
    imageKeysOfUnits =  UgandaDataProcessed_$geo_long_lat_key,
    file = tfrecord_loc, # location of tf record (absolute paths are safest)
    dataType = "video",
    acquireImageFxn = NULL,
    conda_env = "tensorflow_m1", # change "tensorflow_m1" to the location of your conda environment containing tensorflow v2 and tensorflow_probability,
    conda_env_required = T,
    X = X,
    plotBands = 1L:3L,
    lat =  UgandaDataProcessed_$geo_lat, # not required but helpful for dealing with redundant locations in EO data
    long =  UgandaDataProcessed_$geo_long, # not required but helpful for dealing with redundant locations in EO data

    # inputs to control where visual results are saved as PDF or PNGs
    # (these image grids are large and difficult to display in RStudio's interactive mode)
    plotResults = T,
    figuresPath = "~/Downloads/",
    printDiagnostics = T,
    figuresTag = "causalimagesTutorial",

    # optional arguments for generating transportability maps
    # here, we leave those NULL for simplicity
    transportabilityMat = NULL, #

    # other modeling options
    #modelClass = "cnn", kernelSize = 3L, # CNN image modeling class
    modelClass = "embeddings", nEmbedDim = 64L, kernelSize = 9L, temporalKernelSize = 2L, # image/video embeddings model class
    nSGD = 10L, # make this larger for real applications (e.g., 2000L)
    orthogonalize = F,
    heterogeneityModelType = "variational_minimal",
    kClust_est = 2, # vary depending on problem. Usually < 5
    nMonte_variational = 5L, # make this larger for real application (e.g., 10)
    batchSize = 50L, # make this larger for real application (e.g., 50L)
    compile = T,
    channelNormalize = T,
    yDensity = "normal",
    maxPoolSize = 2L, strides = 3L,
    nDepthHidden_conv = 2L, # in practice, nDepthHidden_conv would be more like 4L
    nFilters = 64L, # vary the following depending on image type and GPU memory
    nDepthHidden_dense = 0L,
    nDenseWidth = 32L,
    nDimLowerDimConv = 3L,
    reparameterizationType = "Flipout"
  )
}
}
