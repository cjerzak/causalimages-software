#!/usr/bin/env Rscript
{
################################
# Image heterogeneity tutorial using causalimages
################################

# remote install latest version of the package
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# local install for development team
# install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

# build backend you haven't ready:
# causalimages::BuildBackend()

# run code if downloading data for the first time
download_folder <- "~/Downloads/UgandaAnalysis.zip"
reSaveTfRecords <- T
if( reDownloadRawData <- F  ){
  # specify uganda data URL
  uganda_data_url <- "https://dl.dropboxusercontent.com/s/xy8xvva4i46di9d/Public%20Replication%20Data%2C%20YOP%20Experiment.zip?dl=0"
  download_folder <- "~/Downloads/UgandaAnalysis.zip"

  # download into new directory
  download.file( uganda_data_url,  destfile = download_folder)

  # unzip and list files
  unzip(download_folder, exdir = "~/Downloads/UgandaAnalysis")
}

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
table( UgandaDataProcessed$age )

# approximate longitude + latitude for units
head(  cbind(UgandaDataProcessed$geo_long, UgandaDataProcessed$geo_lat) )

# image keys of units (use for referencing satellite images)
UgandaDataProcessed$geo_long_lat_key

# an experimental outcome
UgandaDataProcessed$Yobs

# treatment variable
UgandaDataProcessed$Wobs

# information on keys linking to satellite images for all of Uganda
# (not just experimental context, use for constructing transportability maps)
UgandaGeoKeyMat <- read.csv(  "./UgandaGeoKeyMat.csv"  )

# set outcome to an income index
UgandaDataProcessed$Yobs <- UgandaDataProcessed$income_index_e_RECREATED

# drop observations with NAs in key variables
# (you can also use a multiple imputation strategy)
UgandaDataProcessed <- UgandaDataProcessed[!is.na(UgandaDataProcessed$Yobs) &
                                             !is.na(UgandaDataProcessed$Wobs) &
                                             !is.na(UgandaDataProcessed$geo_lat) , ]

# sanity checks
{
  # write a function that reads in images as saved and process them into an array
  NBANDS <- 3L
  imageHeight <- imageWidth <- 351L #  pixel height/width
  acquireImageRep <- function(keys){
    # initialize an array shell to hold image slices
    array_shell <- array(NA, dim = c(1L, imageHeight, imageWidth, NBANDS))

    # iterate over keys:
    # -- images are referenced to keys
    # -- keys are referenced to units (to allow for duplicate images uses)
    array_ <- sapply(keys, function(key_) {
      # iterate over all image bands (NBANDS = 3 for RBG images)
      for (band_ in 1:NBANDS) {
        # place the image in the correct place in the array
        array_shell[,,,band_] <-
          as.matrix(data.table::fread(
            input = sprintf("./Uganda2000_processed/GeoKey%s_BAND%s.csv", key_, band_), header = FALSE)[-1,])
      }
      return(array_shell)
    }, simplify = "array")

    # return the array in the format c(nBatch, imageWidth, imageHeight, nChannels)
    # ensure that the dimensions are correctly ordered for further processing
    if(length(keys) > 1){ array_ <- aperm(array_[1,,,,], c(4, 1, 2, 3) ) }
    if(length(keys) == 1){
      array_ <- aperm(array_, c(1,5, 2, 3, 4))
      array_ <- array(array_, dim(array_)[-1])
    }

    return(array_)
  }

  # try out the function
  # note: some units are co-located in same area (hence, multiple observations per image key)
  ImageBatch <- acquireImageRep( UgandaDataProcessed$geo_long_lat_key[ check_indices <- c(1, 20, 50, 101)  ])
  acquireImageRep( UgandaDataProcessed$geo_long_lat_key[ check_indices[1] ]   )

  # sanity checks in the analysis of earth observation data are essential
  # check that images are centered around correct location
  causalimages::image2(  as.array(ImageBatch)[1,,,1] )
  UgandaDataProcessed$geo_long[check_indices[1]]
  UgandaDataProcessed$geo_lat[check_indices[1]]
  # check against google maps to confirm correctness
  # https://www.google.com/maps/place/1%C2%B018'16.4%22N+34%C2%B005'15.1%22E/@1.3111951,34.0518834,10145m/data=!3m1!1e3!4m4!3m3!8m2!3d1.3045556!4d34.0875278?entry=ttu

  # scramble data (important for reading into causalimages::WriteTfRecord
  # to ensure no systematic biases in data sequence with model training
  set.seed(144L); UgandaDataProcessed <- UgandaDataProcessed[sample(1:nrow(UgandaDataProcessed)),]
}

# Image heterogeneity example with tfrecords
for(optimizeImageRep in c(T, F)){
  # write a tf records repository
  # whenever changes are made to the input data to AnalyzeImageHeterogeneity, WriteTfRecord() should be re-run
  # to ensure correct ordering of data
  tfrecord_loc <- "~/Downloads/UgandaExample.tfrecord"
  if( reSaveTfRecords ){
    causalimages::WriteTfRecord(  file = tfrecord_loc,
                    uniqueImageKeys = unique(UgandaDataProcessed$geo_long_lat_key),
                    acquireImageFxn = acquireImageRep )
  }

  # perform image heterogeneity analysis (toy example)
  ImageHeterogeneityResults <- causalimages::AnalyzeImageHeterogeneity(
    # data inputs
    obsW =  UgandaDataProcessed$Wobs,
    obsY = UgandaDataProcessed$Yobs,
    X = matrix(rnorm(length(UgandaDataProcessed$Yobs)*10),ncol=10),
    imageKeysOfUnits =  UgandaDataProcessed$geo_long_lat_key,
    file = tfrecord_loc, # location of tf record (use absolute file paths)
    lat =  UgandaDataProcessed$geo_lat, # not required but helpful for dealing with redundant locations in EO data
    long =  UgandaDataProcessed$geo_long, # not required but helpful for dealing with redundant locations in EO data

    # inputs to control where visual results are saved as PDF or PNGs
    # (these image grids are large and difficult to display in RStudio's interactive mode)
    plotResults = T,
    figuresPath = "~/Downloads/HeteroTutorial", # where to write analysis figures
    figuresTag = "causalimagesTutorial",plotBands = 1L:3L,

    # optional arguments for generating transportability maps
    # here, we leave those NULL for simplicity
    transportabilityMat = NULL, #

    # other modeling options
    nSGD = 50L, # make this larger for real applications (e.g., 2000L)
    nDepth_ImageRep = ifelse(optimizeImageRep, yes = 1L, no = 1L),
    nWidth_ImageRep = as.integer(2L^6),
    optimizeImageRep = optimizeImageRep,
    batchSize = 8L, # make this larger for real application (e.g., 50L)
    kClust_est = 2 # vary depending on problem. Usually < 5
    )
}

# video heterogeneity example
{
  acquireVideoRep <- function(keys) {
    # Get image data as an array from disk
    tmp <- acquireImageRep(keys)

    # Expand dimensions: we create a new dimension at the start
    tmp <- array(tmp, dim = c(1, dim(tmp)))

    # Transpose dimensions to get the required order
    tmp <- aperm(tmp, c(2, 1, 3, 4, 5))

    # Swap image dimensions to see variability across time
    tmp_ <- aperm(tmp, c(1, 2, 4, 3, 5))

    # Concatenate along the second axis
    tmp <- abind::abind(tmp, tmp_, along = 2)

    return(tmp)
  }

  # write the tf records repository
  tfrecord_loc_imSeq <- "~/Downloads/UgandaExampleVideo.tfrecord"
  if(reSaveTfRecords){
    causalimages::WriteTfRecord(  file = tfrecord_loc_imSeq,
                    uniqueImageKeys = unique(UgandaDataProcessed$geo_long_lat_key),
                    acquireImageFxn = acquireVideoRep, writeVideo = T )
  }
  for(optimizeImageRep in c(F)){
  # Note: optimizeImageRep = T breaks with video
  VideoHeterogeneityResults <- causalimages::AnalyzeImageHeterogeneity(
    # data inputs
    obsW =  UgandaDataProcessed$Wobs,
    obsY = UgandaDataProcessed$Yobs,
    imageKeysOfUnits =  UgandaDataProcessed$geo_long_lat_key,
    file = tfrecord_loc_imSeq, # location of tf record (absolute paths are safest)
    dataType = "video",
    lat =  UgandaDataProcessed$geo_lat, # not required but helpful for dealing with redundant locations in EO data
    long =  UgandaDataProcessed$geo_long, # not required but helpful for dealing with redundant locations in EO data

    # inputs to control where visual results are saved as PDF or PNGs
    # (these image grids are large and difficult to display in RStudio's interactive mode)
    plotResults = T,
    figuresPath = "~/Downloads/HeteroTutorial",
    plotBands = 1L:3L, figuresTag = "causalimagesTutorial",

    # optional arguments for generating transportability maps
    # here, we leave those NULL for simplicity
    transportabilityMat = NULL, #

    # other modeling options
    nSGD = 50L, # make this larger for real applications (e.g., 2000L)
    nDepth_ImageRep = ifelse(optimizeImageRep, yes = 1L, no = 1L),
    nWidth_ImageRep = as.integer(2L^6),
    optimizeImageRep = optimizeImageRep,
    kClust_est = 2, # vary depending on problem. Usually < 5
    batchSize = 8L, # make this larger for real application (e.g., 50L)
    strides = 2L )
  }
}

print("Done with image heterogeneity tutorial!")
}
