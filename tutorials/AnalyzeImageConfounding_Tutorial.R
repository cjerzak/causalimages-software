#!/usr/bin/env Rscript
{
  ################################
  # Image confounding tutorial using causalimages
  ################################
  setwd("~/Downloads/")

  # remote install latest version of the package if needed
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

  # local install for development team
  # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

  # build backend you haven't ready:
  # causalimages::BuildBackend()

  # load in package
  library( causalimages )

  # resave TfRecords?
  reSaveTFRecord <- F

  # load in tutorial data
  data(  CausalImagesTutorialData )

  # mean imputation for toy example in this tutorial
  X <- apply(X[,-1],2,function(zer){
    zer[is.na(zer)] <- mean( zer,na.rm = T ); return( zer )
  })

  # select observation subset to make the tutorial quick
  set.seed(4321L);take_indices <- unlist( tapply(1:length(obsW),obsW,function(zer){sample(zer, 100)}) )

  # perform causal inference with image and tabular confounding
  {
  # example acquire image function (loading from memory)
  # in general, you'll want to write a function that returns images
  # that saved disk associated with keys
  acquireImageFxn <- function(keys){
      # here, the function input keys
      # refers to the unit-associated image keys
      # we also tweak the image dimensions for testing purposes
      #m_ <- FullImageArray[match(keys, KeysOfImages),c(1:35,1:35),c(1:35,1:35),1:2] # test with two channels
      #m_ <- FullImageArray[match(keys, KeysOfImages),c(1:35,1:35),c(1:35,1:35),] # test with three channels
      m_ <- FullImageArray[match(keys, KeysOfImages),c(1:35,1:35),c(1:35,1:35),c(1:3,1:2)] # test with five channels

      # if keys == 1, add the batch dimension so output dims are always consistent
      # (here in image case, dims are batch by height by width by channel)
      if(length(keys) == 1){ m_ <- array(m_,dim = c(1L,dim(m_)[1],dim(m_)[2],dim(m_)[3])) }

      return( m_ )
  }

  # write tf record
  TFRecordName_im <- "./ImageTutorial/TutorialData_im.tfrecord"
  if( reSaveTFRecord ){
    causalimages::WriteTfRecord(
      file = TFRecordName_im,
      uniqueImageKeys = unique(KeysOfObservations[ take_indices ]),
      acquireImageFxn = acquireImageFxn)
  }

  for(ImageModelClass in (c("VisionTransformer","CNN"))){
  for(optimizeImageRep in c(T,F)){
  print(sprintf("Image confounding analysis & optimizeImageRep: %s & ImageModelClass: %s",optimizeImageRep, ImageModelClass))
  ImageConfoundingAnalysis <- causalimages::AnalyzeImageConfounding(
    obsW = obsW[ take_indices ],
    obsY = obsY[ take_indices ],
    X = X[ take_indices,apply(X[ take_indices,],2,sd)>0],
    long = LongLat$geo_long[ take_indices ],  # optional argument
    lat = LongLat$geo_lat[ take_indices ], # optional argument
    imageKeysOfUnits = KeysOfObservations[ take_indices ],
    file = TFRecordName_im,

    batchSize = 16L,
    nBoot = 5L,
    optimizeImageRep = optimizeImageRep,
    ImageModelClass = ImageModelClass,
    nDepth_ImageRep = ifelse(optimizeImageRep, yes = 1L, no = 1L),
    nWidth_ImageRep = as.integer(2L^6),
    LEARNING_RATE_BASE = 0.001, nSGD = 10L, #
    dropoutRate = NULL, # 0.1,
    plotBands = c(1,2,3),
    plotResults = T, figuresTag = "ConfoundingImTutorial",
    figuresPath = "~/Downloads/ImageTutorial")
    try(dev.off(), T)
  #ImageConfoundingAnalysis$ModelEvaluationMetrics
  }
  }

  # ATE estimate (image confounder adjusted)
  ImageConfoundingAnalysis$tauHat_propensityHajek

  # ATE se estimate (image confounder adjusted)
  ImageConfoundingAnalysis$tauHat_propensityHajek_se

  # some out-of-sample evaluation metrics
  ImageConfoundingAnalysis$ModelEvaluationMetrics

}

  # perform causal inference with image *sequence*  and tabular confounding
  {
  acquireVideoRep <- function(keys) {
      # Note: this is a toy function generating image representations
      # that simply reuse a single temporal slice. In practice, we will
      # weant to read in images of different time periods.

      # Get image data as an array from disk
      tmp <- acquireImageFxn(keys)

      # Expand dimensions: we create a new dimension at the start
      tmp <- array(tmp, dim = c(1, dim(tmp)))

      # Transpose dimensions to get the target order
      tmp <- aperm(tmp, c(2, 1, 3, 4, 5))

      # Swap image dimensions to see variability across time
      tmp_ <- aperm(tmp, c(1, 2, 4, 3, 5))

      # Concatenate along the second axis
      tmp <- abind::abind(tmp, tmp, tmp_, tmp_, along = 2)

      return(tmp)
    }

  # write tf record
  TFRecordName_imSeq <- "./ImageTutorial/TutorialData_imSeq.tfrecord"
  if( reSaveTFRecord ){
      causalimages::WriteTfRecord(
        file = TFRecordName_imSeq,
        uniqueImageKeys = unique(KeysOfObservations[ take_indices ]),
        acquireImageFxn = acquireVideoRep,
        writeVideo = T)
  }

  for(ImageModelClass in c("VisionTransformer","CNN")){
  for(optimizeImageRep in c(T, F)){
    print(sprintf("Image seq confounding analysis & optimizeImageRep: %s & ImageModelClass: %s",optimizeImageRep, ImageModelClass))
    ImageSeqConfoundingAnalysis <- causalimages::AnalyzeImageConfounding(
      obsW = obsW[ take_indices ],
      obsY = obsY[ take_indices ],
      X = X[ take_indices,apply(X[ take_indices,],2,sd)>0],
      long = LongLat$geo_long[ take_indices ],
      lat = LongLat$geo_lat[ take_indices ],
      file = TFRecordName_imSeq, dataType = "video",
      imageKeysOfUnits = KeysOfObservations[ take_indices ],

      # model specifics
      batchSize = 16L,
      optimizeImageRep = optimizeImageRep,
      ImageModelClass = ImageModelClass,
      nDepth_ImageRep = ifelse(optimizeImageRep, yes = 1L, no = 1L),
      nWidth_ImageRep = as.integer(2L^7),
      LEARNING_RATE_BASE = 0.001, nSGD = 50L, #
      nBoot = 5L,
      plotBands = c(1,2,3),
      plotResults = T, figuresTag = "ConfoundingImSeqTutorial",
      figuresPath = "~/Downloads/ImageTutorial") # figures saved here
      try(dev.off(), T)
  }
  }

  # ATE estimate (image confounder adjusted)
  ImageSeqConfoundingAnalysis$tauHat_propensityHajek

  # ATE se estimate (image seq confounder adjusted)
  ImageSeqConfoundingAnalysis$tauHat_propensityHajek_se

  # some out-of-sample evaluation metrics
  ImageSeqConfoundingAnalysis$ModelEvaluationMetrics
  print("Done with confounding tutorial!")
}
}
