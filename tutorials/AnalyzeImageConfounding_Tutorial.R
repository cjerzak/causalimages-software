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
  library( causalimages  )

  # resave TfRecords?
  reSaveTFRecord <- T

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
      m_ <- FullImageArray[match(keys, KeysOfImages),,,]

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

  for(optimizeImageRep in c(F,T)){
  ImageConfoundingAnalysis <- causalimages::AnalyzeImageConfounding(
    obsW = obsW[ take_indices ],
    obsY = obsY[ take_indices ],
    X = X[ take_indices,apply(X[ take_indices,],2,sd)>0],
    long = LongLat$geo_long[ take_indices ],  # optional argument
    lat = LongLat$geo_lat[ take_indices ], # optional argument
    imageKeysOfUnits = KeysOfObservations[ take_indices ],
    file = TFRecordName_im,

    batchSize = 16L,
    nBoot = 10L,
    optimizeImageRep = optimizeImageRep,
    LEARNING_RATE_BASE = 0.005, nSGD = 100, #
    nWidth_ImageRep = as.integer(2L^7),
    strides = 2L,
    plotBands = c(1,2,3),
    plotResults = T, figuresTag = "TutorialExample",
    figuresPath = "~/Downloads/ImageTutorial")
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
      # Get image data as an array from disk
      tmp <- acquireImageFxn(keys)

      # Expand dimensions: we create a new dimension at the start
      tmp <- array(tmp, dim = c(1, dim(tmp)))

      # Transpose dimensions to get the required order
      # Original: tf$transpose(tmp, c(1L, 0L, 2L, 3L, 4L))
      # R equivalent: aperm with the new order of dimensions
      tmp <- aperm(tmp, c(2, 1, 3, 4, 5))

      # Swap image dimensions to see variability across time
      # Original: tf$transpose(tmp, c(0L, 1L, 3L, 2L, 4L))
      # R equivalent: aperm with the new order of dimensions
      tmp_ <- aperm(tmp, c(1, 2, 4, 3, 5))

      # Concatenate along the second axis
      # Note: R does not have a direct equivalent of tf$concat,
      # so we use abind from the 'abind' package
      if (requireNamespace("abind", quietly = TRUE)) {
        tmp <- abind::abind(tmp, tmp_, along = 2)
      } else {
        stop("The 'abind' package is required for this function to work.")
      }

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

  for(optimizeImageRep in c(F,T)){
    optimizeImageRep <- F
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
      LEARNING_RATE_BASE = 0.005, nSGD = 100, #
      nWidth_ImageRep = as.integer(2L^7),
      strides = 2L,
      nBoot = 10L,
      plotBands = c(1,2,3),
      plotResults = T, figuresTag = "TutorialExample",
      figuresPath = "~/Downloads/ImageTutorial") # figures saved here
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
