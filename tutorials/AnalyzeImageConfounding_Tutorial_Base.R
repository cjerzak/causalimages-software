#!/usr/bin/env Rscript
{
  ################################
  # Image confounding tutorial using causalimages
  ################################
  
  # clean workspace 
  rm(list=ls()); options(error = NULL)
  
  # setup environment 
  if(Sys.getenv()["RSTUDIO_USER_IDENTITY"] == "cjerzak"){
     setwd("~/Downloads/")
  }
  if(Sys.getenv()["RSTUDIO_USER_IDENTITY"] != "cjerzak"){
    setwd("./")
    # or set directory as desired 
  }

  # remote install latest version of the package if needed
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

  # local install for development team
  # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

  # build backend you haven't ready (run this only once upon (re)installing causalimages!)
  # causalimages::BuildBackend()

  # load in package
  library( causalimages )

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
      # we also tweak the image dimensions for testing purposes
      #m_ <- FullImageArray[match(keys, KeysOfImages),c(1:35,1:35),c(1:35,1:35),1:2] # test with two channels
      m_ <- FullImageArray[match(keys, KeysOfImages),c(1:35,1:35),c(1:35,1:35),] # test with three channels
      #m_ <- FullImageArray[match(keys, KeysOfImages),c(1:35,1:35),c(1:35,1:35),c(1:3,1:2)] # test with five channels

      # if keys == 1, add the batch dimension so output dims are always consistent
      # (here in image case, dims are batch by height by width by channel)
      if(length(keys) == 1){ m_ <- array(m_,dim = c(1L,dim(m_)[1],dim(m_)[2],dim(m_)[3])) }

      return( m_ )
  }
    
  # look at one of the images 
  causalimages::image2( FullImageArray[1,,,1]  )
  
  # write tf record 
  # AnalyzeImageConfounding can efficiently stream batched image data from disk
  # (avoiding repeated in-memory loads and speeding up I/O during model training)
  TFRecordName_im <- "./TutorialData_im.tfrecord"
  if( reSaveTFRecord ){
    causalimages::WriteTfRecord(
      file = TFRecordName_im,
      uniqueImageKeys = unique(KeysOfObservations[ take_indices ]),
      acquireImageFxn = acquireImageFxn
    )
  }

  # perform causal inference with image-based and tabular confounding
  imageModelClass <- "VisionTransformer"
  optimizeImageRep <- TRUE # train the model to predict treatment, for use in IPW
  print(sprintf("Image confounding analysis & optimizeImageRep: %s & imageModelClass: %s",optimizeImageRep, imageModelClass))
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
    imageModelClass = imageModelClass,
    nDepth_ImageRep = ifelse(optimizeImageRep, yes = 1L, no = 1L),
    nWidth_ImageRep = as.integer(2L^6),
    learningRateMax = 0.001, nSGD = 300L, #
    dropoutRate = NULL, # 0.1,
    plotBands = c(1,2,3),
    plotResults = T, figuresTag = "ConfoundingImTutorial",
    figuresPath = "./")
    try(dev.off(), T)
  
  # Analyze in/out sample metrics
  ImageConfoundingAnalysis$ModelEvaluationMetrics
  
  # ATE estimate (image confounder adjusted)
  ImageConfoundingAnalysis$tauHat_propensityHajek
    
  # ATE se estimate (image confounder adjusted)
  ImageConfoundingAnalysis$tauHat_propensityHajek_se
    
  # some out-of-sample evaluation metrics
  ImageConfoundingAnalysis$ModelEvaluationMetrics
}
}
