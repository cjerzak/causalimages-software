#!/usr/bin/env Rscript
{
  ################################
  # Simulated causal data with images using causalimages
  ################################
  
  # clean workspace 
  rm(list=ls()); options(error = NULL)
  
  # setup environment 
  # setwd as needed
  
  # Install causalimages if not installed
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")
  
  # load package
  library(causalimages)
  
  # Build backend if not already done
  # causalimages::BuildBackend()
  
  # Simulation parameters
  n <- 1000  # number of units
  tau <- 2.0  # true ATE (constant treatment effect)
  image_dim <- 64L  # image size
  
  # Generate synthetic data
  set.seed(12321)
  
  # Latent confounder C that generates images 
  C <- rnorm(n)
  
  # Tabular confounder X
  X <- matrix(rnorm(n * 2), ncol = 2)
  colnames(X) <- c("X1", "X2")
  
  # Note: we have a dual confounder setup here (tabular + image confounders)
  
  # Treatment assignment: depends on C and X
  logit_prob <- C + X[,1] + X[,2]
  prob_W <- 1 / (1 + exp(-logit_prob))
  obsW <- rbinom(n, 
                 size = 1, 
                 prob = prob_W)
  
  # Outcome: depends on both confounders and treatment and noise 
  obsY <- tau * obsW + C + X[,1] + X[,2] + rnorm(n)
  
  # Generate synthetic images based on C
  # For simplicity, create images where intensity depends on C
  # We'll make a 32x32x3 array, with red channel proportional to C
  KeysOfImages <- paste0("img_", 1:n)  # unique keys
  FullImageArray <- array(0, dim = c(n, image_dim, image_dim, 3))
  for(i in 1:n) {
    base_intensity <- (C[i] - min(C)) / (max(C) - min(C))  # normalize to [0,1]
    FullImageArray[i,,,1] <- base_intensity  # red channel
    FullImageArray[i,,,2] <- runif(1)  # green random
    FullImageArray[i,,,3] <- runif(1)  # blue random
    # Add a non-causal pattern to image, e.g., a gradient
    for(row in 1:image_dim) {
      FullImageArray[i, row,,] <- FullImageArray[i, row,,] * log(1+row / image_dim)^runif(n=1,min=0,max=1/i)
    }
  }
  
  # Keys of observations
  KeysOfObservations <- KeysOfImages  # one-to-one for simplicity
  
  # Define acquireImageFxn
  acquireImageFxn <- function(keys) {
    m_ <- FullImageArray[match(keys, KeysOfImages),,,]
    if(length(keys) == 1) {
      m_ <- array(m_, dim = c(1L, dim(m_)[1], dim(m_)[2], dim(m_)[3]))
    }
    return(m_)
  }
  
  # run once 
  # causalimages::BuildBackend()
  
  # Look at one image
  causalimages::image2(FullImageArray[sample(1:1000,1),,,1])
  
  # Write TFRecord (optional, but as in tutorial)
  reSaveTFRecord <- TRUE
  TFRecordName_im <- "~/Downloads/SimulatedData_im.tfrecord"
  if(reSaveTFRecord) {
    causalimages::WriteTfRecord(
      file = TFRecordName_im,
      uniqueImageKeys = unique(KeysOfObservations),
      acquireImageFxn = acquireImageFxn
    )
  }
  
  # Perform causal inference with image and tabular confounding
  imageModelClass <- "VisionTransformer"
  optimizeImageRep <- TRUE
  print(sprintf("Image confounding analysis & optimizeImageRep: %s & imageModelClass: %s", optimizeImageRep, imageModelClass))
  
  ImageConfoundingAnalysis <- causalimages::AnalyzeImageConfounding(
    # input data 
    obsW = obsW,
    obsY = obsY,
    X = X[, apply(X, 2, sd) > 0],
    imageKeysOfUnits = KeysOfObservations,
    file = TFRecordName_im,
    
    # modeling parameters 
    batchSize = 16L,
    nBoot = 5L,
    optimizeImageRep = TRUE,
    imageModelClass = imageModelClass,
    nDepth_ImageRep = 4L,
    nWidth_ImageRep = as.integer(2^8),
    learningRateMax = 0.001, nSGD = 300L,
    dropoutRate = 0.1,
    plotBands = c(1,2,3),
    plotResults = TRUE, figuresTag = "SimConfoundingIm",
    figuresPath = "./"
  )
  try(dev.off(), TRUE)
  
  # Output results
  print("True ATE:")
  print(tau)
  
  print("Estimated ATE (no confounder adjustment):")
  print(ImageConfoundingAnalysis$tauHat_diffInMeans)
  
  print("Estimated ATE (image+tabular confounder adjusted):")
  print(ImageConfoundingAnalysis$tauHat_propensityHajek)
  
  
  print("Estimated SE:")
  print(ImageConfoundingAnalysis$tauHat_propensityHajek_se)
  
  print("Model Evaluation Metrics:")
  print(ImageConfoundingAnalysis$ModelEvaluationMetrics)
  
  # Comparison
  bias_naive <- ImageConfoundingAnalysis$tauHat_diffInMeans - tau
  print("Bias (diff in means):")
  print(bias_naive)
  
  bias_adjusted <- ImageConfoundingAnalysis$tauHat_propensityHajek - tau
  print("Bias (image+tabular deconfounding):")
  print(bias_adjusted)
}
