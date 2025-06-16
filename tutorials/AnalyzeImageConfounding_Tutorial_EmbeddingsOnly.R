#!/usr/bin/env Rscript
{
  ################################
  # Image tutorial using embeddings 
  ################################
  
  # clean workspace 
  rm(list=ls()); options(error = NULL)
  
  # setup environment 
  setwd("~/Downloads/")
   
  # fetch image embeddings (requires an Internet connection)
  m_embeddings <- read.csv("https://huggingface.co/datasets/cjerzak/PCI_TutorialMaterial/resolve/main/nDepthIS1_analysisTypeISheterogeneity_imageModelClassISVisionTransformer_optimizeImageRepISclip-rsicd_MaxImageDimsIS64_dataTypeISimage_monte_iIS1_applicationISUganda_perturbCenterISFALSE.csv")
  
  # load data 
  library(causalimages)
  data( CausalImagesTutorialData )
  
  # image embedding dimensions - one image for each set of geo-located units
  # geo-location done by village name
  # pre-treatment image covariates via embeddings
  # obtained from an EO-fined tuned CLIP model
  # https://huggingface.co/flax-community/clip-rsicd-v2
  dim(m_embeddings)
  
  # correlation matrix
  cor(m_embeddings)
  
  # analyze via pca 
  pca_anlaysis <- predict(prcomp(m_embeddings,scale = T, center = T))
  
  # first two principal components 
  plot(pca_anlaysis[,1:2])
  
  # treatment indicator (recipient of YOP cash transfer)
  obsW 
   
  # outcome - measure of human capital post intervention 
  obsY
  
  # run double ML for image deconfounding
  # install.packages(c("DoubleML","mlr3","ranger"))
  library(DoubleML)      # double/debiased ML framework
  library(mlr3)          # core mlr3 infrastructure
  library(mlr3learners)  # access a wide range of learners
  
  # Combine outcome, treatment, and image embeddings into a single data.frame
  df_dml <- data.frame(
    Y = obsY,                # outcome vector
    W = obsW,                # treatment indicator
    m_embeddings             # precomputed image embeddings (one column per embedding dimension)
  )
  
  # Create a DoubleMLData object
  dml_data <- DoubleMLData$new(
    data   = df_dml,
    y_col  = "Y",
    d_cols = "W",
    x_cols = colnames(m_embeddings)
  )
  
  # Specify learners for the nuisance functions
  learner_g <- lrn("regr.ranger")                                # regression learner for E[Y|X,W]
  learner_m <- lrn("classif.ranger", predict_type = "prob")     # classification learner for P[W=1|X]
  
  # Instantiate the partially linear regression (PLR) DML model
  dml_plr <- DoubleMLPLR$new(
    dml_data,
    ml_g    = learner_g,
    ml_m    = learner_m,
    n_folds = 5       # number of folds for cross-fitting
  )
  
  # Fit the model and extract results
  dml_plr$fit()
  dml_plr$summary()   # prints estimated ATE and standard error
  
}
