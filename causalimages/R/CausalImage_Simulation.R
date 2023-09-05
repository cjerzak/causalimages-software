#!/usr/bin/env Rscript
#' Simulate causal systems involving images
#'
#' This function generates simulated causal structures using images. It is currently under construction.
#'
#' @usage
#'
#' SimulateImageSystem(...)
#'
#' @param dag \emph{(character string)} An input DAG specifying causal structure.
#' This input should be of the form \url{`i->t,i->y,t->y,....'}
#' Currently, only one node in a DAG can be an image (this should be labeled ``\url{i}'').
#' The non-image nodes can have arbitrary string labels.
#' The image can be a confounder, effect moderator, effect mediator.
#' If the image is to be used as a moderator, use the notation, \url{t-i>y}.
#'
#' @param treatment \emph{(character string, optional)} In estimation mode, users specify the treatment variable here.
#' If \url{treatment} is specified, users must provide other data inputs to the DAG (see \url{...}).
#'
#' @param image_pool \emph{(character string, optional)} The path to where analysis specific
#' images are located. This can be specified both in simulation and estimation mode.
#' If not specified, the simulation uses a pool of Landsat images from Nigeria.
#'
#' @param analysis_level \emph{(character string, default is \url{`scene'})} Defines the unit
#' of analysis used in the simulation framework. This is ignored in estimation mode,
#' where the unit of analysis is inferred from the data dimensions.
#'
#' @param control \emph{(list)} A list containing control parameters in the data generating process.
#'
#' @param ... \emph{(optional)} In estimation mode, users input the data matrices associated with the
#' non-image nodes of \url{DAG} and image node \url{i}. For example, if \url{x} is a DAG node,
#' users must, in estimation mode, supply data to \url{x} in a form that can be coerced to a tensor.
#'
#'
#' @return A list:
#' \itemize{
#' \item In \emph{simulation mode}, the function returns a list with as many elements as
#' unique nodes in \verb{DAG}. Each element represents the simulated data.
#'
#' \item In \emph{estimation mode},the function returns an estimated treatment effect with 95\% confidence intervals.
#' }
#'
#' @section References:
#' \itemize{
#' \item Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Image-based Treatment Effect Heterogeneity. Forthcoming in *Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)*, 2023.
#' }
#'
#' @examples
#' #set seed
#' set.seed(1)
#'
#' # Simulation mode
#' #simulatedData <- causalimage('r->i, i->t, t->y, r->y')
#' #print(names(simulatedData))
#'
#' # Estimation mode
#' #estimatedResults <- causalimage('r->i, i->t, t->y, r->y', y=y, r=r, y=y', treatment='t')
#' #print( estimatedResults )
#'
#' @export
#' @md

SimulateImageSystem <- function(dag = NULL,...){

  dag <- "i->t,i->y,t->y"
  dag_vec <- strsplit(dag,split=",")[[1]]
  dag_mat <- do.call(rbind,sapply(dag_vec,function(zer){
    strsplit(zer,split = "->") }) )
  # load in tensorflow + helper fxns
  {
    library(tensorflow); library(keras)
    try(tensorflow::use_python(python = "/Users/cjerzak/miniforge3/bin/python", required = T),T)
    try(tensorflow::use_condaenv("tensorflow_m1", required = T, conda = "/opt/miniconda3/envs/tensorflow_m1"),T)
    try(tf$sqrt(1.),T)
    try(tf$config$list_physical_devices('CPU'),T)
    try(tf$config$list_physical_devices('GPU'),T)
  }

  # obtain raw images
  IMAGE_DIAMETER <- nrow(read.csv(image_pool[1])[,-1])
  ImageBlocks <- array(NA, dim = targetDims)
  for(i in 1:nImages){
    str_name <- sample(image_pool, 1)
    starting_row <- sample(1:(IMAGE_DIAMETER-nSpatialDims[1]),1)
    starting_col <- sample(1:(IMAGE_DIAMETER-nSpatialDims[2]),1)

    for(band_ in 1:nBands){
      newEntry <- as.matrix(read.csv(image_pool[1])[,-1])
      #eval(parse(text = sprintf("band%s <- raster(str_name, band = %s)",band_,band_)))
      #eval(parse(text = sprintf("BAND_ = band%s",band_)))
      #newEntry <-  matrix(raster::getValuesBlock(BAND_,
      #row = 1, nrows = nrow(BAND_), col = 1, ncols = ncol(BAND_), format = "m", lyrs = 1L),
      #nrow = nrow(BAND_), ncol = ncol(BAND_), byrow=T)
      newEntry <- newEntry[starting_row:(starting_row+nSpatialDims[1]-1),
                           starting_col:(starting_col+nSpatialDims[2]-1)]
      for(t_ in 1:nPeriods){
        ImageBlocks[i,t_,,,band_] <- newEntry
      }
    }
  }

  # clear workspace and set wd + seed
  {
    setwd("~/Dropbox/ObservatoryOfPoverty")
    image_directory <- "./Data/Nigeria2000_processed"
    image_pool <- paste(image_directory,list.files(image_directory), sep="/")
    image_pool <- image_pool[grepl(image_pool,pattern="BAND1")]
  }

  # parameters
  {
    DIAG_PARAM <- 1
    trueTau <- 1
    nImages <- 500L
    nPeriods <- 2L
    nBands <- 1L
    nSpatialDims <- as.integer(rep(2^8,2))
    targetDims <- c(nImages, nPeriods,nSpatialDims,nBands)
  }

  # obtain images
  {
    IMAGE_DIAMETER <- nrow(read.csv(image_pool[1])[,-1])
    ImageBlocks <- array(NA, dim = targetDims)
    for(i in 1:nImages){
      str_name <- sample(image_pool, 1)
      starting_row <- sample(1:(IMAGE_DIAMETER-nSpatialDims[1]),1)
      starting_col <- sample(1:(IMAGE_DIAMETER-nSpatialDims[2]),1)

      for(band_ in 1:nBands){
        newEntry <- as.matrix(read.csv(image_pool[1])[,-1])
        #eval(parse(text = sprintf("band%s <- raster(str_name, band = %s)",band_,band_)))
        #eval(parse(text = sprintf("BAND_ = band%s",band_)))
        #newEntry <-  matrix(raster::getValuesBlock(BAND_,
        #row = 1, nrows = nrow(BAND_), col = 1, ncols = ncol(BAND_), format = "m", lyrs = 1L),
        #nrow = nrow(BAND_), ncol = ncol(BAND_), byrow=T)
        newEntry <- newEntry[starting_row:(starting_row+nSpatialDims[1]-1),
                             starting_col:(starting_col+nSpatialDims[2]-1)]
        for(t_ in 1:nPeriods){
          ImageBlocks[i,t_,,,band_] <- newEntry
        }
      }
    }
  }

  # simulation setup
  {
    BAND_SELECT <- 1#which.max( tmp_sd )
    kernelWidth <- 32L
    imageWidth <- imageHeight <- dim(ImageBlocks)[3]
    ImageBlocks_tf <- ImageBlocks[,1,,,BAND_SELECT]
    ImageBlocks_tf <- (ImageBlocks_tf - mean(ImageBlocks_tf)) / (0.01+sd(ImageBlocks_tf))
    ImageBlocks_tf <- tf$cast(ImageBlocks_tf,dtype=tf$float32)
    ImageBlocks_tf <- tf$expand_dims(ImageBlocks_tf,1L)
    ImageBlocks_tf <- tf$expand_dims(ImageBlocks_tf,4L)

    # define convolution
    ConfoundingConv <- keras$layers$Conv2D(filters=1L,
                                           kernel_size=c(kernelWidth,kernelWidth),
                                           activation="linear",
                                           padding = 'valid')
    ResponseConv <- keras$layers$Conv2D(filters=1L,
                                        kernel_size=c(kernelWidth,kernelWidth),
                                        activation="linear",
                                        padding = 'valid')

    # initialize convolution - start for next time
    ConfoundingConv(  ImageBlocks_tf )
    ResponseConv(  ImageBlocks_tf )

    # fix convolution kernel for confounding
    ConvolveInit <- matrix(0,nrow = kernelWidth , ncol = kernelWidth )
    for(i in 1:ncol(ConvolveInit)){
      for(j in 1:nrow(ConvolveInit)){
        if(abs(i - j) < DIAG_PARAM){  ConvolveInit[i,j] <- 1 }
      } }
    ConvolveInit <- ConvolveInit[nrow(ConvolveInit):1,]
    ProcessConvFxn <- function(conv){
      #conv <- ( conv/sum(conv) )
      SCALING_FACTOR <- sqrt(0.001 + var(c(conv)) * length(conv))
      if(is.na(SCALING_FACTOR)){SCALING_FACTOR<-1}
      conv <- (conv-mean(conv)) / SCALING_FACTOR
    }
    ConvolveInit <- ProcessConvFxn( ConvolveInit )
    tmp_ <- replicate(1000,sum(rnorm(length(ConvolveInit))*c(ConvolveInit)))
    mean(tmp_); sd(tmp_)
    ConfoundingConvTensor <- tf$expand_dims(tf$expand_dims(tf$cast(ConvolveInit,dtype=tf$float32),2L), 3L)
    ConfoundingConv$kernel$assign(ConfoundingConvTensor)

    # fix convolution kernel for moderation
    ResponseConvolveInit <- ConvolveInit;ResponseConvolveInit[] <- -1
    tmp_ <- 0.1; ResponseConvolveInit[round(nrow(ResponseConvolveInit)*(0.5-tmp_)):
                                        round(nrow(ResponseConvolveInit)*(0.5+tmp_)),
                                      round(ncol(ResponseConvolveInit)*(0.5-tmp_)):
                                        round(ncol(ResponseConvolveInit)*(0.5+tmp_))] <- 1
    ResponseConvolveInit <- -1*ResponseConvolveInit
    ResponseConvTensor <- tf$expand_dims(tf$expand_dims(tf$cast(ResponseConvolveInit,dtype=tf$float32),2L), 3L)
    ResponseConv$kernel$assign(ResponseConvTensor)

    # visualize convolution kernels
    pdf("./Figures/ConfoundingConvolutionKernel.pdf")
    par(mfrow=c(1,1));image2(as.matrix( ConfoundingConvTensor[,,1,1] ),
                             #main = sprintf("%s by %s Confounding Kernel", kernelWidth, kernelWidth),
                             main = "", cex.main = 2.5,box=T)
    dev.off()

    pdf("./Figures/ResponseConvolutionKernel.pdf")
    par(mfrow=c(1,1));image2(as.matrix( ResponseConvTensor[,,1,1] ),
                             #main = sprintf("%s by %s Moderating Kernel", kernelWidth, kernelWidth),
                             cex.main = 2.5,
                             box=T)
    dev.off()

    # perform the new convolution
    ConfoundingConvolvedImage <- ConfoundingConv(  ImageBlocks_tf )
    ResponseConvolvedImage <- ResponseConv(  ImageBlocks_tf )

    # convert back to R for analysis
    takeDims <- imageHeight - kernelWidth + 1
    takeIndices <- (kernelWidth/2):(imageHeight-kernelWidth/2)
    #RawImage <- as.array(  ImageBlocks_tf[,1,takeIndices,takeIndices,1] )
    UnobservedConfounder <- as.array(  ConfoundingConvolvedImage[,1,,,1] )
    UnobservedConfounder_mean <- mean( UnobservedConfounder ); UnobservedConfounder_sd <- sd( UnobservedConfounder )
    UnobservedConfounder <- (UnobservedConfounder - UnobservedConfounder_mean) / UnobservedConfounder_sd

    ResponseModerator <- as.array(  ResponseConvolvedImage[,1,,,1] )
    ResponseModerator_mean <- mean( ResponseModerator ); ResponseModerator_sd <- sd( ResponseModerator )
    ResponseModerator <- (ResponseModerator - ResponseModerator_mean) / ResponseModerator_sd
  }

  # illustrate convolution
  takeDims <- imageHeight - kernelWidth + 1
  takeIndices <- (kernelWidth/2):(imageHeight-kernelWidth/2)
  #pdf("./Figures/ConvolutionIllustrate.pdf",width = 8*2,height = 8)
  {
    im_i <- 1
    par(mfrow = c(1,2));par(mar=c(1,1,5,1))
    m1_raw <- as.matrix(ImageBlocks_tf[im_i,1,takeIndices,takeIndices,1])
    m1_c <- as.matrix(ConfoundingConvolvedImage[im_i,1,,,1])
    image2(m1_raw, main = "Raw Image",cex.main = 4 )
    image2(m1_c , main = "After Convolution" ,cex.main = 4 )
    #image2( (m1_raw - mean(m1_raw))/sd(m1_raw) - (m1_c - mean(m1_c))/sd(m1_c) , main = "Normalized Difference" ,cex.main = 4 )
    #image2( matrix(rank(-m1_c),ncol=ncol(m1_raw),byrow=F) -matrix(rank(-m1_raw),ncol=ncol(m1_raw),byrow=F) , main = "Normalized Difference" ,cex.main = 4 )
  }
  #dev.off()

  for(analysisType in rev(c("scene"))){
    #for(analysisType in rev(c("sceneBigModel"))){

    # sim + optimization parameters
    #nSGD <- 300;
    nSGD <- 300;
    if(grepl(analysisType,pattern="scene")){nMonte <- 20}
    if(analysisType == "pixel"){nMonte <- 10}
    if(analysisType == "robustness"){nMonte <- 1}
    LEARNING_RATE_BASE <- 0.01; widthCycle <- 25
    ep_sd <- 0.1; wtInYForU <- 1; wtInWForU <- 1
    batchSize <- 30; rm ( treatNumb )
    name_key <- analysisType
    kernelWidth_true_seq <- outer_seq <- kernelWidth_master_seq <- (as.integer(2^(1:5)))
    if(analysisType == "robustness"){ nMonte <- 1L; kernelWidth_true_seq <- outer_seq <- as.integer(2^3) }

    # perform Monte Carlo experiment
    convolution_list <- rep(rep(times=length(outer_seq),list()), times=nMonte)
    tauHat_mat <- tauHat_propensity_true_mat <- tauHat_diffInMeans_mat <- tauHat_propensityHajek_mat <- tauHat_propensity_mat <- c()
    pval_propensityHajek_mat <- pval_propensity_mat <- c()
    outer_counter <- 0; for(outer_ in outer_seq){
      outer_counter <- outer_counter + 1
      print(sprintf("Outer %s", outer_))
      tauHat_propensityHajek <- tauHat_propensity <- tauHat_diffInMeans <- rep(NA,times=nMonte)
      tauHat_propensity_true <- pval_propensityHajek <- pval_propensity <- tauHat_propensityHajek

      kernelWidth_est <- as.integer( 2^3 )
      #kernelWidth_true <- kernelWidth_est
      kernelWidth_true <- outer_
      for(sim_i in 1:nMonte){
        # generate confounder
        {
          #GlobalPoolType <- "ave";GlobalPoolLayer <- keras$layers$GlobalAveragePooling2D(data_format="channels_last",name="GlobalAve")
          GlobalPoolType <- "max";GlobalPoolLayer <- keras$layers$GlobalMaxPool2D(data_format="channels_last",name="GlobalMax")
          ConfoundingConv <- keras$layers$Conv2D(filters=1L,
                                                 kernel_size=c(kernelWidth_true,kernelWidth_true),
                                                 activation="linear",
                                                 padding = 'same')

          # initialize kernel
          ConfoundingConv(  ImageBlocks_tf )

          # fix convolution kernel for confounding
          ConvolveInit <- matrix(0,nrow = kernelWidth_true , ncol = kernelWidth_true )
          for(i in 1:ncol(ConvolveInit)){
            for(j in 1:nrow(ConvolveInit)){
              if(abs(i - j) < DIAG_PARAM){  ConvolveInit[i,j] <- 1 }
            } }
          ConvolveInit <- ConvolveInit[nrow(ConvolveInit):1,]
          if(kernelWidth_true == 1){ConvolveInit <- as.matrix(ConvolveInit)}
          if(kernelWidth_true > 1){
            ConvolveInit <- ProcessConvFxn(ConvolveInit)
          }
          ConfoundingConvTensor <- tf$expand_dims(tf$expand_dims(tf$cast(ConvolveInit,dtype=tf$float32),2L), 3L)
          ConfoundingConv$kernel$assign(ConfoundingConvTensor)
          ConfoundingConvolvedImage <- ConfoundingConv(  ImageBlocks_tf )

          # convert back to R for analysis
          takeDims <- imageHeight - kernelWidth_true + 1
          takeIndices <- (kernelWidth_true/2):(imageHeight-kernelWidth_true/2)
          #RawImage <- as.array(  ImageBlocks_tf[,1,takeIndices,takeIndices,1] )
          UnobservedConfounder <- as.array(  ConfoundingConvolvedImage[,1,,,1] )
          UnobservedConfounder_mean <- mean( UnobservedConfounder ); UnobservedConfounder_sd <- sd( UnobservedConfounder )
          UnobservedConfounder <- (UnobservedConfounder - UnobservedConfounder_mean) / UnobservedConfounder_sd
        }

        # fix dimensions
        kernelWidth_max <- max(max(kernelWidth_true_seq),kernelWidth_est)
        takeIndices_withMax_1Indexed <- (kernelWidth_max/2):(imageHeight-kernelWidth_max/2)
        takeIndices_withMax_0Indexed <- tf$constant(takeIndices_withMax_1Indexed - 1L,dtype=tf$int32)

        # generate treatment data
        {
          if(analysisType=="pixel"){
            trueProbW <- sapply(1:nrow(UnobservedConfounder),function(zer){
              UnObsConf_ <- UnobservedConfounder[zer,,]
              ep_ <- rnorm(length(UnObsConf_),sd=ep_sd)
              WProb <- UnObsConf_;
              WProb[] <- 1/(1+exp(-(wtInWForU * UnObsConf_ + ep_ )))
              list(tf$constant(WProb,dtype=tf$float32))
            })
            trueProbW <- as.array(  tf$stack(trueProbW,0L) )
            trueProbW <- trueProbW[,takeIndices_withMax_1Indexed,takeIndices_withMax_1Indexed]
            trueProbW_tf <- tf$constant(trueProbW,dtype=tf$float32)
          }
          if(analysisType=="scene"){
            UnobservedConfounder <- sapply(1:nrow(UnobservedConfounder),function(zer){
              UnObsConf_ <- UnobservedConfounder[zer,takeIndices_withMax_1Indexed,takeIndices_withMax_1Indexed]
              if(GlobalPoolType == "max"){UnObsConf_ <- max(UnObsConf_) }
              if(GlobalPoolType == "ave"){UnObsConf_ <- mean(UnObsConf_) }
              UnObsConf_
            })
            UnobservedConfounder <- c(scale(UnobservedConfounder))
            trueProbW <- sapply(1:length(UnobservedConfounder),function(zer){
              UnObsConf_ <- UnobservedConfounder[zer]
              if(GlobalPoolType == "max"){UnObsConf_ <- max(UnObsConf_) }
              if(GlobalPoolType == "ave"){UnObsConf_ <- mean(UnObsConf_) }
              ep_ <- rnorm(length(UnObsConf_),sd=ep_sd)
              WProb <- UnObsConf_;
              WProb[] <- 1/(1+exp(-  (wtInWForU * UnObsConf_ + ep_ )))
              list(tf$constant(WProb,dtype=tf$float32))
            })
            trueProbW <- as.array(  tf$stack(trueProbW,0L) )
            #trueProbW <- trueProbW[,takeIndices_withMax_1Indexed,takeIndices_withMax_1Indexed]
            trueProbW_tf <- tf$constant(trueProbW,dtype=tf$float32)
            #dim_orig <- dim( trueProbW )
            #trueProbW <- as.array(trueProbW_tf <- GlobalPoolLayer(tf$expand_dims(trueProbW_tf,3L)))
          }
          obsW <- trueProbW
          obsW[] <- rbinom(length(trueProbW),size=1, prob = c(trueProbW))
          obsW <- as.array( obW_tf <- tf$constant(obsW,tf$float32) )

          # generate outcome data
          obsY0 <- sapply(1:nImages,function(zer){
            if(analysisType == "pixel"){ UnObsConf_ <- UnobservedConfounder[zer,takeIndices_withMax_1Indexed,takeIndices_withMax_1Indexed]}
            if(analysisType == "scene"){ UnObsConf_ <- UnobservedConfounder[zer]}
            Yobs <- wtInYForU*UnObsConf_ + rnorm(length(UnObsConf_),sd=ep_sd)
            return( list( Yobs ) )
          })
          obsY <- sapply(1:nImages,function(zer){
            if(analysisType == "pixel"){ W__ <- obsW[zer,,]; UnObsConf_ <- UnobservedConfounder[zer,takeIndices_withMax_1Indexed,takeIndices_withMax_1Indexed]}
            if(analysisType == "scene"){ W__ <- obsW[zer]; UnObsConf_ <- UnobservedConfounder[zer]}
            Yobs <- wtInYForU*UnObsConf_  + trueTau * W__ + rnorm(length(UnObsConf_),sd=ep_sd)
            return( list( Yobs ) )
          })
          obsY <- as.array( obsY_tf <- tf$stack(obsY,0L) )

          # flatten and estimate ATE
          obsY_flat <- c(obsY <- obsY)
          obsW_flat <- c(obsW <- obsW)
          trueProbW_flat <- c(trueProbW <- trueProbW)
          tauHat_diffInMeans[sim_i] <- mean(obsY_flat[ obsW_flat == 1],na.rm=T) - mean(obsY_flat[ obsW_flat == 0],na.rm=T)
        }

        # tf model
        testIndices <- trainIndices <- 1:nrow(ImageBlocks_tf)
        kernelWidth_est_seq <-  kernelWidth_est
        if(analysisType == "robustness"){
          kernelWidth_est_seq <-  kernelWidth_master_seq
          trainIndices <- sort(sample( 1:nrow(ImageBlocks_tf),nrow(ImageBlocks_tf)*0.75))
          testIndices <- (1:nrow(ImageBlocks_tf))[! 1:nrow(ImageBlocks_tf) %in% trainIndices]
        }
        out_loss <- c()
        for(kernelWidth_est_USE in kernelWidth_est_seq){

          # tf model initialization
          BNLayer_Axis3 <- keras$layers$BatchNormalization(axis = 3L, center = T, scale = T, momentum = 0.9, epsilon = 0.001)
          BNLayer_Axis1 <- keras$layers$BatchNormalization(axis = 1L, center = T, scale = T, momentum = 0.9, epsilon = 0.001)
          #Conv_Mean = keras$layers$Conv2D(filters=1L, kernel_initializer  = tf$ones, trainable = F, bias_initializer = tf$zeros, activation = 'linear', kernel_size=as.integer(c(kernelWidth_est_USE,kernelWidth_est_USE)),padding = 'same')
          #getMeanConv <- tf_function( function(dar){ 1./kernelWidth_est_USE^2*tf$squeeze(Conv_Mean( tf$expand_dims(dar,3L) ),3L)} )
          PropensityConv <- keras$layers$Conv2D(filters=nFilters<-1L,
                                                kernel_size=c(kernelWidth_est_USE,kernelWidth_est_USE),
                                                activation="linear",
                                                padding = 'same')
          ConvAttnProj = keras$layers$Dense(1L, activation='linear')
          getTreatProb <- tf_function( function(datt,training = T){
            # convolution
            if(nFilters >1){RawProj  <- tf$squeeze(ConvAttnProj(PropensityConv( tf$expand_dims(datt,3L) )),3L)}
            if(nFilters==1){RawProj  <- tf$squeeze(PropensityConv( tf$expand_dims(datt,3L) ),3L)}

            # fix dimensions
            #if(analysisType == "pixel"){
            RawProj <- tf$gather(RawProj,indices = takeIndices_withMax_0Indexed, axis = 1L)
            RawProj <- tf$gather(RawProj,indices = takeIndices_withMax_0Indexed, axis = 2L)
            #}

            # obtain probability
            RawProj_n <- tf$squeeze(BNLayer_Axis3(tf$expand_dims(RawProj,3L), training = training),3L)
            if(analysisType=="scene"){
              RawProj_n <- BNLayer_Axis1(GlobalPoolLayer(tf$expand_dims(RawProj_n,3L)),training = training)

              # broadcast
              #RawProj_n <- tf$expand_dims(RawProj_n,2L)+RawProj*0
            }
            TreatProb <- tf$keras$activations$sigmoid( RawProj_n )

            # return
            return( TreatProb )
          })
          getLoss <- tf_function( function(datt,treatt){
            treatProb <- getTreatProb( datt )
            #hist(c(as.array( treatProb )))
            #minThis <- tf$reduce_mean( tf$abs(treatt - treatProb ) )
            treatt_r <- tf$cast(tf$reshape(treatt,list(-1L,1L)),dtype=tf$float32)
            treatProb_r <- tf$reshape(treatProb,list(-1L,1L))
            #minThis <-   -  tf$reduce_mean( tf$math$log(treatProb_r)*(treatt_r) + tf$math$log(1-treatProb_r)*(1-treatt_r) )
            minThis <- tf$reduce_mean( tf$keras$losses$binary_crossentropy(treatt_r, treatProb_r))
            #minThis <- tf$reduce_mean(tf$square(tf$subtract(treatt_r,treatProb_r)))
            return( minThis )
          })

          with(tf$GradientTape() %as% tape, {
            myLoss_forGrad <- getLoss( datt = as.array(ImageBlocks_tf)[,1,,,],
                                       treatt = obsW )  })
          trainable_variables <- c(  PropensityConv$trainable_variables ,
                                     ConvAttnProj$trainable_variables,
                                     BNLayer_Axis1$trainable_variables,
                                     BNLayer_Axis3$trainable_variables )

          # define optimizer and training step
          optimizer_tf = tf$optimizers$Nadam(clipnorm=10.)
          trainStep <-  (function(dat, truth){
            with(tf$GradientTape() %as% tape, {
              myLoss_forGrad <<- getLoss( datt = tf$constant(dat,tf$float32),
                                          treatt = tf$constant(truth,tf$float32))   })
            my_grads <<- tape$gradient( myLoss_forGrad, trainable_variables )
            optimizer_tf$learning_rate$assign(   LEARNING_RATE_BASE*abs(cos(i/nSGD*widthCycle)  )*(i<=nSGD/2)+
                                                   LEARNING_RATE_BASE*(i>nSGD/2)/(0.001+abs(i-nSGD/2)^0.2 ))
            optimizer_tf$apply_gradients( rzip(my_grads, trainable_variables)[!unlist(lapply(my_grads,is.null)) ])
          })

          # perform training
          loss_vec <- rep(NA,times=nSGD)
          in_ <- ip_ <- 0; for(i in 1:nSGD){
            if(i%%50==0){print(i);par(mfrow = c(1,1));plot(loss_vec)}
            batch_indices <- sample(trainIndices,batchSize,replace=F)
            if(analysisType == "pixel"){W_ <- obsW[batch_indices,,]}
            if(analysisType == "scene"){W_ <- as.matrix(obsW[batch_indices])}
            trainStep(dat = as.array(ImageBlocks_tf)[batch_indices,1,,,],
                      truth = W_)
            loss_vec[i] <- myLoss_forGrad <- as.numeric( myLoss_forGrad )
            if(is.na(myLoss_forGrad)){print("NA in LOSS");browser()}
          }

          AllTreatProb <- as.array( getTreatProb( as.array(ImageBlocks_tf)[,1,,,],training = F ) )
          AllWobs <- as.array((obsW))
          AllYobs <- as.array((obsY))
          # print( cor(c(AllWobs[1:1e4]),c(AllTreatProb[1:1e4])) )

          # test loss
          if(analysisType == "pixel"){W_ <- obsW[testIndices,,]}
          if(analysisType == "scene"){W_ <- as.matrix(obsW[testIndices])}
          trainStep(dat = as.array(ImageBlocks_tf)[testIndices,1,,,],
                    truth = W_)
          out_loss <- c(out_loss,as.numeric( myLoss_forGrad))
        }
      }

      # append data
      tauHat_diffInMeans_mat<- cbind(tauHat_diffInMeans_mat,tauHat_diffInMeans)
      tauHat_propensity_mat <- cbind(tauHat_propensity_mat,tauHat_propensity)
      tauHat_propensity_true_mat <- cbind(tauHat_propensity_true_mat,tauHat_propensity_true)
      tauHat_propensityHajek_mat <- cbind(tauHat_propensityHajek_mat,tauHat_propensityHajek)
    }
  }
}
