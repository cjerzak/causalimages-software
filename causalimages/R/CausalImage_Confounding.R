#!/usr/bin/env Rscript
#' Perform causal estimation under image confounding
#'
#' *Under beta release. Full release in Spring of 2023.*
#'
#' @usage
#'
#' AnalyzeImageConfounding(obsW, obsY, acquireImageFxn, ...)
#'
#' @param obsW A numeric vector where `0`'s correspond to control units and `1`'s to treated units.
#' @param obsY A numeric vector containing observed outcomes.
#' @param acquireImageRepFxn A function specifying how to load images representations associated with `imageKeysOfUnits` into memory. For example, if observation `3` has a value  of `"a34f"` in `imageKeysOfUnits`, `acquireImageFxn` should extract the image associated with the unique key `"a34f"`.
#' First argument should be image key values and second argument have be `training` (in case behavior in training/)
#' @param acquireImageFxn (default = `acquireImageRepFxn`) Similar to `acquireImageRepFxn`; this is a function specifying how to load images associated with `imageKeysOfUnits` into memory.
#' @param transportabilityMat (optional) A matrix with a column named `keys` specifying keys to be used by `acquireImageRepFxn` for generating treatment effect predictions for out-of-sample points.
#' @param imageKeysOfUnits (default = `1:length(obsY)`) A vector of length `length(obsY)` specifying the unique image ID associated with each unit. Samples of `imageKeysOfUnits` are fed into `acquireImageFxn` to call images into memory.
#' @param long,lat (optional) Vectors specifying longitude and latitude coordinates for units. Used only for describing highest and lowest probability neighorhood units if specified.
#' @param X (optional) A numeric matrix containing tabular information used if `orthogonalize = T`.
#' @param conda_env (default = `NULL`) A string specifying a conda environment wherein `tensorflow`, `tensorflow_probability`, and `gc` are installed.
#' @param conda_env_required (default = `F`) A Boolean stating whether use of the specified conda environment is required.
#' @param orthogonalize (default = `F`) A Boolean specifying whether to perform the image decomposition after orthogonalizing with respect to tabular covariates specified in `X`.
#' @param nMonte_variational (default = `5L`) An integer specifying how many Monte Carlo iterations to use in the
#' calculation of the expected likelihood in each training step.
#' @param nMonte_predictive (default = `20L`) An integer specifying how many Monte Carlo iterations to use in the calculation
#' of posterior means (e.g., mean cluster probabilities).
#' @param nMonte_salience (default = `100L`) An integer specifying how many Monte Carlo iterations to use in the calculation
#' of the salience maps (e.g., image gradients of expected cluster probabilities).
#' @param reparameterizationType (default = `"Deterministic"`) Currently, only deterministic layers are used. Future releases will add the option to make the CNN model arms probabilistic.
#' @param figuresTag (default = `""`) A string specifying an identifier that is appended to all figure names.
#' @param figuresPath (default = `"./"`) A string specifying file path for saved figures made in the analysis.
#' @param plotBand (default = `1L`) An integer specifying which band position (from the acquired image representation) should be plotted in the visual results.
#' @param kernelSize (default = `5L`) Dimensions used in convolution kernels.
#' @param nSGD (default = `400L`) Number of stochastic gradient descent (SGD) iterations.
#' @param batchSize (default = `25L`) Batch size used in SGD optimization.
#' @param doConvLowerDimProj (default = `T`) Should we project the `nFilters` convolutional feature dimensions down to `nDimLowerDimConv` to reduce the number of required parameters.
#' @param nDimLowerDimConv (default = `3L`) If `doConvLowerDimProj = T`, then, in each convolutional layer, we project the `nFilters` feature dimensions down to `nDimLowerDimConv` to reduce the number of parameters needed.
#' @param nFilters (default = `32L`) Integer specifying the number of convolutional filters used.
#' @param nDenseWidth (default = `32L`) Width of dense projection layers post-convolutions.
#' @param nDepthHidden_conv (default = `3L`) Hidden depth of convolutional layer.
#' @param nDepthHidden_dense (default = `0L`) Hidden depth of dense layers. Default of `0L` means a single projection layer is performed after the convolutional layer (i.e., no hidden layers are used).
#' @param quiet (default = `F`) Should we suppress information about progress?
#' @param maxPoolSize (default = `2L`) Integer specifying the max pooling size used in the convolutional layers.
#' @param strides (default = `2L`) Integer specifying the strides used in the convolutional layers.=
#' @param simMode (default = `F`) Should the analysis be performed in comparison with ground truth from simulation?
#' @param tf_seed (default = `NULL`) Specification for the tensorflow seed.
#' @param plotResults (default = `T`) Should analysis results be plotted?
#' @param channelNormalize (default = `T`) Should channelwise image feature normalization be attempted? Default is `T`, as this improves training.
#'
#' @return A list consiting of \itemize{
#'   \item `ATE_est` ATE estimate.
#'   \item `ATE_se` Standard error estimate for the ATE.
#'   \item `(images saved to disk if plotResults = T)` If `plotResults = T`, causal salience plots are saved to disk characterizing the image confounding structure. See references for details.
#' }
#'
#' @section References:
#' \itemize{
#' \item  Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Integrating Earth Observation Data into Causal Inference: Challenges and Opportunities. *ArXiv Preprint*, 2023.
#' }
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/causalimages-software/
#'
#' @export
#' @md

AnalyzeImageConfounding <- function(
                                   obsW,
                                   obsY,
                                   X = NULL,
                                   file = NULL,
                                   keys = NULL,
                                   nDepth = 3L,
                                   doConvLowerDimProj = T,
                                   nDimLowerDimConv = 3L,
                                   nFilters = 32L,
                                   samplingType = "none",
                                   doHiddenDim = T,
                                   HiddenDim  = 32L,
                                   DenseActivation = "linear",
                                   input_ave_pooling_size = 1L, # if seeking to downshift the resolution
                                   useTrainingPertubations = T,

                                   orthogonalize = F,
                                   imageKeysOfUnits = 1:length(obsY),
                                   acquireImageRepFxn = NULL ,
                                   acquireImageFxn = NULL ,
                                   transportabilityMat = NULL ,
                                   lat = NULL,
                                   long = NULL,
                                   conda_env = NULL,
                                   conda_env_required = F,

                                   figuresTag = "",
                                   figuresPath = "./",
                                   plotBand = 1L,

                                   simMode = F,
                                   plotResults = T,

                                   nDepthHidden_conv = 1L,
                                   nDepthHidden_dense = 0L,
                                   maxPoolSize = 2L,
                                   strides = 1L,
                                   compile = T,
                                   nMonte_variational = 5L,
                                   nMonte_predictive = 20L,
                                   nMonte_salience = 100L,
                                   batchSize = 25L,
                                   kernelSize = 3L,
                                   nSGD  = 400L,
                                   nDenseWidth = 32L,
                                   reparameterizationType = "Deterministic",
                                   channelNormalize = T,
                                   printDiagnostics = F,
                                   tf_seed = NULL,
                                   quiet = F){


  print("Initializing the tensorflow environment...")
  print("Looking for Python modules tensorflow, tensorflow_probability, gc...")
  {
    library(tensorflow); library(keras)
    try(tensorflow::use_condaenv(conda_env, required = conda_env_required),T)
    Sys.sleep(1.); try(tf$square(1.),T); Sys.sleep(1.)
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]],T),T)
    try( tf$config$set_soft_device_placement( T ) , T)
    tfp <- tf_probability()
    tfd <- tfp$distributions
    #tfa <- reticulate::import("tensorflow_addons")

    try(tf$random$set_seed(  c( ifelse(is.null(tf_seed),
                                yes = 123431L, no = as.integer(tf_seed)  ) )), T)
    try(tf$keras$utils$set_random_seed( c( ifelse(is.null(tf_seed),
                                yes = 123419L, no = as.integer(tf_seed)  ) )), T)

    # import python garbage collectors
    py_gc <- reticulate::import("gc")
    gc(); py_gc$collect()
  }

  {
    acquireImageMethod <- "functional";
    # define base tf record + train/test fxns
    if(  !is.null(  file  )  ){
      acquireImageMethod <- "tf_record"

      # established tfrecord connection
      orig_wd <- getwd()
      tf_record_name <- file
      tf_record_name <- strsplit(tf_record_name,split="/")[[1]]
      new_wd <- paste(tf_record_name[-length(tf_record_name)],collapse = "/")
      setwd( new_wd )
      tf_dataset = tf$data$TFRecordDataset(  tf_record_name[length(tf_record_name)] )

      # helper functions
      getParsed_tf_dataset_inference <- function(tf_dataset){
        dataset <- tf_dataset$map( parse_tfr_element ) # return
        return( dataset <- dataset$batch( as.integer(max(2L,round(batchSize/2L)  ))) )
      }

      getParsed_tf_dataset_train <- function(tf_dataset){
        dataset <- tf_dataset$map( parse_tfr_element )
        dataset <- dataset$shuffle(tf$constant(as.integer(10*batchSize),dtype=tf$int64),
                                   reshuffle_each_iteration = T)
        dataset <- dataset$batch(as.integer(batchSize))
      }

      # setup iterators
      tf_dataset_train <- getParsed_tf_dataset_train( tf_dataset )
      tf_dataset_inference <- getParsed_tf_dataset_inference( tf_dataset )

      # reset iterators
      ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
      ds_iterator_inference <- reticulate::as_iterator( tf_dataset_inference )

      # checks
      # ds_iterator_inference$output_shapes; ds_iterator_train$output_shapes
      # ds_next_train <- reticulate::iter_next( ds_iterator_train )
      # ds_next_inference <- reticulate::iter_next( ds_iterator_inference )
      setwd(  orig_wd  )
    }

    trainingPertubations <- tf$identity
    if(useTrainingPertubations){
      trainingPertubations <- (function(im__){
        im__ <- tf$image$random_flip_left_right(im__)
        im__ <- tf$image$random_flip_up_down(im__)
        return( im__ )
      })
    }

    binaryCrossLoss <- function(W,prW){return( - mean( log(prW)*W + log(1-prW)*(1-W) ) ) }

    InitImageProcess <- function(im, training = F, input_ave_pooling_size = 1){

      # expand dims if needed
      if(length(keys) == 1){ im <- tf$expand_dims(im,0L) }

      # normalize
      im <- (im - NORM_MEAN_array) / NORM_SD_array

      # training pertubations if desired
      # note: trainingPertubations mus be performed on CPU
      if(training == T){ im <- trainingPertubations(im) }

      # downshift resolution if desired
      if(input_ave_pooling_size > 1){ im <- AvePoolingDownshift(im) }
      return( im  )
    }

    # some hyperparameters parameters
    figuresPath <- paste(strsplit(figuresPath,split="/")[[1]],collapse = "/")
    KernalActivation <- "swish"
    KernalProjActivation <- "swish"
    HiddenActivation <- "swish"
    BN_MOMENTUM <- 0.90
    poolingAt <- 1L # do pooling every poolingAt iterations
    poolingBy <- 2L # pool by poolingBy by poolingBy
    poolingType <- "max"
    LEARNING_RATE_BASE <- 0.005; widthCycle <- 50
    doParallel <- F
    testIndices <- trainIndices <- 1:length(obsY)

    # initialize layers
    AvePoolingDownshift <- tf$keras$layers$AveragePooling2D(pool_size = as.integer(c(input_ave_pooling_size,input_ave_pooling_size)))
    try(eval(parse(text = paste("rm(", paste(trainable_layers,collapse=","),")"))),T)
    trainable_layers <- ls()
    {
      GlobalMaxPoolLayer <- tf$keras$layers$GlobalMaxPool2D(data_format="channels_last",name="GlobalMax")
      GlobalAvePoolLayer <- tf$keras$layers$GlobalAveragePooling2D(data_format="channels_last",name="GlobalAve")
      GlobalPoolLayer <- function(z){
        return(tf$concat(list(GlobalMaxPoolLayer(z),GlobalAvePoolLayer(z)),1L)) }
      BNLayer_Axis1_dense <- tf$keras$layers$BatchNormalization(axis = 1L, center = F, scale = F, momentum = BN_MOMENTUM, epsilon = 0.001)
      BNLayer_Axis1_hidden <- tf$keras$layers$BatchNormalization(axis = 1L, center = F, scale = F, momentum = BN_MOMENTUM, epsilon = 0.001)
      BNLayer_Axis1_final <- tf$keras$layers$BatchNormalization(axis = 1L, center = T, scale = T, momentum = BN_MOMENTUM, epsilon = 0.001)
      HiddenLayer <- tf$keras$layers$Dense(HiddenDim, activation = HiddenActivation)
      DenseLayer <- tf$keras$layers$Dense(1L, activation = DenseActivation)
      FlattenLayer <- tf$keras$layers$Flatten(data_format = "channels_last")

      #getMeanConv <- tf_function( function(dar){ 1./kernelSize^2*tf$squeeze(Conv_Mean( tf$expand_dims(dar,3L) ),3L)} )
      BNLayer_Axis3_init <- tf$keras$layers$BatchNormalization(axis = 3L, center = F, scale = F, momentum = BN_MOMENTUM, epsilon = 0.001,name="InitNorm")
      for(d_ in 1:nDepth){
        eval(parse(text = sprintf('Conv%s <- tf$keras$layers$Conv2D(filters=nFilters,
                                  kernel_size=c(kernelSize,kernelSize),
                                  activation=KernalActivation,
                                  strides = c(strides,strides),
                                  padding = "valid")',d_)))
        eval(parse(text = sprintf('ConvProj%s = tf$keras$layers$Dense(nDimLowerDimConv,
                                      activation=KernalProjActivation)',d_)))
        eval(parse(text = sprintf('BNLayer_Axis3_%s <- tf$keras$layers$BatchNormalization(axis = 3L, center = T, scale = T, momentum = BN_MOMENTUM, epsilon = 0.001)',d_)))
        eval(parse(text = sprintf('BNLayer_Axis3_%s_inner <- tf$keras$layers$BatchNormalization(axis = 3L, center = T, scale = T, momentum = BN_MOMENTUM, epsilon = 0.001)',d_)))

        # every third, do pooling
        if(d_ %% poolingAt == 0){
          Pool = tf$keras$layers$MaxPool2D(pool_size = c(poolingBy,poolingBy))
          if(poolingType=="ave"){ Pool = tf$keras$layers$AveragePooling2D(pool_size = c(poolingBy,poolingBy)) }
        }
        if(d_ %% poolingAt != 0){ eval(parse(text = sprintf('Pool%s = tf$identity',d_))) }
      }
    }
    trainable_layers <- ls()[!ls() %in% c(trainable_layers)]
    getProcessedImage <- tf_function( function(imm,training){
      # initial normalization - DEPRECIATIED, imm is now normalized elsewhere
      #imm <- BNLayer_Axis3_init( imm , training = training )

      # convolution + pooling
      for(d_ in 1:nDepth){
        if(doConvLowerDimProj){
          eval(parse(text = sprintf("imm <- BNLayer_Axis3_%s_inner(Pool( Conv%s( imm )),training=training)",d_,d_)))
          if(d_ < nDepth){ eval(parse(text = sprintf("imm <- BNLayer_Axis3_%s(  ConvProj%s( imm ), training = training)",d_,d_))) }
        }
        if(doConvLowerDimProj == F){ eval(parse(text = sprintf("imm <- BNLayer_Axis3_%s( Pool( Conv%s( imm )), training = training)",d_,d_,d_)))}
      }
      return(imm)
    })

    getTreatProb <- tf_function( function(im_getProb,x_getProb, training_getProb){

      # flatten
      im_getProb <- GlobalPoolLayer( getProcessedImage(im_getProb,training = training_getProb) )
      im_getProb <- BNLayer_Axis1_dense(im_getProb, training = training_getProb)

      # concatinate with scene-level data
      im_getProb <- tf$concat(list(im_getProb,x_getProb),1L)

      # optimal hidden layer
      if(doHiddenDim == T){
        im_getProb <- HiddenLayer(  im_getProb   )
      }

      # final projection layer + sigmoid
      im_getProb <- DenseLayer( im_getProb   )
      im_getProb <- tf$keras$activations$sigmoid( im_getProb )

      # return
      return( im_getProb )
    })
    getLoss <- tf_function( function(im_getLoss, x_getLoss, treatt_getLoss, training_getLoss){
      treatProb <- getTreatProb( im_getProb = im_getLoss,
                                 x_getProb = x_getLoss,
                                 training_getProb = training_getLoss)
      treatt_r <- tf$cast(tf$reshape(treatt_getLoss,list(-1L,1L)),dtype=tf$float32)
      treatProb_r <- tf$reshape(treatProb,list(-1L,1L)) # check
      minThis <-   -tf$reduce_mean( tf$multiply(tf$math$log(treatProb_r),(treatt_r)) +
                                      tf$multiply(tf$math$log(1-treatProb_r),(1-treatt_r)) )
      return( minThis )
    })

    # get first iter batch for initializations
    NORM_SD <- NORM_MEAN <- c()
    for(momentCalIter in 1:(momentCalIters<-10)){
      if(acquireImageMethod == "tf_record"){
        ds_next_train <- reticulate::iter_next( ds_iterator_train )
        batch_indices <- as.array(ds_next_train[[2]])
      }
      if(acquireImageMethod == "functional"){
        batch_indices <- sample(1:length(obsY),batchSize,replace = F)
        ds_next_train <- list(
          r2const( acquireImageRepFxn(keys[batch_indices],) , dtype = tf$float32 )
        )
      }

      # setup normalizations
      if(is.null(NORM_MEAN)){
        NORM_MEAN <- NORM_SD <- apply(as.array(ds_next_train[[1]]),4,sd)
        NORM_MEAN[] <- NORM_SD[] <- 0
      }

      # update normalizations
      NORM_SD <- NORM_SD + apply(as.array(ds_next_train[[1]]),4,sd) / momentCalIters
      NORM_MEAN <- NORM_MEAN + apply(as.array(ds_next_train[[1]]),4,mean) / momentCalIters
    }
    NORM_MEAN_array <- tf$constant(array(NORM_MEAN,dim=c(1,1,1,length(NORM_MEAN))),tf$float32)
    NORM_SD_array <- tf$constant(array(NORM_SD,dim=c(1,1,1,length(NORM_SD))),tf$float32)

    # arms
    for(ARM in c(T,F)){
      with(tf$GradientTape() %as% tape, {
        myLoss_forGrad <- getLoss( im_getLoss = InitImageProcess(ds_next_train[[1]],
                                                                 training = T,
                                                                 input_ave_pooling_size = input_ave_pooling_size),
                                   x_getLoss = tf$constant(X[batch_indices,],tf$float32),
                                   treatt_getLoss = tf$constant(as.matrix(obsW[batch_indices]),tf$float32 ),
                                   training_getLoss = ARM )
      })
    }
    trainable_variables <- tape$watched_variables()

    # initialize beta
    init_beta_ref <- c(sapply(init_beta <- seq(-4,4,length.out = 1000),function(zer){ mean( 1/(1+exp(- (rnorm(1000) + zer))) )} ))
    init_beta <- init_beta [ which.min(abs(init_beta_ref - mean(obsW) ) ) ]
    if(samplingType == "initializeBeta"){
      print("INITIALIZING beta");BNLayer_Axis1_final$trainable_variables[[2]]$assign( tf$expand_dims(tf$constant(init_beta,dtype=tf$float32),0L) ) # beta is offset factor
    }

    # define optimizer and training step
    NA20 <- function(zer){zer[is.na(zer)] <- 0;zer[is.infinite(zer)] <- 0;zer}
    optimizer_tf = tf$optimizers$legacy$Nadam()
    getGrad <- tf_function(function(im_train, x_train, truth_train){
      with(tf$GradientTape() %as% tape, {
        myLoss_forGrad <- getLoss( im_getLoss = im_train,
                                   x_getLoss = x_train,
                                   treatt_getLoss = truth_train,
                                   training_getLoss = T)
      })
      my_grads <- tape$gradient( myLoss_forGrad, trainable_variables )
      return(list(myLoss_forGrad,my_grads))
    })
    trainStep <- (function(im_train, x_train, truth_train){
      my_grads <- getGrad(im_train, x_train, truth_train)
      myLoss_forGrad <- my_grads[[1]]
      my_grads <- my_grads[[2]]
      optimizer_tf$learning_rate$assign(   tf$constant(LEARNING_RATE_BASE*abs(cos(i/nSGD*widthCycle))*(i<nSGD/2)+
                                                         NA20(LEARNING_RATE_BASE*(i>=nSGD/2)/(i-nSGD/2+1)^.3) ) )
      optimizer_tf$apply_gradients( rzip(my_grads, trainable_variables)[!unlist(lapply(my_grads,is.null)) ])
      return(list(myLoss_forGrad,my_grads))
    })

    # number of trainable variables
    nTrainable <- sum( unlist(  lapply(trainable_variables,function(zer){ prod(dim(zer)) }) ) )
    print(sprintf("%s Trainable Parameters",nTrainable))

    # perform training
    print("Starting training sequence...")
    loss_vec <- rep(NA,times=nSGD)
    in_ <- ip_ <- 0; for(i in 1:nSGD){
      if((i %% 100 == 0 | (i == 10) | i == nSGD) & doParallel == F | i < 50){
        print(sprintf("Iteration: %i",i) );
        try(par(mfrow = c(1,1)),T);try(plot(loss_vec),T); try(points(smooth.spline(na.omit(loss_vec)),type="l",lwd=3),T)
      }
      if(i %% 10 == 0){ py_gc$collect() }
      if((i %% 10 == 0 | i == 1 ) & doParallel == T){
        write.csv(file = sprintf("./checkpoint%s.csv",CommandArg_i), data.frame("CommandArg_i"=CommandArg_i, "i"=i))
      }

      if(acquireImageMethod == "functional"){
        if(samplingType != "balancedTrain"){
          batch_indices <- sample(trainIndices,batchSize,replace=F)
        }
        if(samplingType == "balancedTrain"){
          batch_indices <- c(sample(trainIndices[which(obsW[trainIndices]==1)], batchSize/2),
                             sample(trainIndices[which(obsW[trainIndices]==0)], batchSize/2) )
        }
        ds_next_train <- list(
          r2const( acquireImageRepFxn(keys[batch_indices],) , dtype = tf$float32 )
        )
      }

      if(acquireImageMethod == "tf_record"){
        ds_next_train <- reticulate::iter_next( ds_iterator_train )

        # if we run out of observations, reset iterator...
        if(is.null(ds_next_train)){
          tf$random$set_seed(as.integer(runif(1,1,1000000)))
          tf_dataset_train <- tf_dataset_train$`repeat`()
          ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
          #ds_next_train <- reticulate::iter_next( ds_iterator_train )
        }

        # if we haven't run out of observations, set up data
        if(!is.null(ds_next_train)){
          if(length(as.array(ds_next_train[[2]])) < batchSize){
            tf_dataset_train <- getParsed_tf_dataset_train( tf_dataset )
            ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
          }
        }
        batch_indices <- c(as.array(ds_next_train[[2]]))
      }

      myLoss_forGrad <- trainStep(
        im_train = InitImageProcess(ds_next_train[[1]],
                                    training = T,
                                    input_ave_pooling_size = input_ave_pooling_size),
        x_train = tf$constant(X[batch_indices,],dtype=tf$float32),
        truth_train = tf$constant(as.matrix(obsW[batch_indices]),tf$float32))
      loss_vec[i] <- as.numeric( myLoss_forGrad[[1]] )
    }
    print("Done with training sequence...")

    # remove big objects to free memory for inference
    rm(ds_next_train);rm(myLoss_forGrad)

    # get probabilities for inference
    print("Starting to get probabilities for inference...")
    gc();py_gc$collect()
    prWEst_convnet <- rep(NA,times = length(obsW))
    last_i <- 0; ok_counter <- 0; ok<-F;while(!ok){
      ok_counter <- ok_counter + 1
      print(sprintf("[%s] %.2f%% done with getting inference probabilities", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), 100*last_i / length(obsW)))

      # in functional mode
      if(acquireImageMethod == "functional"){
        batch_indices_inference <- (last_i+1):(last_i+batchSize)
        batch_indices_inference <- batch_indices_inference[batch_indices_inference<=length(obsW)]
        last_i <- batch_indices_inference[length(batch_indices_inference)]
        if(last_i == length(obsW)){ ok <- T }

        batchSizeOneCorrection <- F; if(length(batch_indices_inference) == 1){
          batch_indices_inference <- c(batch_indices_inference,batch_indices_inference)
          batchSizeOneCorrection <- T
        }

        batch_inference <- list(
          r2const( acquireImageRepFxn(keys[batch_indices_inference],) , dtype = tf$float32 )
        )

        insert_probs <- try(c(as.array(getTreatProb(im_getProb = InitImageProcess(batch_inference[[1]],
                                                               input_ave_pooling_size = input_ave_pooling_size),
                                                    x_getProb = tf$constant(X[batch_indices_inference,],dtype=tf$float32),
                                                    training_getProb = F ))),T)
        if(class(insert_probs) == "try-error"){browser()}
        if(batchSizeOneCorrection){ insert_probs <- insert_probs[-1]; batch_indices_inference <- batch_indices_inference[-1] }
        prWEst_convnet[batch_indices_inference] <- insert_probs
      }

      # in tf record mode
      if(acquireImageMethod == "tf_record"){
        batch_inference <- reticulate::iter_next( ds_iterator_inference )
        ok<-T;if(!all(is.null(batch_inference))){
          ok<-F
          batch_indices_inference <- as.array(batch_inference[[2]])
          drop_<-F;if(length(batch_indices_inference)==1){
            drop_ <- T
            batch_indices_inference<-c(batch_indices_inference,batch_indices_inference)
            batch_inference[[1]] <- tf$concat(list(batch_inference[[1]],batch_inference[[1]]),0L)
          }
          insert_probs <- try(c(as.array(getTreatProb(im_getProb = InitImageProcess(batch_inference[[1]],
                                                                                    input_ave_pooling_size = input_ave_pooling_size),
                                                      x_getProb = tf$constant(X[batch_indices_inference,],dtype=tf$float32),
                                                      training_getProb = F ))),T)
          if(drop_ == T){  insert_probs <- insert_probs[-1]  }
          if(class(insert_probs) == "try-error"){browser()}
          prWEst_convnet[batch_indices_inference] <- insert_probs
        }
      }

      gc();py_gc$collect()
    }
    rm( batch_inference )

    # clip extreme estimated probabilities
    prWEst_convnet[prWEst_convnet<0.01] <- 0.01
    prWEst_convnet[prWEst_convnet>0.99] <- 0.99
    if(any(is.na(prWEst_convnet)) ) {browser()}
    print(   cor( c(obsW),c(prWEst_convnet) ) )

    # compute base loss
    prWEst_base <- prWEst_convnet
    prWEst_base[] <- mean(obsW[-testIndices])
    baseLoss_ce_ <- binaryCrossLoss(obsW[testIndices], prWEst_base[testIndices])
    baseLossIN_ce_ <- binaryCrossLoss(obsW[trainIndices], prWEst_base[trainIndices])
    baseLoss_class_ <- 1/length(testIndices) * (sum( prWEst_base[testIndices][ obsW[testIndices] == 1] < 0.5) +
                                                  sum( prWEst_base[testIndices][ obsW[testIndices] == 0] > 0.5))

    outLoss_class_ <- 1/length(testIndices) * (sum( prWEst_convnet[testIndices][ obsW[testIndices] == 1] < 0.5) +
                                                 sum( prWEst_convnet[testIndices][ obsW[testIndices] == 0] > 0.5))
    outLoss_ce_ <-  binaryCrossLoss(  obsW[testIndices], prWEst_convnet[testIndices]  )
    inLoss_ce_ <-  binaryCrossLoss(  obsW[trainIndices], prWEst_convnet[trainIndices]  )

    # do some analysis with examples
    processedDims <- NULL
    if(    plotResults == T  ){
      print("Starting to plot the image confounding results...")
      # get treatment image
      testIndices_t <- testIndices[which(obsW[testIndices]==1)]
      testIndices_c <- testIndices[which(obsW[testIndices]==0)]

      showPerGroup <- min(c(3,unlist(table(obsW))), na.rm = T)
      top_treated <- testIndices_t[indices_top_t <- order( prWEst_convnet[testIndices_t] ,decreasing=T)[1:(showPerGroup*3)]]
      top_control <- testIndices_c[indices_top_c <- order( prWEst_convnet[testIndices_c] ,decreasing=F)[1:(showPerGroup*3)]]

      # drop duplicates
      longLat_test_t <- paste(round(long[testIndices_t],1L),
                              round(lat[testIndices_t],1L),sep="_")
      longLat_test_c <- paste(round(long[testIndices_c],1L),
                              round(lat[testIndices_c],1L),sep="_")
      top_treated <- top_treated[!duplicated(longLat_test_t[indices_top_t])][1:showPerGroup]
      top_control <- top_control[!duplicated(longLat_test_c[indices_top_c])][1:showPerGroup]

      plot_indices <- c(top_control, top_treated)

      makePlots <- function(){

        try({
        pdf(sprintf("%s/CausalSalienceMap_KW%s_AvePool%s_Tag%s.pdf",
                    figuresPath,
                    kernelSize,
                    input_ave_pooling_size,
                    figuresTag),
            width = length(plot_indices)*5+2,height = 3*5)
        {
          layout(matrix(1:(3*(1+length(plot_indices))),
                        ncol = 1+length(plot_indices)),
                 width = c(0.5,rep(5,length(plot_indices))),
                 height = c(5,5,5)); in_counter <- 0
          for(text_ in c("Raw Image","Salience Map","Final Spatial Layer")){
            par(mar=c(0,0,0,0))
            plot(0, main = "", ylab = "",cex=0,
                 xlab = "", ylim = c(0,1), xlim = c(0,1),
                 xaxt = "n",yaxt = "n",bty = "n")
            text(0.5,0.5,labels = text_, srt=90,cex=3)
          }
          for(in_ in plot_indices){
            if(acquireImageMethod == "tf_record"){
              ds_next_in <- GetElementFromTfRecordAtIndex( index = in_,
                                                           filename = file )
              if(length(ds_next_in$shape) == 3){ ds_next_in[[1]] <- tf$expand_dims(ds_next_in[[1]], 0L) }
            }
            if(acquireImageMethod == "functional"){
              ds_next_in <- r2const( acquireImageRepFxn(keys[in_],), dtype = tf$float32 )
              if(length(ds_next_in$shape) == 3){ ds_next_in <- tf$expand_dims(ds_next_in,0L) }
              ds_next_in <- list( ds_next_in )
            }

            print(in_)
            col_ <- ifelse(in_ %in% top_treated,
                           yes = "black", no = "gray")
            in_counter <- in_counter + 1
            long_lat_in_ <- sprintf("Lat, Long: %.3f, %.3f",
                                    lat[in_],long[in_])

            # extract
            im_orig <- im_ <- InitImageProcess(
                                ds_next_in[[1]],
                                input_ave_pooling_size = input_ave_pooling_size)
            XToConcat_values <- tf$constant(t(X[in_,]),tf$float32)
            im_processed <- getProcessedImage(im_, training = F)
            processedDims <- dim(im_)
            im_ <- as.array(tf$squeeze(im_,c(0L)))

            # calculate salience map
            im_orig <- tf$Variable(im_orig,trainable = T)
            with(tf$GradientTape() %as% tape, {
              tape$watch(im_orig)
              treat_prob_im <- tf$squeeze(tf$squeeze(getTreatProb( im_getProb = im_orig,
                                                                   x_getProb = XToConcat_values,
                                                                   training_getProb = F),0L),0L)
            })

            salience_map <- tape$gradient( treat_prob_im, im_orig )
            salience_map <- tf$math$reduce_euclidean_norm(salience_map,3L,keepdims=T)
            salience_map <- tf$keras$layers$AveragePooling2D(c(3L,3L))(salience_map)
            salience_map <- as.array(salience_map)[1,,,]
            salience_map <- apply(salience_map^2,1:2,sum)^0.5

            # do plotting
            orig_scale_im_ <- sapply(1:length(NORM_MEAN),
                                     function(band_){
                                       im_[,,band_] <- 0.1+im_[,,band_]*NORM_SD[band_] + NORM_MEAN[band_]
                                       im_[,,band_] },simplify="array")
            par(mar = (mar_vec <- c(2,1,3,1)))
            orig_scale_im_raster <- raster::brick(orig_scale_im_)

            # plot raw image
            #plot(0, main = long_lat_in_,col.main = col_,
                 #ylab = "", xlab = "", cex.main = 4, ylim = c(0,1), xlim = c(0,1),
                 #cex = 0, xaxt = "n",yaxt = "n",bty = "n")
            #raster::plotRGB(orig_scale_im_raster, r=1,g=2,b=3, add = T, main = long_lat_in_)
            causalimages::image2(
              as.matrix( orig_scale_im_[,,plotBand] ),
              main = long_lat_in_, cex.main = 4, col.main =  col_
            )

            # plot salience map
            par(mar = mar_vec)
            salience_map[salience_map>0] <- salience_map[salience_map>0] / sd(salience_map[salience_map>0])
            print(summary(c(salience_map)))
            salience_map <- sign(salience_map)*log(abs(salience_map)+1)
            print(summary(c(salience_map)))
            causalimages::image2( salience_map )

            # plot final layer
            par(mar = mar_vec)
            causalimages::image2( as.array(im_processed)[1,,,1] )
          }
        }
        dev.off()
        }, T)

        try({
        pdf(sprintf("%s/PropHist_KW%s_AvePool%s_Tag%s.pdf",
                    figuresPath,
                    kernelSize,
                    input_ave_pooling_size,
                    figuresTag))
        {
          par(mfrow=c(1,1))
          d0 <- density(prWEst_convnet[obsW==0])
          d1 <- density(prWEst_convnet[obsW==1])
          plot(d1,lwd=2,xlim = c(0,1),ylim =c(0,max(c(d1$y,d0$y),na.rm=T)*1.2),
               cex.axis = 1.2,ylab = "",xlab = "",
               main = "Density Plots for \n Estimated Pr(T=1 | Confounders)",cex.main = 2)
          points(d0,lwd=2,type = "l",col="gray",lty=2)
          text(d0$x[which.max(d0$y)[1]],
               max(d0$y,na.rm=T)*1.1,label = "W = 0",col="gray",cex=2)
          text(d1$x[which.max(d1$y)[1]],
               max(d1$y,na.rm=T)*1.1,label = "W = 1",col="black",cex=2)
        }
        dev.off()
        }, T)
      }

      if(plotResults){  try(makePlots(),T) }

      preDiff <- colMeans(cbind(long[obsW == 1],lat[obsW == 1])) -
                      colMeans(cbind(long[obsW == 0],lat[obsW == 0]))
      wt1 <- prop.table(1/prWEst_convnet[obsW == 1])
      wt0 <- prop.table(1/(1-prWEst_convnet[obsW == 0]))
      postDiff <- colSums(cbind(long[obsW == 1],lat[obsW == 1])*wt1) -
        colSums(cbind(long[obsW == 0],lat[obsW == 0])*wt0)

      tauHat_propensity = mean(  obsW*obsY/(prWEst_convnet) - (1-obsW)*obsY/(1-prWEst_convnet) )
      tauHat_propensityHajek = sum(  obsY*prop.table(obsW/(prWEst_convnet))) -
        sum(obsY*prop.table((1-obsW)/(1-prWEst_convnet) ))
    }

    print(  "Done with image confounding analysis!"  )
    return(    list(
      "tauHat_propensityHajek"  = tauHat_propensityHajek,
      "tauHat_propensity"  = tauHat_propensity,
      "tauHat_diffInMeans"  = mean(obsY[which(obsW==1)],na.rm=T) - mean(obsY[which(obsW==0)],na.rm=T),
      "outLoss_ce" = outLoss_ce_,
      "out_loss_ce_base" = baseLoss_ce_,
      "inLoss_ce" = inLoss_ce_,
      "outLoss_class" = outLoss_class_,
      "out_loss_class_base" = baseLoss_class_,
      "processedDims" = processedDims,
      "nTrainableParameters" = nTrainable,
      "prWEst_convnet" = prWEst_convnet,
      "input_ave_pooling_size" = input_ave_pooling_size
    ) )
  }
}
