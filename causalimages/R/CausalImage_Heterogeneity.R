#' AnalyzeImageHeterogeneity
#'
#' Implements ...
#'
#' @param DAG 'DAG'.

#' @return A list consiting of \itemize{
#'   \item Items.
#' }
#'
#' @section References:
#' \itemize{
#' \item References here
#' }
#'
#' @examples
#' #set seed
#' set.seed(1)
#'
#' #Geneate data
#' x <- rnorm(100)
#'
#' @export


AnalyzeImageHeterogeneity <- function(obsW,
                                      obsY,
                                      X = NULL,
                                      imageKeys = NULL,
                                      transportabilityMat = NULL,
                                      lat = NULL,
                                      long = NULL,
                                      externalFigureKey = "",
                                      acquireImageRepFxn = acquireImageFxn_full ,
                                      acquireImageFxn_full = NULL ,
                                      acquireImageFxn_transportability = acquireImageFxn_full ,

                                      TYPE = "variational_minimal",
                                      SimMode = F,
                                      nDepth_conv = 1,
                                      nDepth_dense = 1,
                                      plotResults = F,figuresPath = "./",
                                      kClust_est = 2,
                                      maxPoolSize = 2L,
                                      strides = 1L,
                                      y_density = "normal",
                                      orthogonalize = F,
                                      compile = F,
                                      nMonte_predictive = 10L,
                                      nMonte_variational = 5L,
                                      nMonte_salience = 100L,
                                      kernelWidth,
                                      nSGD  = 400,
                                      nDenseWidth = 64L,
                                      nFilters=7L){
  if(T == F){
    library(tensorflow); library(keras)
    try(tensorflow::use_python(python = "/Users/cjerzak/miniforge3/bin/python", required = T),T)
    try(tensorflow::use_condaenv("tensorflow_m1",
                                 required = T, conda = "~/miniforge3/bin/conda"), T)
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]], T),T)
    try(tfp <- tf_probability(),T)
    try(tfd <- tfp$distributions,T)
    print(tf$version$VERSION)

    #tfa <- tensorflow::import("tensorflow_addons",as="tfa")
    try(jax <- tensorflow::import("jax",as="jax"), T)
    #try(jnp <- tensorflow::import("jax.numpy"), T)
    # try(tf2jax <- tensorflow::import("tf2jax",as="tf2jax"), T)
  }

  # set environment of image sampling functions
  environment(acquireImageRepFxn) <- environment()
  environment(acquireImageFxn_full) <- environment()
  figuresPath <- paste(strsplit(figuresPath,split="/")[[1]],collapse = "/")

  # orthogonalize if specified
  whichNA_dropped <- c()
  if(orthogonalize){
    print("Orthogonalizing Potential Outcomes...")
    if(is.null(X)){stop("orthogonalize set to TRUE, but no X specified to perform orthogonalization!")}

    # drop observations with NAs in their orthogonalized outcomes
    whichNA_dropped <- which( is.na(  rowSums( X ) ) )
    if(length(whichNA_dropped) > 0){
      # note: transportabilityMat doesn't need to drop dropNAs
      obsW <- obsW[-whichNA_dropped]
      obsY <- obsY[-whichNA_dropped]
      X <- X[-whichNA_dropped,]
      imageKeys <- imageKeys[-whichNA_dropped]
      lat <- lat[ -whichNA_dropped ]
      long <- long[ -whichNA_dropped ]
    }
    Yobs_ortho <- resid(temp_lm <- lm(obsY ~ X))
    if(length(Yobs_ortho) != length(obsY)){
      stop("length(Yobs_ortho) != length(obsY)")
    }
    plot(obsY,Yobs_ortho)
    obsY <- Yobs_ortho
    #YandW_mat[f2n(names(my_resid)),]$Yobs_ortho <- as.numeric(YandW_mat[f2n(names(my_resid)),]$Yobs_ortho)
    #YandW_mat[["Yobs_ortho"]][f2n(names(my_resid))] <- my_resid
    #try({plot(YandW_mat$Yobs_ortho,YandW_mat$Yobs);abline(a=0,b=1)},T)
    #YandW_mat$Yobs_ortho[is.na(YandW_mat$Yobs_ortho)] <- mean(YandW_mat$Yobs_ortho,na.rm=T)
  }

  # specify some training parameters + helper functions
  rzip <- function(l1,l2){  fl<-list(); for(aia in 1:length(l1)){ fl[[aia]] <- list(l1[[aia]], l2[[aia]]) }; return( fl  ) }
  GlobalMax <- tf$keras$layers$GlobalMaxPool2D()
  GlobalAve <- tf$keras$layers$GlobalAveragePooling2D()
  GlobalFlatten <- tf$keras$layers$Flatten()
  #GlobalSpatial <- tfa$layers$SpatialPyramidPooling2D(bins = list(4L,4L))
  #FinalImageSummary <- GlobalAve
  FinalImageSummary <- function(x){tf$concat(list(GlobalMax(x),GlobalAve(x)),1L)}
  #FinalImageSummary <- function(x){GlobalFlatten( GlobalSpatial( x ) )}
  #FinalImageSummary <- GlobalFlatten

  adaptiveMomentum <- F
  BNPreOutput <- F;
  ConvActivation <- "swish"
  doConvLowerDimProj <- T; LowerDimActivation <- "swish"; LowerDimInputDense <- F
  doBN_conv1 <- T
  doBN_conv2 <- T
  nDimLowerDimConv <- 3L
  kernelWidth_est <- as.integer(  kernelWidth )
  #batchSize <- 10*2L
  batchSize <- 10*2L
  batchFracOut <- max(1/3*batchSize,3) / batchSize
  nMonte_variational <- as.integer( nMonte_variational  )
  LEARNING_RATE_BASE <- .001; widthCycle <- 50
  #INV_TEMP_GLOBAL <- 1/0.5
  INV_TEMP_GLOBAL <- 1/0.5
  WhenPool <- c(1,2)
  #plot(as.matrix(do.call(rbind,replicate(10,tfd$RelaxedOneHotCategorical(temperature = 1/INV_TEMP_GLOBAL, probs = c(0.1,0.9))$sample(1L))))[,2],ylim = c(0,1))
  #points(as.matrix(do.call(rbind,replicate(10,tfd$RelaxedOneHotCategorical(temperature = 1/INV_TEMP_GLOBAL, probs = c(0.5,0.5))$sample(1L))))[,2],pch = 2,col="gray")
  #points(as.matrix(do.call(rbind,replicate(10,tfd$RelaxedOneHotCategorical(temperature = 1/INV_TEMP_GLOBAL, probs = c(0.1,0.9))$sample(1L))))[,2],pch = 1,col="black")
  BN_MOM <- 0.9
  BN_EP <- 0.01
  if(grepl(TYPE,pattern = "variational")){ BN_MOM <- 0.90^(1/nMonte_variational) }
  ConvNormText <- function(dim_){"tf$keras$layers$BatchNormalization(axis = 3L, center = T, scale = T, momentum = BN_MOM, epsilon = BN_EP)"}
  #ConvNormText <- function(dim_){dim_ <- paste(as.character(dim_),"L",sep=""); sprintf("tfa$layers$GroupNormalization(groups = c(%s), center = T, scale = T, epsilon = BN_EP)", dim_) }
  #ConvNormText <- function(dim_){"tf$keras$layers$LayerNormalization(center = T, scale = T, epsilon = BN_EP)"}
  #IdentityFxn <- function(x,training){tf$identity(x)}; ConvNormText <- function(dim_){"IdentityFxn"}
  #tmp <- eval(parse(text = ConvNormText))
  # c(mean(c(as.array((tmp_[,,,1])))), sd(c(as.array((tmp_[,,,1])))))
  #DenseNormText <- function(){"tf$keras$layers$LayerNormalization(center = T, scale = T, epsilon = BN_EP)"}
  DenseNormText <- function(){"tf$keras$layers$BatchNormalization(center = T, scale = T, momentum = BN_MOM, epsilon = BN_EP)"}
  #IdentityFxn <- function(x,training){tf$identity(x)}; DenseNormText <- function(){"IdentityFxn"}

  # set up some placeholders
  y0_true <- r2_y1_out <- r2_y0_out <- ClusterProbs_est <- NULL
  tau_i_est <- sd_tau1 <- sd_tau2 <- negELL <- y1_est <- y0_est <- y1_true <- y0_true <- NULL
  if(!"ClusterProbs" %in% ls() &
     !"ClusterProbs" %in% ls(envir=globalenv())){ClusterProbs<-NULL}

  # normalize outcomes for stability (estimates are re-normalized after training)
  if( y_density == "normal"){
    Y_mean <- mean(obsY); Y_sd <- sd(obsY)
    obsY <- (obsY - Y_mean)  /  Y_sd
  }
  if( y_density == "lognormal"){
    Y_mean <- -abs(min(obsY))-0.1; Y_sd <- sd(obsY)
    obsY <- (obsY - Y_mean)  /  Y_sd
  }
  Tau_mean_init_prior <- Tau_mean_init <- mean(obsY[obsW==1]) - mean(obsY[obsW==0])
  Tau_sd_init <- sqrt( var(obsY[obsW==1]) + var( obsY[obsW==0]) )
  Y0_sd_vec <- na.omit(replicate(10000,{ top_ <- sample(1:length(obsY),batchSize); return(sd(obsY[top_][obsW[top_]==0])) }))
  Y1_sd_vec <- na.omit(replicate(10000,{ top_ <- sample(1:length(obsY),batchSize); sd(obsY[top_][obsW[top_]==1]) }))
  tau_vec <- na.omit(replicate(10000,{ top_ <- sample(1:length(obsY),batchSize); mean(obsY[top_][obsW[top_]==1]) - mean(obsY[top_][obsW[top_]==0]) }))
  Y0_mean_init_prior <- Y0_mean_init <- mean(obsY[obsW==0]); Y0_sd_init_prior <- Y0_sd_init <- max(0.01, median(Y0_sd_vec,na.rm=T))
  Y1_mean_init_prior <- Y1_mean_init <- mean(obsY[obsW==1]); Y1_sd_init_prior <- Y1_sd_init <- max(0.01, median(Y1_sd_vec,na.rm=T))
  Y0_sd_init_prior <- tfp$math$softplus_inverse(Y0_sd_init_prior)
  Y1_sd_init_prior <- tfp$math$softplus_inverse(Y1_sd_init_prior)

  for(BAYES_STEP in c(1,2)){
    if(BAYES_STEP == 1){ print("Empirical Bayes Calibration Step (see  Krishnan et al. (2020))...") }
    if(BAYES_STEP == 2){ print("Empirical Bayes Estimation Step...") }

    if(BAYES_STEP == 1){
      nSGD_ORIG <- nSGD
      nSGD <- nSGD
      #L2_grad_scale <- 2;
      L2_grad_scale <- 0.5
      SD_PRIOR_MODEL <- .01; KL_wt <- 0
      PRIOR_MODEL_FXN <- function(name_){
        eval(parse(text = 'function(dtype, shape, name, trainable, add_variable_fn){
      d_prior <- tfd$Normal(loc = tf$zeros(shape), scale = SD_PRIOR_MODEL)
      tfd$Independent(d_prior, reinterpreted_batch_ndims = tf$size(d_prior$batch_shape_tensor())) }'))
      }
      PRIOR_MODEL_FXN("hap")
    }
    if(BAYES_STEP == 2){
      nSGD <- nSGD_ORIG #/ 2
      #L2_grad_scale <- 0.04
      L2_grad_scale <- 0.5
      #L2_grad_scale <- 2
      KL_wt <- batchSize / length(obsY)
      PRIOR_MODEL_FXN <- function(name_){
        prior_loc_name <- sprintf("%s_PRIOR_MEAN_HASH818",name_)
        prior_SD_name <- sprintf("%s_PRIOR_SD_HASH818",name_)
        ZERO_LEN_IN <- length( eval(parse(text = sprintf("%s$variables",name_)))) == 0
        if( ZERO_LEN_IN){
          prior_loc_name <- "tf$zeros(shape)"
          prior_SD_name <- "1"
        }
        if( !ZERO_LEN_IN){
          eval.parent(parse(text = sprintf("%s <- tf$constant(%s$variables[[1]],tf$float32)",prior_loc_name,name_)))
          #eval.parent(parse(text = sprintf("%s <- tf$constant(2*tf$sqrt(tf$math$reduce_variance(%s$variables[[1]])),tf$float32)",prior_SD_name,name_)))
          eval.parent(parse(text = sprintf("%s <- tf$constant(1*tf$sqrt(tf$math$reduce_variance(%s$variables[[1]])),tf$float32)",prior_SD_name,name_)))
          # eval.parent(parse(text = sprintf("%s <- tf$constant(0.1*tf$sqrt(tf$math$reduce_variance(%s$variables[[1]])),tf$float32)",prior_SD_name,name_)))# previous use
        }
        eval(parse(text = sprintf('function(dtype, shape, name, trainable, add_variable_fn){
              d_prior <- tfd$Normal(loc = (%s),
                                  scale = (%s))
              tfd$Independent(d_prior, reinterpreted_batch_ndims = tf$size(d_prior$batch_shape_tensor())) }',
                                  prior_loc_name, prior_SD_name)))
      }

      # set other priors
      Tau_mean_init_prior <- as.vector(MeanDist_tau[k_,"Mean"][[1]])
      Y0_sd_init_prior <- as.vector(SDDist_Y0[k_,"Mean"][[1]])
      Y1_sd_init_prior <- as.vector(SDDist_Y1[k_,"Mean"][[1]])
    }
    print("Building clustering model...")
    {
      BNLayer_Axis1_Clust <-  eval(parse(text = DenseNormText()))
      BNLayer_Axis1_Y0 <- eval(parse(text = DenseNormText()))
      BNLayer_Axis1_Proj <- eval(parse(text = DenseNormText()))
      BNLayer_Axis1_ProjY0 <- tf$keras$layers$BatchNormalization(axis = 1L, center = T, scale = T, momentum = BN_MOM, epsilon = BN_EP,
                                                                 beta_initializer = tf$constant_initializer(Y0_mean_init),
                                                                 gamma_initializer = tf$constant_initializer(Y0_sd_init),
                                                                 name = "BN_Y0")
      BNLayer_Axis1_ProjTau <- tf$keras$layers$BatchNormalization(axis = 1L, center = T, scale = T, momentum = BN_MOM, epsilon = BN_EP,
                                                                  beta_initializer = tf$constant_initializer(Tau_mean_init),
                                                                  gamma_initializer = tf$constant_initializer(Tau_sd_init))
      tmp_ <- 1*1/length(obsW[obsW==1])*var(obsY[obsW==1])+1/length(obsW[obsW==0])*var(obsY[obsW==0])
      if(TYPE == "variational_CNN"){ for(k___ in 1:kClust_est){
        eval(parse(text = sprintf("BNLayer_Axis1_Tau%s <- %s", k___, DenseNormText())))
        eval(parse(text =
                     sprintf("BNLayer_Axis1_ProjTau%s <- tf$keras$layers$BatchNormalization(axis = 1L, center = T, scale = T, momentum = BN_MOM, epsilon = BN_EP,
                                                               beta_initializer = tf$constant_initializer(Tau_mean_init),
                                                               gamma_initializer = tf$constant_initializer(Tau_sd_init))",k___)))
        eval(parse(text=
                     sprintf("TauProj%s = ProbDenseType(as.integer(1L),
                          kernel_prior_fn = PRIOR_MODEL_FXN('TauProj%s'),
                          name = 'TauProj%s',
                          activation='linear')",k___, k___,k___)))
      }}
      ProbLayerExecutionDevice <- '/CPU:0'
      ProbConvType <- tfp$layers$Convolution2DFlipout # more efficient, must wrap execution in with(tf$device('/CPU:0'),{...})
      ProbDenseType <-  tfp$layers$DenseFlipout # more efficient, must wrap execution in with(tf$device('/CPU:0'),{...})
      #ProbLayerExecutionDevice <- '/GPU:0'
      #ProbConvType <- tfp$layers$Convolution2DReparameterization # less efficient
      #ProbDenseType <-  tfp$layers$DenseReparameterization # less efficient
      for(conv_ in 1:nDepth_conv){
        eval(parse(text = sprintf("BNLayer_Axis3_Clust_%s <- %s",conv_,ConvNormText(nFilters) )))
        eval(parse(text = sprintf("BNLayer_Axis3_Y0_%s <- %s",conv_,ConvNormText(nFilters) )))
        tmp <- conv_ == nDepth_conv & (LowerDimInputDense == F)
        ProjNormInput <- ifelse(tmp, yes = nFilters, no = nDimLowerDimConv)
        eval(parse(text = sprintf("BNLayer_Axis3_Clust_Proj_%s <- %s",conv_,ConvNormText(ProjNormInput) )))
        eval(parse(text = sprintf("BNLayer_Axis3_Y0_Proj_%s <- %s",conv_,ConvNormText(ProjNormInput)  )))
        eval(parse(text = sprintf("ClusterConv%s <- ProbConvType(filters = nFilters,
                                                   kernel_size = kernelWidth_est,
                                                   activation = ConvActivation,
                                                   kernel_prior_fn = PRIOR_MODEL_FXN('ClusterConv%s'),
                                                   strides = strides,
                                                   name = 'ClusterConv%s',
                                                   padding = 'valid')",conv_,conv_,conv_)))
        eval(parse(text = sprintf("ClusterConvProj%s <- ProbDenseType(nDimLowerDimConv,
                                kernel_prior_fn = PRIOR_MODEL_FXN('ClusterConvProj%s'),
                                activation=LowerDimActivation,
                                name = 'ClusterConvProj%s')",conv_,conv_,conv_)))
        if(grepl(TYPE,pattern="variational")){
          eval(parse(text = sprintf("Y0Conv%s <- ProbConvType(filters = nFilters,
                                                     kernel_size=c(kernelWidth_est,kernelWidth_est),
                                                     activation = ConvActivation,
                                                     kernel_prior_fn = PRIOR_MODEL_FXN('Y0Conv%s'),
                                                     strides = strides,
                                                     name = 'Y0Conv%s',
                                                     padding = 'valid')",conv_,conv_,conv_)))
          eval(parse(text = sprintf("Y0ConvProj%s <- ProbDenseType(nDimLowerDimConv,
                                  kernel_prior_fn = PRIOR_MODEL_FXN('Y0ConvProj%s'), activation=LowerDimActivation)",conv_,conv_)))
          if(TYPE == "variational_CNN"){ for(k____ in 1:kClust_est){
            eval(parse(text = sprintf("TauConv%s_%s <- ProbConvType(filters = nFilters,
                                                     kernel_size=c(kernelWidth_est,kernelWidth_est),
                                                     activation=ConvActivation,
                                                     kernel_prior_fn = PRIOR_MODEL_FXN('TauConv%s_%s'),
                                                     strides = strides,
                                                     name = 'TauConv%s_%s',
                                                     padding = 'valid')",conv_,k____, conv_, k____,conv_, k____)))
            eval(parse(text = sprintf("TauConvProj%s_%s <- ProbDenseType(nDimLowerDimConv,
                                    kernel_prior_fn = PRIOR_MODEL_FXN('TauConvProj%s_%s'),
                                    activation=LowerDimActivation)",conv_,k____,conv_,k____)))
            eval(parse(text = sprintf("BNLayer_Axis3_Tau_%s_%s <- %s",conv_,k____,ConvNormText(nFilters))))
            eval(parse(text = sprintf("BNLayer_Axis3_Tau_Proj_%s_%s <- %s",conv_,k____,ConvNormText(nDimLowerDimConv))))
          }}
        }
        if(TYPE == "tarnet"){
          eval(parse(text = sprintf("Y0Conv%s <- tf$keras$layers$Conv2D(filters = nFilters,
                                                       kernel_size=c(kernelWidth_est,kernelWidth_est),
                                                       activation=ConvActivation,
                                                       kernel_prior_fn = PRIOR_MODEL_FXN('Y0Conv%s'),
                                                       strides = strides,
                                                       name = 'Y0Conv%s',
                                                       padding = 'valid')",conv_,conv_,conv_)))
          eval(parse(text = sprintf("Y0ConvProj%s <- ProbDenseType(nDimLowerDimConv,
                                  kernel_prior_fn = PRIOR_MODEL_FXN('Y0ConvProj%s'),
                                  name = 'Y0ConvProj%s',
                                  activation=LowerDimActivation)",conv_,conv_,conv_)))
        }
      }
      for(dense_ in 1:nDepth_dense){
        eval(parse(text = sprintf("BNLayer_Axis1_Clust_%s <- %s",dense_,DenseNormText())))
        eval(parse(text = sprintf("BNLayer_Axis1_Y0_%s <- %s",dense_,DenseNormText())))
        eval(parse(text = sprintf("DenseProj_Clust_%s <- ProbDenseType(as.integer(nDenseWidth),
                                kernel_prior_fn = PRIOR_MODEL_FXN('DenseProj_Clust_%s'),
                                activation='swish')",dense_,dense_)))
        eval(parse(text = sprintf("DenseProj_Y0_%s <- ProbDenseType(as.integer(nDenseWidth),
                                kernel_prior_fn = PRIOR_MODEL_FXN('DenseProj_Y0_%s'),
                                activation='swish')",dense_,dense_)))

      }
      ClusterProj = ProbDenseType( as.integer(kClust_est-1L),
                                   kernel_prior_fn = PRIOR_MODEL_FXN('ClusterProj'), activation='linear' )
      if(grepl(TYPE, pattern = "variational")){Y0Proj = ProbDenseType(as.integer(1L),
                                                                      kernel_prior_fn = PRIOR_MODEL_FXN('Y0Proj'),activation='linear')}
      BNLayer_Axis1_ProjY1 <- tf$keras$layers$BatchNormalization(axis = 1L, center = T, scale = T, momentum = BN_MOM, epsilon = BN_EP,
                                                                 beta_initializer = tf$constant_initializer( Y1_mean_init ),
                                                                 gamma_initializer = tf$constant_initializer( Y1_sd_init ) )
      if(TYPE == "tarnet"){
        Y0Proj = tf$keras$layers$Dense(as.integer(1L), activation='linear')
        Y1Proj = tf$keras$layers$Dense(as.integer(1L), activation='linear')
      }
      SD_scaling <- 1
      Tau_mean_init <- mean(obsY[obsW==1]) - mean(obsY[obsW==0])
      Tau_means_init <- Tau_mean_init + .01*seq(-1,1,length.out=kClust_est)*max(0.01,abs(Tau_mean_init))
      Y0_sds_prior_mean <- tfp$math$softplus_inverse(SD_scaling*(Y0_sd_init))#<-SD_scaling*sd(obsY[obsW==0])))
      Y1_sds_prior_mean <- tfp$math$softplus_inverse(SD_scaling*(Y1_sd_init))#<-SD_scaling*sd(obsY[obsW==1])))

      base_mat <- as.data.frame( matrix(list(),nrow=kClust_est,ncol=3L) ); colnames( base_mat ) <- c("Mean","SD","Prior")
      SDDist_Y1 <- SDDist_Y0 <- MeanDist_tau <- base_mat
      for(k_ in 1:kClust_est){
        # prior SD - subject-matter knowledge informs this
        # set this to a small number so network starts off as nearly deterministic
        sd_init_trainableParams <- as.numeric(tfp$math$softplus_inverse(0.01))

        MeanDist_tau[k_,"Mean"][[1]] <- list( tf$Variable(Tau_means_init[k_],trainable=T,name=sprintf("MeanTau%s_mean",k_) ) )
        MeanDist_tau[k_,"SD"][[1]] <- list( tf$Variable(sd_init_trainableParams,trainable=T,name = sprintf("MeanTau%s_sd",k_) ) )
        MeanDist_tau[k_,"Prior"][[1]] <- list( tfd$Normal(Tau_mean_init_prior, 2*sd(tau_vec) ))

        # Y0
        SDDist_Y0[k_,"Mean"][[1]] <- list( tf$Variable(3*Y0_sds_prior_mean,trainable=T,name=sprintf("SDY0%s_mean",k_) ) )
        SDDist_Y0[k_,"SD"][[1]] <- list( tf$Variable(sd_init_trainableParams,trainable=T,name=sprintf("SDY0%s_sd",k_)) )
        SDDist_Y0[k_,"Prior"][[1]] <- list( tfd$Normal(Y0_sd_init_prior,2*sd(Y0_sd_vec)))

        # Y0
        SDDist_Y1[k_,"Mean"][[1]] <- list( tf$Variable(3*Y1_sds_prior_mean,trainable=T,name=sprintf("SDY1%s_mean",k_) ) )
        SDDist_Y1[k_,"SD"][[1]] <- list( tf$Variable(sd_init_trainableParams,trainable=T,name=sprintf("SDY1%s_sd",k_)) )
        SDDist_Y1[k_,"Prior"][[1]] <- list( tfd$Normal(Y1_sd_init_prior,2*sd(Y1_sd_vec)))

        # check SD posterior
        #hist(as.numeric(tf$nn$softplus(rnorm(1000,as.numeric((SDDist_Y0[k_,"Mean"][[1]])), sd=as.numeric(tf$nn$softplus(SDDist_Y0[k_,"SD"][[1]]))))))
        #abline(v=as.numeric(tf$nn$softplus(as.numeric((SDDist_Y0[k_,"Mean"][[1]])))),col="red")
        #abline(v=sd(obsY[obsW==0]),lwd= 2)
      }
      CategoricalPrior <- tfd$Categorical(probs=rep(1/kClust_est,kClust_est))
      LocalMax <- tf$keras$layers$MaxPool2D(pool_size = maxPoolSize)
      LocalAve <- tf$keras$layers$AveragePooling2D(pool_size = maxPoolSize)
      #LocalPool <- function(x){tf$concat(list(LocalMax(x),LocalAve(x)),3L)}
      LocalPool <- LocalMax
    }

    if(compile == T){ tf_function_fxn <- function(x){tf_function(x,experimental_relax_shapes = F)}}
    if(compile == F){ tf_function_fxn <- function(x){x} }
    getClusterLogits <- tf_function_fxn( function(  m , training){
      if( !TYPE %in% "variational_minimal_visualizer" ){
        # convolution part
        for(conv__ in 1L:nDepth_conv){
          eval(parse(text = sprintf("m <-  with(tf$device( ProbLayerExecutionDevice ), { ClusterConv%s( m ) }) ",conv__)))
          if(conv__ %in% WhenPool){ m <- LocalPool( m ) }
          doLower <- (ifelse(LowerDimInputDense,yes = T, no = conv__ < nDepth_conv)) & doConvLowerDimProj
          if(doLower & doBN_conv1){eval(parse(text = sprintf("m <- BNLayer_Axis3_Clust_%s(m,training=training)",conv__)))}
          if(doLower){eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { ClusterConvProj%s( m ) })",conv__))) }
          if(doBN_conv2){eval(parse(text = sprintf("m <- BNLayer_Axis3_Clust_Proj_%s(m,training=training)",conv__)))}
        }
        print(dim(m))
        m <- FinalImageSummary(m)
        m <- BNLayer_Axis1_Clust(m, training = training)

        # dense part
        for(dense_ in 1:nDepth_dense){
          m_tminus1 <- m
          eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { DenseProj_Clust_%s(m)})",dense_)))
          eval(parse(text = sprintf("m <- BNLayer_Axis1_Clust_%s(m,training = training)",dense_)))
          if(dense_ > 1 & dense_ < nDepth_dense){ m <- m + m_tminus1 }
        }
      }

      # final projection layer
      m <- with(tf$device( ProbLayerExecutionDevice ), { ClusterProj(m) })
      if(BNPreOutput){m <- BNLayer_Axis1_Proj(m, training = training) }
      m <- tf$concat(list( tf$zeros(list(tf$shape(m)[1],1L)),m),1L)
      return( m  )
    })

    if(TYPE == "tarnet"){
      getImageRep <- tf_function_fxn( function(m,training){
        for(conv__ in 1:nDepth_conv){
          eval(parse(text = sprintf("m <-  with(tf$device( ProbLayerExecutionDevice ), { Y0Conv%s( m  ) })",conv__)))
          if(conv__ %in% WhenPool){ m <- LocalPool( m ) }
          doLower <- (ifelse(LowerDimInputDense,yes = T, no = conv__ < nDepth_conv)) & doConvLowerDimProj
          if(doLower & doBN_conv1){eval(parse(text = sprintf("m <- BNLayer_Axis3_Y0_%s(m,training=training)",conv__)))}
          if(doLower){eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { Y0ConvProj%s( m ) })",conv__))) }
          if(doBN_conv2){eval(parse(text = sprintf("m <- BNLayer_Axis3_Y0_Proj_%s(m,training=training)",conv__)))}
        }
        m <- FinalImageSummary( m  )
        m <- BNLayer_Axis1_Y0(m, training = training)
      } )
      getY0 <- tf_function_fxn(function(  m , training  ){
        m <- getImageRep(m,training=training)
        return( getY0_finalStep(m,training=training) )
      })
      getY1 <- tf_function_fxn(function(  m , training  ){
        m <- getImageRep(m,training=training)
        return( getY1_finalStep(m,training=training) )
      })
      getY0_finalStep <- tf_function_fxn(function(  m , training  ){
        m <- with(tf$device( ProbLayerExecutionDevice ), { Y0Proj(m) } )
        if(BNPreOutput){m <- BNLayer_Axis1_ProjY0(m, training = training)}
        return( m  )
      } )
      getY1_finalStep <- tf_function_fxn( function(  m , training){
        m <- with(tf$device( ProbLayerExecutionDevice ), { Y1Proj(m) } )
        if(BNPreOutput){m <- BNLayer_Axis1_ProjY1(m, training = training)}
        return( m  )
      } )
    }
    if(grepl(TYPE, pattern = "variational")){
      getY0 <- tf_function_fxn(function(  m , training  ){
        if(! TYPE %in% "variational_minimal_visualizer"){
          # convolution part
          for(conv__ in 1:nDepth_conv){
            eval(parse(text = sprintf("m <-  with(tf$device( ProbLayerExecutionDevice ), { Y0Conv%s( m )} )",conv__)))
            if(conv__ %in% WhenPool){ m <- LocalPool( m ) }
            doLower <- (ifelse(LowerDimInputDense,yes = T, no = conv__ < nDepth_conv)) & doConvLowerDimProj
            if(doLower & doBN_conv1){eval(parse(text = sprintf("m <- BNLayer_Axis3_Y0_%s(m,training=training)",conv__)))}
            if(doLower){eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { Y0ConvProj%s( m ) })",conv__))) }
            if(doBN_conv2){eval(parse(text = sprintf("m <- BNLayer_Axis3_Y0_Proj_%s(m,training=training)",conv__)))}
          }
          m <- FinalImageSummary( m  )
          m <- BNLayer_Axis1_Y0(m, training = training)

          # dense part
          for(dense_ in 1:nDepth_dense){
            m_tminus1 <- m
            eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { DenseProj_Y0_%s(m) } ) ",dense_)))
            eval(parse(text = sprintf("m <- BNLayer_Axis1_Y0_%s(m,training = training)",dense_)))
            if(dense_ > 1 & dense_ < nDepth_dense){ m <- m + m_tminus1 }
          }
        }
        m <- with(tf$device( ProbLayerExecutionDevice ), { Y0Proj(m) } )
        if(BNPreOutput){m <- BNLayer_Axis1_ProjY0(m, training = training)}
        return( m  )
      } )
      getClusterProb <- tf_function_fxn(function(m , training){
        return( tf$nn$softmax(getClusterLogits(m, training = training), 1L) )
      })
      getClusterSamp_logitInput <- tf_function_fxn( function(logits_){
        #clustT_samp = tf$cast(tfd$OneHotCategorical(logits = logits_)$sample(1L),dtype = tf$float32)
        clustT_samp = tfd$RelaxedOneHotCategorical(temperature = 1/INV_TEMP_GLOBAL, logits = logits_)$sample(1L)
      })

      if(TYPE == "variational_CNN"){
        getTau <- tf_function_fxn(function(  m, training  ){
          m_orig <- m
          m_ret<-list();for(k____ in 1:kClust_est){
            for(conv__ in 1:nDepth_conv){
              if(conv__ == 1){ m <- m_orig }
              eval(parse(text = sprintf("m <-  TauConv%s_%s( m )",conv__,k____)))
              if(conv__ %in% WhenPool){ m <- LocalPool( m ) }
              doLower <- (ifelse(LowerDimInputDense,yes = T, no = conv__ < nDepth_conv)) & doConvLowerDimProj
              if(doLower & doBN_conv1){eval(parse(text = sprintf("m <- BNLayer_Axis3_Tau_%s_%s(m,training=training)",conv__,k____)))}
              if(doLower){eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { TauConvProj%s_%s( m )})",conv__,k____))) }
              if(doBN_conv2){eval(parse(text = sprintf("m <- BNLayer_Axis3_Tau_Proj_%s_%s(m,training=training)",conv__,k____)))}
            }
            m <- FinalImageSummary( m  )
            eval(parse(text = sprintf("m <- BNLayer_Axis1_Tau%s(m, training = training)",k____)))
            eval(parse(text = sprintf("m <- with(tf$device( ProbLayerExecutionDevice ), { TauProj%s(m)} )",k____)))
            if(BNPreOutput){
              eval(parse(text = sprintf("m <- BNLayer_Axis1_ProjTau%s(m, training = training)",k____)))
            }
            m_ret[[k____]] <- m
          }
          m_ret <- tf$concat(m_ret,1L)
          return( m_ret  )
        } )
        getY1 <- tf_function_fxn( function(  m , training){
          Y0 <- getY0(m=m,training = training)
          Clust_logits <- getClusterLogits(m,training = training)
          clustT <- tf$squeeze(getClusterSamp_logitInput(Clust_logits),0L)
          tau_i <- getTau(m,training = training)
          tau_i <- tf$reduce_sum(tf$multiply(tau_i, clustT),1L,keepdims=T)
          Y1 <- Y0 + tau_i
          return(  Y1   )
        } )
      }
      if(grepl(TYPE, pattern = "variational_minimal")){
        getY1 <- tf_function_fxn( function(  m , training){
          Y0 <- getY0(m=m,training = training)
          Clust_logits <- getClusterLogits(m,training = training)
          clustT <- tf$squeeze(getClusterSamp_logitInput(Clust_logits),0L)
          ETau_draw <-  (tfd$Normal(getTau_means(),
                                    tf$nn$softplus(MeanDist_tau[,"SD"])))$sample(1L)
          tau_i <- tf$reduce_sum(tf$multiply(ETau_draw, clustT),1L,keepdims=T)
          Y1 <- Y0 + tau_i
          return(  Y1   )
        } )
      }
      marginal_tau <- tf$constant(mean(obsY[obsW==1],na.rm=T)-mean(obsY[obsW==0],na.rm=T), tf$float32)
      marginal_lambda <- tf$constant(.01, tf$float32)
      TauRunning <- tf$Variable(t(rep(as.matrix(marginal_tau),times=kClust_est)),
                                dtype=tf$float32,trainable=F)
      getTau_means <- tf_function_fxn( function(){
        if(y_density == "normal"){ ret_ <- tf$identity(MeanDist_tau[,"Mean"]) }
        if(y_density == "lognormal"){ ret_ <- tf$identity(MeanDist_tau[,"Mean"]) }
        return(ret_)
      } )
    }

    getLoss <- tf_function_fxn( function(dat,treat,y,training){
      if(TYPE == "tarnet"){
        m <- getImageRep(dat,training = training)
        Y0_hat <- getY0_finalStep(m,training=training)
        Y1_hat <- getY1_finalStep(m,training=training)
        Yobs_hat <- tf$multiply(Y1_hat,tf$expand_dims(treat,1L)) + tf$multiply(Y0_hat,tf$expand_dims(1-treat,1L))
        minThis <- tf$reduce_sum(tf$square(tf$expand_dims(y,1L) - Yobs_hat))
      }

      if(grepl(TYPE,pattern= "variational")){
        # cluster probabilities
        Clust_logits <- replicate(nMonte_variational,tf$expand_dims(getClusterLogits(dat, training = training),0L))
        Clust_logits <- tf$concat(Clust_logits,0L)
        clustT <- tf$squeeze(getClusterSamp_logitInput(Clust_logits),0L)
        Clust_probs <- tf$nn$softmax(Clust_logits, 2L)
        #CategoricalPost <- tfd$Categorical(probs = Clust_probs)

        EY0_i <- replicate(nMonte_variational,tf$expand_dims(getY0(dat,training = training),0L))
        EY0_i <- tf$squeeze(tf$concat(EY0_i,0L),2L)

        # enforce ATE
        if(TYPE == "variational_CNN"){
          ETau_draw <- tf$concat(replicate(nMonte_variational,
                                           tf$expand_dims(getTau(dat,training = training),0L)),0L)
        }
        if(grepl(TYPE, pattern = "variational_minimal")){
          Tau_mean_vec <- getTau_means()
          MeanDist_Tau_post = (tfd$Normal(Tau_mean_vec, Tau_sd_vec <- tf$nn$softplus(MeanDist_tau[,"SD"])))
          ETau_draw <- tf$expand_dims(MeanDist_Tau_post$sample(nMonte_variational),1L)
        }

        SDDist_Y1_post = (tfd$Normal(tf$identity(SDDist_Y1[,"Mean"]), tf$nn$softplus(SDDist_Y1[,"SD"])))
        SDDist_Y0_post = (tfd$Normal(tf$identity(SDDist_Y0[,"Mean"]), tf$nn$softplus(SDDist_Y0[,"SD"])))

        EY1SD_draw <- tf$nn$softplus(SDDist_Y1_post$sample(nMonte_variational))
        EY0SD_draw <- tf$nn$softplus(SDDist_Y0_post$sample(nMonte_variational))

        tau_i <- tf$reduce_sum( tf$multiply(ETau_draw, clustT), 2L )
        EY1_i <- EY0_i + tau_i
        impliedATE <- tf$reduce_mean(tau_i,1L)
        Sigma2_Y0_i <- tf$reduce_sum(tf$multiply( tf$expand_dims(EY0SD_draw^2,1L), clustT),2L)
        Sigma2_Y1_i <- tf$reduce_sum(tf$multiply( tf$expand_dims(EY1SD_draw^2,1L), clustT),2L)
        treat <- tf$expand_dims( treat , 0L)
        Y_Sigma <- (tf$multiply( 1 - treat , Sigma2_Y0_i ) +
                      tf$multiply( treat, Sigma2_Y1_i ))^0.5
        Y_Mean <- tf$multiply( 1 - treat, EY0_i ) +
          tf$multiply( treat, EY1_i)

        # KL terms
        KLterm <- tf$zeros(list())
        if(! TYPE  %in% c("variational_minimal_visualizer")){
          for(conv_ in 1:nDepth_conv){
            KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(ClusterConv%s$kernel_posterior,ClusterConv%s$kernel_prior)",conv_,conv_)))
            KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(Y0Conv%s$kernel_posterior,Y0Conv%s$kernel_prior)",conv_,conv_)))
            if((ifelse(LowerDimInputDense,yes = T, no = conv_ < nDepth_conv)) & doConvLowerDimProj){
              KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(Y0ConvProj%s$kernel_posterior,Y0ConvProj%s$kernel_prior)",conv_,conv_)))
              KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(ClusterConvProj%s$kernel_posterior,ClusterConvProj%s$kernel_prior)",conv_,conv_)))
            }
            if(TYPE == "variational_CNN"){
              for(k_ in 1:kClust_est){
                KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(TauConv%s_%s$kernel_posterior,TauConv%s_%s$kernel_prior)",conv_,k_,conv_,k_)))
                if((ifelse(LowerDimInputDense,yes = T, no = conv_ < nDepth_conv)) & doConvLowerDimProj){
                  KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(TauConvProj%s_%s$kernel_posterior,TauConvProj%s_%s$kernel_prior)",conv_,k_,conv_,k_)))
                }
              }
            }
          }
          for(dense_ in 1:nDepth_dense){
            KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(DenseProj_Y0_%s$kernel_posterior,DenseProj_Y0_%s$kernel_prior)",nDepth_dense, nDepth_dense)))
            KLterm <- KLterm + eval(parse(text=sprintf("tfd$kl_divergence(DenseProj_Clust_%s$kernel_posterior,DenseProj_Clust_%s$kernel_prior)",nDepth_dense, nDepth_dense)))
          }
        }
        KLterm <- KLterm + tfd$kl_divergence(ClusterProj$kernel_posterior,ClusterProj$kernel_prior)
        KLterm <- KLterm + tfd$kl_divergence(Y0Proj$kernel_posterior,Y0Proj$kernel_prior)
        if(TYPE == "variational_minimal"){
          KLterm <- KLterm + tf$reduce_sum(tfd$kl_divergence(MeanDist_Tau_post, (MeanDist_tau[,"Prior"][[1]])))
        }
        KLterm <- KLterm + tf$reduce_sum(tfd$kl_divergence(SDDist_Y0_post, (SDDist_Y0[,"Prior"][[1]])))
        KLterm <- KLterm + tf$reduce_sum(tfd$kl_divergence(SDDist_Y1_post, (SDDist_Y1[,"Prior"][[1]])))

        # some commented analyses to triple-check code correctness re: initialization
        #plot(as.numeric(tf$reduce_mean(Y_Mean,0L)),as.numeric(y),col=as.numeric(treat)+1);abline(a=0,b=1)
        #lim_<-summary(c(as.numeric(tf$reduce_mean(Y_Mean,0L)),as.numeric(y)))[c(1,6)]
        #try({plot(as.numeric(tf$reduce_mean(Y_Mean,0L)),as.numeric(y),ylim=lim_,xlim=lim_,col=as.numeric(treat)+1);abline(a=0,b=1)},T)
        #print( 1-sum( (as.numeric(tf$reduce_mean(Y_Mean,0L))-as.numeric(y))^2)/sum((as.numeric(y)-mean(as.numeric(y)))^2))
        if(y_density == "normal"){
          likelihood_distribution_draws <- tfd$Normal(loc = Y_Mean, scale = Y_Sigma)
        }
        if(y_density == "lognormal"){
          likelihood_distribution_draws <- tfd$LogNormal(loc = Y_Mean, scale = Y_Sigma)
        }
        likelihood_distribution_expectation <- tf$reduce_mean(likelihood_distribution_draws$log_prob( tf$expand_dims(y,0L) ),0L)
        #plot(as.numeric(tf$reduce_mean(likelihood_distribution_draws$log_prob( tf$expand_dims(y,0L) ),0L)),as.numeric( y  ),col = as.numeric( treat)+1)
        minThis <- tf$negative(tf$reduce_sum( likelihood_distribution_expectation )) +
          KL_wt * KLterm + tf$reduce_mean(tf$multiply(marginal_lambda, tf$square(marginal_tau - impliedATE)))
        minThis <- minThis / batchSize
      }
      return( minThis )
    })

    print("Initial forward pass...")
    for(bool_ in c(T,F)){
      print(bool_)
      with(tf$GradientTape() %as% tape, {
        samp_ <- sample(1:length(obsW),batchSize)
        myLoss_forGrad <- getLoss( dat = acquireImageRepFxn(keys = imageKeys[samp_],training = bool_),
                                   treat = tf$constant(obsW[samp_],tf$float32),
                                   y = tf$constant(obsY[samp_],tf$float32),
                                   training = bool_ )
        if(bool_==T){ trainable_variables  <- tape$watched_variables() }
      })
    }
    print(sprintf("nTrainableParams: %i",nTrainableParams <- sum(unlist(lapply(unlist(trainable_variables,recursive=F),function(zer){
      len_<-try(length((zer)),T);return( len_ ) })))))

    # define optimizer and training step
    #optimizer_tf = tf$optimizers$legacy$Adam(learning_rate = LEARNING_RATE_BASE)
    optimizer_tf = tf$optimizers$legacy$Nadam(learning_rate = LEARNING_RATE_BASE)
    if(adaptiveMomentum == T){
      BETA_1_INIT <- 0.1
      optimizer_tf = tf$optimizers$legacy$Adam(learning_rate = 0,beta_1 = BETA_1_INIT)#$,clipnorm=1e1)
      #optimizer_tf = tf$optimizers$legacy$Nadam(learning_rate=LEARNING_RATE_BASE,beta_1 = BETA_1_INIT)#$,clipnorm=1e1)
    }
    InvLR <- tf$Variable(0.,trainable  =  F)

    trainStep <-  (function(dat,y,treat, training){
      if(is.na(as.numeric(myLoss_forGrad))){browser()}

      with(tf$GradientTape() %as% tape, {
        myLoss_forGrad <<- getLoss( dat = dat,
                                    treat = treat,
                                    y = y,
                                    training = training)  })
      my_grads <<- tape$gradient( myLoss_forGrad, tape$watched_variables() )

      # update LR
      #optimizer_tf$learning_rate$assign(   LEARNING_RATE_BASE*abs(cos(i/nSGD*widthCycle)  )*(i<=nSGD/2)+LEARNING_RATE_BASE*(i>nSGD/2)/(0.001+abs(i-nSGD/2)^0.2 )   )
      L2_grad_i <<- sqrt(sum((grad_i <- as.numeric(tf$concat(lapply(my_grads,function(x) tf$reshape(x,list(-1L,1L))),0L) ))^2) )
      x_i <- as.numeric( tf$concat((lapply(trainable_variables, function(zer){tf$reshape(zer,-1L)})),0L))
      if(i == 1){
        L2_grad_init <<- L2_grad_scale*L2_grad_i
        InvLR$assign(  L2_grad_init )
        print(sprintf("Initial LR: %.3f",1/L2_grad_init))
      }
      if(i > 1) { InvLR$assign_add( tf$divide( L2_grad_i,InvLR ) ) }
      {
        optimizer_tf$apply_gradients( rzip(my_grads, trainable_variables))
        optimizer_tf$learning_rate$assign( tf$math$reciprocal( InvLR ) )
      }

      # update momentum
      if(adaptiveMomentum == T){
        if(i >= 4){
          DENOM <- sqrt( sum((x_i-x_i_minus_1)^2))
          NUM <- sqrt( sum((grad_i-grad_i_minus_1)^2))
          LR_current <- as.numeric( optimizer_tf$learning_rate  )
          #UPPER_MOM <- 1-10^(-3)
          UPPER_MOM <- 1-10^(-2)
          MomenetumNextIter <<- max(0,min((1-sqrt(LR_current*NUM/(0.000001+DENOM)))^2,UPPER_MOM))
          #optimizer_tf$momentum$assign(MomenetumNextIter)
          optimizer_tf$beta_1$assign( MomenetumNextIter )
        }
        x_i_minus_1 <<- x_i
        grad_i_minus_1 <<- grad_i
      }
    })

    # perform training jump
    print("Starting training...")
    if(BAYES_STEP == 2){
      eval(parse(text = sprintf("rm(%s)",paste(ls()[grepl(ls(),pattern="HASH818")] ,collapse= ',') )))
    }
    trainIndices <-sort(sample(1:length(obsW),length(obsW) - (nTest <- 0L)))
    testIndices <-sort(  (1:length(obsW))[! 1:length(obsW) %in% trainIndices] )
    if(length(testIndices)==0){testIndices <- trainIndices}
    print(length(testIndices))
    L2grad_vec <- loss_vec <- rep(NA,times=(nSGD))
    batch_indices_list <- (sapply(1:(max(1,round((batchSize*nSGD) / length(trainIndices)))),function(zer){
      zer*length(trainIndices)+sample(1:length(trainIndices) %% ceiling(length(trainIndices)/batchSize)+1)
    }))

    # training loop
    IndicesByW <- tapply(1:length(obsW),obsW,c)
    UniqueImageKeysByW <- tapply(imageKeys,obsW,function(zer){sort(unique(zer))})
    UniqueImageKeysByIndices <- list(tapply(which(obsW==0),imageKeys[obsW==0],function(zer){sort(unique(zer))}),
                                     tapply(which(obsW==1),imageKeys[obsW==1],function(zer){sort(unique(zer))}))
    tauMeans <- c();i_<-1;for(i in i_:(n_iters <- length(unique_batch_indices <- sort(unique(c(batch_indices_list)))))){
      if(i %% 25 == 0){gc(); py_gc$collect()}
      #batch_indices <- unlist(apply(batch_indices_list == unique_batch_indices[i],2,which))
      #batch_indices_reffed <- trainIndices[batch_indices]
      #keys_SELECTED <- sample(unique(imageKeys),batchSize)
      #batch_indices_reffed <- sapply(keys_SELECTED,function(zer){
      #f2n(sample(as.character(which(imageKeys %in% zer)),1L)) })
      batch_indices_reffed <-  c(  sapply(1:2, function(ze){
        w_keys <- UniqueImageKeysByW[[ze]]
        samp_w_keys <- sample(unique(w_keys),max(1,round(batchSize/4)))
        unlist(  lapply(UniqueImageKeysByIndices[[ze]][as.character(samp_w_keys)],function(fa){
          f2n(sample(as.character(fa),1))
        }) ) }))
      #table(  obsW[batch_indices_reffed] )
      #table(YandW_mat$geo_long_lat_key[batch_indices_reffed])
      trainStep(dat = acquireImageRepFxn( keys = imageKeys[batch_indices_reffed],training = T ),
                y = tf$constant(obsY[batch_indices_reffed],tf$float32),
                treat = tf$constant(obsW[batch_indices_reffed],tf$float32),
                training = T)
      loss_vec[i] <- myLoss_forGrad <- as.numeric( myLoss_forGrad )
      L2grad_vec[i] <- as.numeric( L2_grad_i )
      if(is.na(myLoss_forGrad)){print("NA in LOSS");browser()}
      i_ <- i ; if(i %% 20==0 | i == 1){
        print(sprintf("Optim iter %i of %i",i,n_iters));par(mfrow = c(1,1));
        try({plot(loss_vec);points(smooth.spline( na.omit(loss_vec) ),log="y",col="red",type = "l",lwd=5)},T)
        if(TYPE == "variational_minimal"){  print(as.numeric(getTau_means())) }
      }
    }
  }
  try(plot(L2grad_vec),T)
  print("Obtaining out of sample predicted means...")
  try({par(mfrow = c(1,1));plot(loss_vec);try(points(smooth.spline( na.omit(loss_vec) ),col="red",type = "l",lwd=5),T)},T)
  print("Getting Ys...")
  for(y_t_ in c(0,1)){
    test_tab <- sort( 1:length(testIndices)%%round(length(testIndices)/max(1,round(batchFracOut*batchSize))));
    Y_test_est <-  tapply(testIndices,test_tab,function(zer){
      if(runif(1)<0.1){ gc(); py_gc$collect() }
      atP <- max(zer)/length(test_tab)
      if((round(atP,2)*100) %% 10 == 0){ print(atP) }
      im_zer <- acquireImageRepFxn(keys = imageKeys[zer], training = F)
      l_ <- replicate(nMonte_predictive,
                      eval(parse(text = sprintf("list(tf$expand_dims(getY%s(m=im_zer, training = F),0L))",y_t_))))
      names(l_) <- NULL;
      l_ <- tf$concat(l_,0L)
      as.matrix(tf$reduce_mean(l_,0L))
    })
    Y_test_est <- do.call(rbind,Y_test_est)
    eval(parse(text = sprintf("Y%s_test_est <- Y_test_est",y_t_)))

    # get out of sample predictions
    Y_test_truth <- obsY[testIndices]
    W_test <- obsW[testIndices]
    yt_true <- Y_test_truth[W_test==y_t_]
    yt_est <- as.numeric(Y_test_est)[W_test==y_t_]
    yt_lims <- summary(c(yt_true,yt_est))[c(1,6)]
    print((summary(lm(yt_true~yt_est))))
    r2_yt_out <- 1 - sum( (yt_est - yt_true)^2 ) / sum( (yt_true - mean(yt_true))^2 )
    if(y_t_ == 0){ r2_y0_out <- r2_yt_out }
    if(y_t_ == 1){ r2_y1_out <- r2_yt_out }
    rm( Y_test_est )
  }

  try({plot(y0_true,y0_est,ylim = y0_lims,xlim=y0_lims);abline(a=0,b=1,lty=2,col="gray")},T)
  try({plot(y1_true,y1_est,ylim = y1_lims,xlim=y1_lims);abline(a=0,b=1,lty=2,col="gray")},T)

  rm(optimizer_tf);gc(); py_gc$collect()

  # get cluster probs
  print("Getting final cluster probabilities....")
  batch_indices_tab <- sort( 1:length(obsY)%%round(length(obsY)/max(1,ceiling(batchFracOut*batchSize))))
  if(TYPE == "tarnet" | TYPE=="variational_CNN"){
    Y0_est <- do.call(rbind,tapply(1:length(batch_indices_tab),batch_indices_tab, function(indi_){
      if(runif(1)<0.1){ gc(); py_gc$collect() }
      atP <- max(indi_/length(obsY))
      if((round(atP,2)*100) %% 10 == 0){ print(atP) }
      im_indi <- acquireImageRepFxn(keys = imageKeys[indi_],training = F)
      as.matrix(tf$reduce_mean(tf$concat(replicate(nMonte_predictive,getY0(im_indi,training = F)),1L),1L))
    }))
    Y1_est <- do.call(rbind,tapply(1:length(batch_indices_tab),batch_indices_tab, function(indi_){
      if(runif(1)<0.1){ gc(); py_gc$collect() }
      atP <- max(indi_/length(obsY))
      if((round(atP,2)*100) %% 10 == 0){ print(atP) }
      im_indi <- acquireImageRepFxn(keys = imageKeys[indi_],training = F)
      as.matrix(tf$reduce_mean(tf$concat(replicate(nMonte_predictive,getY1(im_indi,training = F)),1L),1L))
    }))
    tau_i_est <- (Y1_est - Y0_est) * Y_sd
    impliedATE <- mean(  tau_i_est )

    clusters_info <- kmeans(tau_i_est,centers=2)
    tau_vec <- c( clusters_info$centers )
    tau1 <- c(tau_vec[1]); tau2 <- c(tau_vec[2])
  }

  if(grepl(TYPE, pattern = "variational")){
    print("Starting estimates for cluster probabilities")
    ClusterProbs_est <- tapply(1:length(batch_indices_tab),batch_indices_tab, function(indi_){
      atP <- max(indi_/length(obsY))
      if((round(atP,2)*100) %% 10 == 0){ print(atP) }
      if(runif(1)<0.1){ gc(); py_gc$collect() }
      im_indi <- acquireImageRepFxn(keys = imageKeys[indi_],training = F)
      ClusterProbs_est_ <- replicate(nMonte_predictive,as.matrix(getClusterProb(im_indi,training = F)))
      ClusterProbs_std_ <- apply(ClusterProbs_est_,1:2,function(re){sd(re,na.rm=T)})
      ClusterProbs_est_ <- apply(ClusterProbs_est_,1:2,function(re){mean(re,na.rm=T)})
      ClusterProbs_lower_conf_ <- ClusterProbs_est_ - 0. * ClusterProbs_std_
      return(list("ClusterProbs_est_"=ClusterProbs_est_,
                  "ClusterProbs_lower_conf_"=ClusterProbs_lower_conf_,
                  "ClusterProbs_std_"=ClusterProbs_std_))
    } )
    ClusterProbs_lower_conf <- do.call(rbind,lapply(ClusterProbs_est,function(zer) zer$ClusterProbs_lower_conf_))
    ClusterProbs_std <- do.call(rbind,lapply(ClusterProbs_est,function(zer) zer$ClusterProbs_std_))
    ClusterProbs_est <- lapply(ClusterProbs_est,function(zer) zer$ClusterProbs_est_)
    ClusterProbs_est <- (ClusterProbs_est_full <- do.call(rbind, ClusterProbs_est))[,2]
    Clust_probs_marginal_final <- colMeans( ClusterProbs_est_full )

    if(grepl(TYPE,pattern="variational_minimal")){
      impliedATE <- mean(replicate(100,sum(Y_sd*as.numeric(getTau_means())*Clust_probs_marginal_final)))
      #getTau_means(); tf$nn$softplus(MeanDist_tau[["SD"]])
    }
    gc(); py_gc$collect()

    # characterizing the treatment effects
    print("Summarizing results...")
    SDDist_Y1_post = (tfd$Normal(tf$identity(SDDist_Y1[,"Mean"]), tf$nn$softplus(SDDist_Y1[,"SD"])))
    SDDist_Y0_post = (tfd$Normal(tf$identity(SDDist_Y0[,"Mean"]), tf$nn$softplus(SDDist_Y0[,"SD"])))
    Sigma1_sd_vec <- as.numeric(tf$reduce_mean(tf$nn$softplus(SDDist_Y1_post$sample(100L)),0L))
    Sigma0_sd_vec <- as.numeric(tf$reduce_mean(tf$nn$softplus(SDDist_Y0_post$sample(100L)),0L))

    # get uncertainties
    if(grepl(TYPE,pattern="variational_minimal")){
      tau_vec <- as.numeric( getTau_means() )
      tau_vec <- tau_vec * Y_sd
      for(k_ in 1:kClust_est){eval(parse(text = sprintf("tau%s <- tau_vec[k_]",k_))) }
      Tau_mean_vec <- getTau_means()
      MeanDist_Tau_post = (tfd$Normal(Tau_mean_vec, Tau_sd_vec <- tf$nn$softplus(MeanDist_tau[,"SD"])))
      Tau_sd_vec_ <- as.numeric(tf$sqrt(tf$math$reduce_variance(MeanDist_Tau_post$sample(100L),0L)))
      Tau_sd_vec <- sqrt(   Sigma1_sd_vec^2 + Sigma0_sd_vec^2 + Tau_sd_vec_^2 )
      Tau_sd_vec <- Tau_sd_vec * Y_sd
      sd_tau1 <- Tau_sd_vec[1]
      sd_tau2 <- Tau_sd_vec[2]
    }

    # obtaining the neg log likelihood if desired
    negELL <- NA; if(T == F){
      # obtaining the negative LL
      KL_wt_orig <- KL_wt
      if(! "function" %in% class(getLoss)){print("getLoss must be R function for this part to work!")}
      KL_wt <- 0
      negELL <- tapply(1:length(batch_indices_tab),batch_indices_tab, function(indi_){
        ret_ <- as.numeric(getLoss( dat = acquireImageRepFxn(  keys = imageKeys[indi_]  , training = F),
                                    treat = tf$constant(obsW[indi_],tf$float32),
                                    y = tf$constant(obsY[indi_],tf$float32),
                                    training = F ))
        return( ret_ )
      })
      negELL <- sum(negELL)
      KL_wt <- KL_wt_orig
    }
  }

  # transportability analysis
  cluster_prob_transport_means <- cluster_prob_transport_distribution <- NULL
  if(!is.null(transportabilityMat)){
    print("Getting posterior predictive for transportability analysis...")
    {
      GetProbAndExpand <- tf_function_fxn(function(m){tf$expand_dims(getClusterProb(m,training = F),0L) })
      full_tab <- sort( 1:nrow(transportabilityMat) %% round(nrow(transportabilityMat)/max(1,round(batchFracOut*batchSize))));
      cluster_prob_transport_info <- tapply(1:nrow(transportabilityMat),full_tab,function(zer){
        if(runif(1)<0.1){ gc(); py_gc$collect() }
        atP <- max(  zer / nrow(transportabilityMat))
        if((round(atP,2)*100) %% 10 == 0){ print(atP) }
        keys_ <-  transportabilityMat$key[zer]
        im_keys <- acquireImageFxn_transportability( keys = keys_ )
        pred_ <- replicate(nMonte_predictive,as.array(GetProbAndExpand(im_keys) ))
        list("mean"=apply(pred_[1,,,],1:2,mean),
             "var"=apply(pred_[1,,,],1:2,var))
      })
      cluster_prob_transport_info <- do.call(rbind,cluster_prob_transport_info)
      cluster_prob_transport_means <- do.call(rbind, cluster_prob_transport_info[,1])
      cluster_prob_transport_var <- do.call(rbind, cluster_prob_transport_info[,2])
      colnames(cluster_prob_transport_means) <- paste('mean_k',1:ncol(cluster_prob_transport_means), sep = "")
      colnames(cluster_prob_transport_var) <- paste('var_k',1:ncol(cluster_prob_transport_means), sep = "")
      transportabilityMat <- try(cbind(transportabilityMat,
                                   cluster_prob_transport_means,
                                   cluster_prob_transport_var),T)
      if("try-error" %in% class(transportabilityMat)){ browser() }
    }
  }


  if(!c("monti") %in% ls(envir = globalenv()) & SimMode == F){
    pdf_name_key <- "RealDataFig"; monti <- NA
  }
  if( plotResults == T){
    if(TYPE == "variational_minimal"){
      synth_seq <- seq(min(tau1 - 2 * as.numeric(sd_tau1),tau2 - 2 * as.numeric(sd_tau2)),
                       max(tau1 + 2 * as.numeric(sd_tau1),tau2 + 2 * as.numeric(sd_tau2)),
                       length.out=1000)

      if(truthKnown <- ("ClusterProbs" %in% ls(envir = globalenv()))){
        my_density <- density(ClusterProbs)
        my_density$y <- my_density$y
        synth_seq <- seq(min(my_density$x,na.rm=T),max(my_density$x,na.rm=T),length.out=100)
      }

      d1 <- dnorm(synth_seq, mean = as.numeric(tau1), sd = ( as.numeric(sd_tau1)) )
      d2 <- dnorm(synth_seq, mean = as.numeric(tau2), sd = ( as.numeric(sd_tau2)) )
      pdf(sprintf("%s/HeteroSimTauDensity%s_%s_ExternalFigureKey%s.pdf",figuresPath, pdf_name_key, TYPE, externalFigureKey))
      {
        par(mar=c(5,5,1,1))
        numbering_seq <- c("1","1")
        col_seq <- c("black","black")
        col_seq[which.max(c(as.numeric(tau1),as.numeric(tau2)))] <- "red"
          numbering_seq[which.max(c(as.numeric(tau1),as.numeric(tau2)))] <- 2
          if(!truthKnown){
            my_density <- data.frame("y"=max(c(d1,d2)))
          }
          plot(my_density,
               col=ifelse(truthKnown,yes="darkgray",no="white"),
               lty = 2,
               xlim = c(min(synth_seq),max(synth_seq)),
               ylim = c(0,max(my_density$y,na.rm=T)*1.5),
               cex.lab = 2, main = "",lwd = 3,
               ylab = "Density",
               xlab = "Per Image Treatment Effect")
          if(!truthKnown){ axis(2,cex.axis = 1) }
          points(tau_vec,c(0,0),
                 col = col_seq,
                 pch = "|", cex = 4)
          points(synth_seq, d1,type = "l", col= col_seq[1],lwd = 3)
          points(synth_seq, d2,type = "l", col= col_seq[2],lwd = 3)
          text(tau_vec,
               rep(max(c(d1,d2),na.rm=T)*0.1,2),
               labels = c(eval(parse(text = sprintf("expression(hat(tau)(%s))",
                                                    numbering_seq[1]))),
                          eval(parse(text = sprintf("expression(hat(tau)(%s))",
                                                    numbering_seq[2])))),
               col = col_seq,
               cex = 2)
          legends_seq_vec <- c(expression("True"~p(Y[i](1)-Y[i](0)~"|"~M[i])),
                               eval(parse(text=sprintf('
              expression(hat(p)~"("~Y[i](1)-Y[i](0)~"|"~Z[i]==%s~")")',numbering_seq[1]))),
                               eval(parse(text=sprintf('
              expression(hat(p)~"("~Y[i](1)-Y[i](0)~"|"~Z[i]==%s~")")',numbering_seq[2]))))
          lty_seq_vec <- c(2,1,1)
          col_seq_vec <- c("gray",col_seq)
          if(!truthKnown){
            legends_seq_vec <- legends_seq_vec[-1]
            col_seq_vec <- col_seq_vec[-1]
            lty_seq_vec <- lty_seq_vec[-1]
          }
          legend("topleft", legend = legends_seq_vec,
                 box.lwd = 0, box.lty = 0, cex = 2,
                 lty = lty_seq_vec, col = col_seq_vec, lwd = 3)
      }
      dev.off()

      if(truthKnown){
        order_ <- order(ClusterProbs)
        if(cor(ClusterProbs, ClusterProbs_est) > 0){
          # we do this so the coloring stays consistent
          col_dim <- rank(ClusterProbs_est)#gtools::quantcut(ClusterProbs_est, q = 100)
        }
        if(cor(ClusterProbs, ClusterProbs_est) < 0){
          col_dim <- rank(-ClusterProbs_est)#gtools::quantcut(ClusterProbs_est, q = 100)
        }
        pdf(sprintf("%s/HeteroSimClusterEx%s_ExternalFigureKey%s.pdf",figuresPath, pdf_name_key, externalFigureKey))
        {
          par(mar=c(5,5,1,1))
          plot( ClusterProbs[order_],
                1:length(order_)/length(order_),
                xlab = "Per Image Treatment Effect",
                ylab = "Empirical CDF(x)",pch = 19,
                col = viridis::magma(n=length(col_dim),alpha=0.9)[col_dim][order_],
                cex = 1.5, cex.lab = 2)
          legend("topleft",
                 box.lwd = 0, box.lty = 0,
                 pch = 19, #box.col = "white",
                 col = c(viridis::magma(5)[2],
                         viridis::magma(5)[3],
                         viridis::magma(5)[4]),
                 cex = 2,
                 legend=c("Higher Clust 1 Prob.",
                          "...",
                          "Higher Clust 2 Prob."))
        }
        dev.off()
      }
    }
  }
  if(SimMode == F){
    par(mfrow=c(1,1))
    try(plot(ClusterProbs_est),T)
    for(type_plot in c("uncertainty","mean")){
      rows_ <- kClust_est; nExamples <- 5
      if(type_plot == "uncertainty"){rows_ <- 1L}
      #type_plot <- "mean"
      plot_fxn <- function(){
        pdf(sprintf("%s/VisualizeHeteroReal_%s_%s_%s_ExternalFigureKey%s.pdf",figuresPath, TYPE,type_plot,orthogonalize,externalFigureKey),
            height = ifelse(type_plot == "mean", yes = 4*rows_*3, no = 4),
            width = 4*nExamples)
        {
          #
          if(type_plot %in% c("mean")){
            par(mar=c(2, 5.9, 3, 0.5))
            layout_mat <- matrix(c(1:nExamples*3-3+1,
                                   1:nExamples*3-1,
                                   1:nExamples*3),nrow = 3, byrow = T)
            layout_mat <- rbind(layout_mat,
                                layout_mat+max(layout_mat))
          }
          if(type_plot %in% c("uncertainty")){
            par(mar=c(2,2,4,2)); layout_mat <- t(1:nExamples)
          }
          layout(mat = layout_mat,
                 widths = rep(2,ncol(layout_mat)),
                 heights = rep(2,nrow(layout_mat)))
          reNorm <- function(ar){for(ib in 1:NBANDS){ar[,,ib] <- (ar[,,ib])*NORM_SD[ib] + NORM_MEAN[ib]  };return(ar) }
          gc(); py_gc$collect()
          plotting_coordinates_mat <- c()
          total_counter <- 0
          for(k_ in 1:rows_){
            used_coordinates <- c()
            for(i in 1:5){
              #if(k_ == 2 & type_plot == "mean"){ browser() }
              #if(k_ == 2 & i == 1){ browser() }
              print(sprintf("Type Plot: %s; k_: %s, i: %s", type_plot, k_, i))
              total_counter <- total_counter + 1
              rfxn <- function(xer){xer}
              bad_counter <- 0;isUnique_ <- F; while(isUnique_ == F){
                if(type_plot == "uncertainty"){
                  main_ <- letters[  total_counter  ]

                  # plot images with largest std's
                  sorted_unique_prob_k <- sort(rfxn(unique(ClusterProbs_std[,k_])),decreasing=T)
                  im_i <- which(ClusterProbs_std[,k_] == sorted_unique_prob_k[i+bad_counter])[1]
                }
                if(type_plot == "mean"){
                  main_ <- total_counter
                  # plot images with largest lower confidence
                  sorted_unique_prob_k <- sort(rfxn(unique(ClusterProbs_lower_conf[,k_])),decreasing=T)
                  im_i <- which(ClusterProbs_lower_conf[,k_] == sorted_unique_prob_k[i+bad_counter])[1]

                  # plot images with largest cluster probs
                  #sorted_unique_prob_k <- sort(rfxn(unique(ClusterProbs_est_full[,k_])),decreasing=T)
                  #im_i <- which(ClusterProbs_est_full[,k_] == sorted_unique_prob_k[i+bad_counter])[1]
                }

                coordinate_i <- c(long[im_i],lat[im_i])
                if(bad_counter>50){browser()}
                if(i > 1){
                  dist_m <- geosphere::distm(coordinate_i, used_coordinates, fun = geosphere::distHaversine)
                  bad_counter <- bad_counter + 1
                  if(all(dist_m >= 2000)){isUnique_ <- T}
                }
                if(i == 1){isUnique_<-T}
                print(sd_im <- sd(as.array(acquireImageFxn_full( keys = imageKeys[im_i],training = F )[1,,,]),na.rm=T))
                if(sd_im < .5){ bad_counter <- bad_counter + 1; isUnique_ <- F }
              }
              used_coordinates <- rbind(coordinate_i,used_coordinates)
              print(c(k_, i, im_i, long[im_i], lat[im_i]))
              if(is.na(sum(reNorm(as.array(acquireImageFxn_full( keys = imageKeys[im_i], training = F )[1,,,]))))){ browser() }
              rbgPlot <- try(raster::plotRGB( raster::brick( 0.0001 + reNorm(as.array(acquireImageFxn_full( keys = imageKeys[im_i],training = F )[1,,,])) ) ,
                               margins = T,
                               mar = (margins_vec <- (ep_<-1e-6)*c(1,3,1,1)),
                               main = main_,
                               cex.lab = 2.5,col.lab=k_,
                               xlab = sprintf("Long: %s, Lat: %s",
                                              fixZeroEndings(round(coordinate_i,2L)[1],2L),
                                              fixZeroEndings(round(coordinate_i,2L)[2],2L)),
                               col.main = k_, cex.main=4),T)
              if("try-error" %in% class(rbgPlot)){print("rbgPlot broken")}
              if(type_plot == "mean"){
                # axis for plot
                ylab_ <- ""; if(i==1){
                  tauk <- eval(parse(text = sprintf("tau%s",k_)))
                  ylab_ <- eval(parse(text = sprintf("expression(hat(tau)[%s]==%.3f)",k_,tauk)))
                  if(orthogonalize == T){
                    library(latex2exp)
                    ylab_ <- eval(parse(text = sprintf("expression(hat(tau)[%s]^{phantom() ~ symbol('\136') ~ phantom()}==%.3f)",k_,tauk)))
                  }
                  axis(side = 2,at=0.5,labels = ylab_,pos=-0.,tick=F,cex.axis=cex_tile_axis <- 4,
                       col.axis=k_)
                }

                #obtain image gradients
                {
                  take_k <- k_
                  if(i == 1){
                    ImageGrad_fxn <- (function(m){
                      m <- tf$Variable(m,trainable = T)
                      with(tf$GradientTape(watch_accessed_variables = F,persistent  = T) %as% tape, {
                        tape$watch( m )
                        PROBS_ <- tf$reduce_mean(tf$concat(
                          replicate(nMonte_salience, getClusterProb(m,training = F)),0L),0L)
                        PROBS_Smoothed <- tf$add(tf$multiply(tf$subtract(tf$constant(1), ep_LabelSmooth<-tf$constant(0.01)),PROBS_),
                                                tf$divide(ep_LabelSmooth,tf$constant(2)))
                        LOGIT_ <- tf$subtract(tf$math$log(PROBS_Smoothed),
                                              tf$math$log(tf$subtract(tf$constant(1), PROBS_Smoothed) ))
                      })
                      ImageGrad <- tape$jacobian( LOGIT_, m , experimental_use_pfor = F)
                      ImageGrad_o <- tf$gather(ImageGrad, indices = as.integer(take_k-1L), axis = 0L)
                      for(jf in 1:2){
                        if(jf == 1){ImageGrad <- tf$math$reduce_euclidean_norm(ImageGrad_o+0.0000001,3L,keepdims = T)}
                        if(jf == 2){ImageGrad <- tf$math$reduce_mean(ImageGrad_o,3L, keepdims = T)}
                        ImageGrad <- tf$gather(AveragingConv(ImageGrad),0L,axis = 0L)
                        if(jf == 1){ImageGrad_L2 <- ImageGrad}
                        if(jf == 2){ImageGrad_E <- ImageGrad}
                      }
                      return(tf$concat(list(ImageGrad_L2,ImageGrad_E),2L))
                    })
                    AveragingConv <- tf$keras$layers$Conv2D(filters=1L,
                                                            kernel_size = gradAnalysisFilterDim <- 10L,
                                                            padding = "valid")
                    AveragingConv( tf$expand_dims(tf$gather(acquireImageRepFxn(keys = imageKeys[im_i],training = F),1L, axis = 3L),3L)  )
                    AveragingConv$trainable_variables[[1]]$assign( 1 / gradAnalysisFilterDim^2 *tf$ones(tf$shape(AveragingConv$trainable_variables[[1]])) )
                  }
                  IG <- as.array( ImageGrad_fxn( acquireImageRepFxn(keys = imageKeys[im_i],training = F) ) )
                  summary( c(IG[,,1] ))
                  summary( c(IG[,,2] ))
                  nColors <- 1000
                  { #if(i == 1){
                    # pos/neg breaks should be on the same scale across observation
                    pos_breaks <- sort( quantile(c(IG[,,2][IG[,,2]>=0]),probs = seq(0,1,length.out=nColors/2),na.rm=T))
                    neg_breaks <- sort(quantile(c(IG[,,2][IG[,,2]<=0]),probs = seq(0,1,length.out=nColors/2),na.rm=T))
                    gradMag_breaks <- sort(quantile((c(IG[,,1])),probs = seq(0,1,length.out = nColors),na.rm=T))
                  }

                  # magnitude
                  print(summary(c( IG[,,1] )))
                  magPlot <- try(image(t(IG[,,1])[,nrow(IG[,,1]):1],
                            col = viridis::magma(nColors - 1),
                            breaks = gradMag_breaks, axes = F),T)
                  if("try-error" %in% class(magPlot)){print("magPlot broken")}
                  ylab_ <- ""; if(i==1){
                    axis(side = 2,at=0.5,labels = "Salience Magnitude",
                         pos=-0.,tick=F, cex.axis=3, col.axis=k_)
                  }

                  # direction
                  dirPlot <- try(image(t(IG[,,2])[,nrow(IG[,,2]):1],
                            col = c(hcl.colors(nColors/2 - 1,"reds"),
                                    hcl.colors(nColors/2 ,"blues")),
                            breaks = c(neg_breaks,pos_breaks), axes = F),T)
                  if("try-error" %in% class(dirPlot)){print("dirPlot broken")}
                  ylab_ <- ""; if(i==1){
                    axis(side = 2,at=0.5,labels = "Salience Direction",
                         pos=-0.,tick=F, cex.axis=3, col.axis=k_)
                  }
                }
              }
            }
            plotting_coordinates_mat <- rbind(plotting_coordinates_mat,used_coordinates)
            print(used_coordinates)
          }
        }
        dev.off()
        return( plotting_coordinates_mat )
      }
      plotting_coordinates_mat <- try(plot_fxn(),T)
      if("try-error" %in% class(plotting_coordinates_mat) ){ browser() }
    }
    par(mfrow=c(1,1))

    return( list("ClusterProbs_est" = ClusterProbs_est,
                 "tau1"=as.numeric(tau1),
                 "tau2"=as.numeric(tau2),
                 "impliedATE" = impliedATE,
                 "sd_tau1" = sd_tau1,
                 "sd_tau2" = sd_tau2,
                 "R2_Y0"=r2_y0_out,
                 "R2_Y1"=r2_y1_out,
                 "tau_i_est"=tau_i_est,
                 "y0_true_out" = y0_true,
                 "y1_true_out" = y1_true,
                 "y0_est_out" = y0_est,
                 "y1_est_out" = y1_est,
                 "cluster_prob_transport_distribution" = cluster_prob_transport_distribution,
                 "transportabilityMat" = transportabilityMat,
                 "ClusterProbs_lower_conf" = ClusterProbs_lower_conf,
                 "ClusterProbs_est_full" = ClusterProbs_est_full,
                 "ClusterProbs_std"=ClusterProbs_std,
                 "plotting_coordinates_mat"=plotting_coordinates_mat,
                 "negELL"=as.numeric(negELL),
                 "whichNA_dropped" = whichNA_dropped) )
  }
  if(SimMode == T){
    return(list("ClusterProbs_est" = ClusterProbs_est,
                "ClusterProbs" = ClusterProbs,
                "tau1" = as.numeric(tau1),
                "tau2" = as.numeric(tau2),
                "sd_tau1" = sd_tau1,
                "sd_tau2" = sd_tau2,
                "R2_Y0"=r2_y0_out,
                "R2_Y1"=r2_y1_out,
                "tau_i_est"=tau_i_est,
                "impliedATE" = impliedATE,
                "y0_true_out" = y0_true,
                "y1_true_out" = y1_true,
                "y0_est_out" = y0_est,
                "y1_est_out" = y1_est,
                "negELL"=as.numeric(negELL),
                "whichNA_dropped" = whichNA_dropped ))
  }
}
