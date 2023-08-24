#!/usr/bin/env Rscript
#' Generates randomized image and video embeddings useful in earth observation tasks for casual inference, generalizing the approach in Rolf, Esther, et al.  (2021).
#'
#' Generates randomized image and video embeddings useful in earth observation tasks for casual inference, generalizing the approach in Rolf, Esther, et al.  (2021).
#'
#' @usage
#'
#' GetRandomizedImageEmbeddings(imageKeysOfUnits, acquireImageFxn, nFeatures, ...)
#'
#' @param acquireImageFxn A function specifying how to load images representations associated with `imageKeysOfUnits` into memory. For example, if observation `3` has a value  of `"a34f"` in `imageKeysOfUnits`, `acquireImageFxn` should extract the image associated with the unique key `"a34f"`.
#' First argument should be image key values and second argument have be `training` (in case different behavior in training/inference mode).
#' @param imageKeysOfUnits A vector of length `length(imageKeysOfUnits)` specifying the unique image ID associated with each unit. Samples of `imageKeysOfUnits` are fed into `acquireImageFxn` to call images into memory.
#' @param conda_env (default = `NULL`) A string specifying a conda environment wherein `tensorflow`, `tensorflow_probability`, and `gc` are installed.
#' @param conda_env_required (default = `F`) A Boolean stating whether use of the specified conda environment is required.
#' @param kernelSize (default = `5L`) Dimensions used in the convolution kernels.
#' @param temporalKernelSize (default = `2L`) Dimensions used in the temporal part of the convolution kernels if using image sequences.
#' @param nFeatures (default = `256L`) Dimensions used in the convolution kernels.
#' @param strides (default = `2L`) Integer specifying the strides used in the convolutional layers.
#' @param batchSize (default = `50L`) Integer specifying batch size in obtaining embeddings.
#' @param seed (default = `NULL`) Integer specifying the seed.
#' @param outputType (default = `"R"`) Either `"R"` or `"tensorflow"` indicating whether to output R or tensorflow arrays.
#'
#' @return A list containing two items:
#' \itemize{
#' \item `embeddings` (matrix) A matrix containingimage/video embeddings, with rows corresponding to observations.
#' \item `embeddings_fxn` (function) The functioning performing the embedding, returned for re-use.
#' }
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/causalimages-software/
#'  @section References:
#' \itemize{
#' \item Rolf, Esther, et al. "A generalizable and accessible approach to machine learning with global satellite imagery." *Nature Communications* 12.1 (2021): 4392.
#' }
#'
#' @export
#' @md

GetRandomizedImageEmbeddings <- function(
    acquireImageFxn  ,
    imageKeysOfUnits,
    conda_env = NULL,
    conda_env_required = F,

    nFeatures = 256L,
    batchSize = 50L,
    strides = 1L,
    temporalKernelSize = 2L,
    kernelSize = 3L,
    compile = T,
    seed = NULL,
    quiet = F){

  if(   "try-error" %in% class(try(tf$constant(1.),T))   ){
    print("Initializing the tensorflow environment...")
    print("Looking for Python modules tensorflow, gc...")
    library(tensorflow); library(keras)
    try(tensorflow::use_condaenv(conda_env, required = conda_env_required),T)
    Sys.sleep(1.); try(tf$square(1.),T); Sys.sleep(1.)
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]],T),T)
    try( tf$config$set_soft_device_placement( T ) , T)
    #tfd <- (tfp <- tf_probability())$distributions
    #tfa <- reticulate::import("tensorflow_addons")

    try(tf$random$set_seed(  c( ifelse(is.null(tf_seed),
                                       yes = 123431L, no = as.integer(tf_seed)  ) )), T)
    try(tf$keras$utils$set_random_seed( c( ifelse(is.null(tf_seed),
                                                  yes = 123419L, no = as.integer(tf_seed)  ) )), T)

    # import python garbage collectors
    py_gc <- reticulate::import("gc")
  }
  gc(); try(py_gc$collect(), T)

  if(batchSize > length(imageKeysOfUnits)){
    batchSize <- length( imageKeysOfUnits  )
  }

  myType <- acquireImageFxn(imageKeysOfUnits[1:2], training = F)

  # coerce output to tf$constant
  environment(acquireImageFxn) <- environment()
  test_ <- acquireImageFxn(imageKeysOfUnits[1:5],training = F)
  if(!"tensorflow.tensor" %in% class(test_)){
    acquireImageFxn_as_input <- acquireImageFxn
    acquireImageFxn <- function(keys, training){
      m_ <- tf$constant(acquireImageFxn_as_input(keys, training),tf$float32)
      if(length(m_$shape) == 3){
        # expand across batch dimension if receiving no batch dimension
        m_ <- tf$expand_dims(m_,0L)
      }
      return( m_ )
    }
  }

  imageDims <- length( dim(test_) ) - 2L

  if(imageDims == 2){
    myConv = tf$keras$layers$Conv2D(filters=round(nFeatures),
                          kernel_size=c(kernelSize,kernelSize),
                          activation="linear",
                          strides = c(strides,strides),
                          padding = "valid")
    GlobalMaxPoolLayer <- tf$keras$layers$GlobalMaxPool2D(data_format="channels_last",name="GlobalMax")
    #GlobalAvePoolLayer <- tf$keras$layers$GlobalAveragePooling2D(data_format="channels_last",name="GlobalAve")
  }
  if(imageDims == 3){
    myConv = tf$keras$layers$Conv3D(filters=round(nFeatures),
                                    kernel_size=c(temporalKernelSize, kernelSize,kernelSize),
                                    activation="linear",
                                    strides = c(1L,strides,strides),
                                    padding = "valid")
    GlobalMaxPoolLayer <- tf$keras$layers$GlobalMaxPool3D(data_format="channels_last",name="GlobalMax")
    #GlobalAvePoolLayer <- tf$keras$layers$GlobalAveragePooling3D(data_format="channels_last",name="GlobalAve")
  }

  #GlobalPoolLayer <- function(z){return(tf$concat(list(GlobalMaxPoolLayer(z),GlobalAvePoolLayer(z)),1L)) }
  GlobalPoolLayer <- function(z){return(GlobalMaxPoolLayer(z)) }

  getEmbedding <- tf_function(function(im_){
    im_ <- GlobalMaxPoolLayer ( myConv( im_ ) )
    return( im_  )
  } )

  embeddings <- matrix(NA,nrow = length(imageKeysOfUnits), ncol = nFeatures)
  last_i <- 0; ok_counter <- 0; ok<-F; while(!ok){
    ok_counter <- ok_counter + 1
    print(sprintf("[%s] %.2f%% done with getting randomized embeddings", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), 100*last_i / length(imageKeysOfUnits)))

    # in functional mode
    {
      batch_indices_inference <- (last_i+1):(last_i+batchSize)
      batch_indices_inference <- batch_indices_inference[batch_indices_inference <= length(imageKeysOfUnits)]
      last_i <- batch_indices_inference[length(batch_indices_inference)]
      if(last_i == length(imageKeysOfUnits)){ ok <- T }

      batchSizeOneCorrection <- F; if(length(batch_indices_inference) == 1){
        batch_indices_inference <- c(batch_indices_inference,batch_indices_inference)
        batchSizeOneCorrection <- T
      }

      batch_inference <- list(
        tf$cast(acquireImageFxn(imageKeysOfUnits[batch_indices_inference], training = F),tf$float32)
      )

      embed_ <- try(as.matrix( getEmbedding(   batch_inference[[1]]  )  ), T)
      if("try-error" %in% class(embed_)){ browser() }
      if(batchSizeOneCorrection){ batch_indices_inference <- batch_indices_inference[-1]; embed_ <- embed_[1,] }
      embeddings[batch_indices_inference,] <- embed_
    }
    gc(); try(py_gc$collect(), T)
  }
  print(sprintf("[%s] %.2f%% done with getting randomized embeddings", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), 100*1))


   return( list( "embeddings"= embeddings,
                 "embeddings_fxn" = getEmbedding ) )
}
