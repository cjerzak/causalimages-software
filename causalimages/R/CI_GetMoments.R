#' Get moments for normalization (internal function)
#'
#' An internal function for obtaining moments for channel normalization.
#'
#' @param iterator An iterator
#' @param dataType A string denoting data type
#' @param image_dtype A string specifying the image data type (e.g., "float16", "float32")
#' @param momentCalIters Number of minibatches with which to estimate moments
#'
#' @return Returns mean/sd arrays for normalization. 
#'
#' @examples
#' # (Not run)
#' # GetMoments(iterator, dataType, image_dtype, momentCalIters = 34L)
#' @export
#' @md
#'
GetMoments <- function(iterator, dataType, image_dtype, momentCalIters = 34L){
  message2("Calibrating moments for input data normalization...")
  NORM_SD <- NORM_MEAN <- c(); for(momentCalIter in 1L:momentCalIters){
    # get a data batch 
    ds_next_ <- try(iterator$get_next(),T) 
    
    if(!"try-error" %in% class(ds_next_)){
      # setup normalizations
      ApplyAxis <- ifelse(dataType == "video", yes = 5, no = 4)
      
      # sanity check 
      # causalimages::image2( cienv$np$array((ds_next_train[[1]])[2,,,1] ) 
      
      # update normalizations
      NORM_SD <- rbind(NORM_SD, apply(cienv$np$array(ds_next_[[1]]),ApplyAxis,sd))
      NORM_MEAN <- rbind(NORM_MEAN, apply(cienv$np$array(ds_next_[[1]]),ApplyAxis,mean))
    }
  }

  # mean calc 
  NORM_MEAN_mat <- NORM_MEAN      # same shape
  NORM_MEAN <- apply(NORM_MEAN,2,mean) # overall mean across all batches
  NORM_MEAN_array <- cienv$jnp$array(array(NORM_MEAN,dim=c(1,1,1,length(NORM_MEAN))),image_dtype)
  
  # SD calc using Rubin’s rule: combine within‐ and between‐batch variances
  NORM_SD_mat   <- NORM_SD        # matrix: rows = batches, cols = features
  
  #combine information to get 
  m <- nrow(NORM_SD_mat)
  W <- colMeans(NORM_SD_mat^2)        # average within‐batch variance
  B <- apply(NORM_MEAN_mat,2,var)     # variance of the batch means
  T_var <- W + (1 + 1/m) * B          # total variance
  NORM_SD <- sqrt(T_var)              # combined SD
  # plot(apply(NORM_SD_mat,2,median),NORM_SD);abline(a=0,b=1)
  if("try-error" %in% class(NORM_SD)){browser()}
  NORM_SD_array <- cienv$jnp$array(array(NORM_SD,dim=c(1,1,1,length(NORM_SD))),image_dtype)

  if(dataType == "video"){
    NORM_MEAN_array <- cienv$jnp$expand_dims(NORM_MEAN_array, 0L)
    NORM_SD_array <- cienv$jnp$expand_dims(NORM_SD_array, 0L)
  }
  
  return(list("NORM_MEAN_array"=NORM_MEAN_array,
              "NORM_SD_array"=NORM_SD_array, 
              "NORM_MEAN" = NORM_MEAN, 
              "NORM_SD" = NORM_SD))
}
