#' Get moments for normalization 
#'
#' A function obtaining moments for normalization 
#'
#' @param iterator An iterator 
#' @param dataType A string denoting data type 
#' @param momentCalIters Number of minibatches with which to estimate moments 
#'
#' @return Returns  mean/sd normalization arrays. 
#'
#' @examples
#' # (Not run)
#' # GetMoments(iterator, dataType, image_dtype, momentCalIters = 34L)
#' @export
#' @md
#'
GetMoments <- function(iterator, dataType, image_dtype, momentCalIters = 34L){
  print2("Calibrating moments for input data normalization...")
  NORM_SD <- NORM_MEAN <- c(); for(momentCalIter in 1L:momentCalIters){
    # get a data batch 
    #ds_next_ <- try(iterator$`next`(),T) # depreciated 
    ds_next_ <- try(iterator$get_next(),T) 
    
    if(!"try-error" %in% class(ds_next_)){
      # setup normalizations
      ApplyAxis <- ifelse(dataType == "video", yes = 5, no = 4)
      
      # sanity check 
      # causalimages::image2( as.array(ds_next_train[[1]])[2,,,1] ) 
      
      # update normalizations
      NORM_SD <- rbind(NORM_SD, apply(as.array(ds_next_[[1]]),ApplyAxis,sd))
      NORM_MEAN <- rbind(NORM_MEAN, apply(as.array(ds_next_[[1]]),ApplyAxis,mean))
    }
  }
  NORM_SD <- apply(NORM_SD,2,median)
  NORM_MEAN <- apply(NORM_MEAN,2,mean)
  NORM_MEAN_array <- jnp$array(array(NORM_MEAN,dim=c(1,1,1,length(NORM_MEAN))),image_dtype)
  NORM_SD_array <- jnp$array(array(NORM_SD,dim=c(1,1,1,length(NORM_SD))),image_dtype)
  
  if(dataType == "video"){
    NORM_MEAN_array <- jnp$expand_dims(NORM_MEAN_array, 0L)
    NORM_SD_array <- jnp$expand_dims(NORM_SD_array, 0L)
  }
  
  return(list("NORM_MEAN_array"=NORM_MEAN_array,
              "NORM_SD_array"=NORM_SD_array, 
              "NORM_MEAN" = NORM_MEAN, 
              "NORM_SD" = NORM_SD))
}
