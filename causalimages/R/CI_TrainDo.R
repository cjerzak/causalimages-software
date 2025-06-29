#' Runs a trainer. 
#'
#' Runs trainers defined in TrainDefine(). 
#'
#' @param . No parameters. 
#' 
#' @return Performs training. 
#'
#' @section References:
#' \itemize{
#' \item Connor T. Jerzak, Fredrik Johansson, Adel Daoud. Image-based Treatment Effect Heterogeneity. Forthcoming in \emph{Proceedings of the Second Conference on Causal Learning and Reasoning (CLeaR), Proceedings of Machine Learning Research (PMLR)}, 2023.
#' }
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/causalimages-software/
#'
#' @import reticulate rrapply
#' @export
#' @md
TrainDo <- function(){
  par(mfrow=c(1,2))
  keys2indices_list <- tapply(1:length(imageKeysOfUnits), imageKeysOfUnits, c)
  GradNorm_vec <- loss_vec <- rep(NA,times=nSGD)
  keysUsedInTraining <- c();i_<-1L ; DoneUpdates <- 0L; for(i in i_:nSGD){
    t0 <- Sys.time(); if(i %% 5 == 0 | i == 1){gc(); cienv$py_gc$collect()}
    
    if(is.null(TFRecordControl)){ 
      # get next batch 
      ds_next_train <- ds_iterator_train$`next`()
      
      # if we run out of observations, reset iterator
      RestartedIterator <- F; if( is.null(ds_next_train) ){
        message("Re-setting iterator! (type 1)"); gc(); cienv$py_gc$collect()
        ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
        ds_next_train <-  ds_iterator_train$`next`(); gc();cienv$py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(dim(ds_next_train[[1]])[1] != batchSize){
        message("Re-setting iterator! (type 2)"); gc(); cienv$py_gc$collect()
        ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
        ds_next_train <-  ds_iterator_train$`next`(); gc(); cienv$py_gc$collect()
      } }
      
      # select batch indices based on keys
      batch_keys <- unlist(  lapply( p2l(ds_next_train[[3]]$numpy()), as.character) )
      batch_indices <- sapply(batch_keys,function(key_){ f2n( sample(as.character( keys2indices_list[[key_]] ), 1) ) })
      ds_next_train <- ds_next_train[[1]]
    }
    if(!is.null(TFRecordControl)){ 
      # get next batch 
      ds_next_train_control <- ds_iterator_train_control$`next`()
      
      # if we run out of observations, reset iterator
      RestartedIterator <- F; if( is.null(ds_next_train_control) ){
        message("Re-setting iterator! (type 1)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
        ds_next_train_control <-  ds_iterator_train_control$`next`(); gc();cienv$py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(dim(ds_next_train_control[[1]])[1] != batchSize){
        message("Re-setting iterator! (type 2)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
        ds_next_train_control <-  ds_iterator_train_control$`next`(); gc(); cienv$py_gc$collect()
      } }
      
      # get next batch 
      ds_next_train_treated <- ds_iterator_train_treated$`next`()
      
      # if we run out of observations, reset iterator
      RestartedIterator <- F; if( is.null(ds_next_train_treated) ){
        message("Re-setting iterator! (type 1)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_treated <- reticulate::as_iterator( tf_dataset_train_treated )
        ds_next_train_treated <-  ds_iterator_train_treated$`next`(); gc();cienv$py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(dim(ds_next_train_treated[[1]])[1] != batchSize){
        message("Re-setting iterator! (type 2)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_treated <- reticulate::as_iterator( tf_dataset_train_treated )
        ds_next_train_treated <-  ds_iterator_train_treated$`next`(); gc(); cienv$py_gc$collect()
      } }
      
      # select batch indices based on keys
      batch_keys <- c(unlist(  lapply( p2l(ds_next_train_control[[3]]$numpy()), as.character) ),
                      unlist(  lapply( p2l(ds_next_train_treated[[3]]$numpy()), as.character) ))
      batch_indices <- sapply(batch_keys,function(key_){ f2n( sample(as.character( keys2indices_list[[key_]] ), 1) ) })
      ds_next_train <- cienv$tf$concat(list(ds_next_train_control[[1]],
                                      ds_next_train_treated[[1]]), 0L)
    }
    if(any(!batch_indices %in% keysUsedInTraining)){ keysUsedInTraining <- c(keysUsedInTraining, batch_keys[!batch_keys %in% keysUsedInTraining]) }
    
    # training step
    t1 <- Sys.time()
    if(i == 1){ # initial forward pass in non-jitted mode for debugging 
      GetLoss(
        MPList[[1]]$cast_to_compute(ModelList), MPList[[1]]$cast_to_compute(ModelList_fixed), # model lists
        InitImageProcessFn(cienv$jnp$array(ds_next_train),  cienv$jax$random$PRNGKey(600L+i), inference = F), # m
        cienv$jnp$array(ifelse( !is.null(X), yes = list(X[batch_indices,]), no = list(1.))[[1]] , dtype = cienv$jnp$float16), # x
        cienv$jnp$array(as.matrix(obsW[batch_indices]), dtype = cienv$jnp$float16), # treat
        cienv$jnp$array(as.matrix(obsY[batch_indices]), dtype = cienv$jnp$float16), # y 
        cienv$jax$random$split(cienv$jax$random$PRNGKey( 500L+i ),length(batch_indices)),  # vseed for observations 
        StateList, # StateList
        cienv$jax$random$PRNGKey( 123L+i ), # seed
        MPList, # MPlist
        F) 
    }
    
    GradientUpdatePackage <- GradAndLossAndAux(
      MPList[[1]]$cast_to_compute(ModelList), MPList[[1]]$cast_to_compute(ModelList_fixed), # model lists
      InitImageProcessFn(cienv$jnp$array(ds_next_train),  cienv$jax$random$PRNGKey(600L+i), inference = F), # m
      cienv$jnp$array(ifelse( !is.null(X), yes = list(X[batch_indices,]), no = list(1.))[[1]], dtype = ComputeDtype), # x
      cienv$jnp$array(as.matrix(obsW[batch_indices]), dtype = ComputeDtype), # treat
      cienv$jnp$array(as.matrix(obsY[batch_indices]), dtype = ComputeDtype), # y 
      cienv$jax$random$split(cienv$jax$random$PRNGKey( 50L+i ),length(batch_indices)),  # vseed for observations 
      StateList, # StateList
      cienv$jax$random$PRNGKey( 123L+i ), # seed
      MPList, # MPlist
      F) # inference

    # perform gradient updates 
    {
      # get updated state
      StateList_tmp <- GradientUpdatePackage[[1]][[2]] # state
      
      # get loss + grad
      if(image_dtype_char == "float16"){ 
        loss_vec[i] <- myLoss_fromGrad <- cienv$np$array( MPList[[2]]$unscale( GradientUpdatePackage[[1]][[1]] ) )# value
      }
      if(image_dtype_char != "float16"){ 
        loss_vec[i] <- myLoss_fromGrad <- cienv$np$array( GradientUpdatePackage[[1]][[1]] )# value
      }
      GradientUpdatePackage <- GradientUpdatePackage[[2]] # grads
      GradientUpdatePackage <- cienv$eq$partition(GradientUpdatePackage, cienv$eq$is_inexact_array)
      GradientUpdatePackage_aux <- GradientUpdatePackage[[2]]; GradientUpdatePackage <- GradientUpdatePackage[[1]]
      
      # unscale + adjust loss scale is some non-finite or NA
      if(i == 1){
        Map2Zero <- cienv$eq$filter_jit(function(input){
          cienv$jax$tree$map(function(x){ cienv$jnp$where(cienv$jnp$isnan(x), cienv$jnp$array(0), x)}, input) })
        GetGetNorms <- cienv$eq$filter_jit(function(input){
          cienv$jax$tree$map(function(x){ cienv$jnp$mean(cienv$jnp$abs(x)) }, input) })
        AllFinite <- cienv$jax$jit( cienv$jmp$all_finite )
      }
      if(image_dtype_char == "float16"){ 
        GradientUpdatePackage <- Map2Zero( MPList[[2]]$unscale( GradientUpdatePackage ) )
      }
      AllFinite_DontAdjust <- AllFinite( GradientUpdatePackage )  & cienv$jnp$squeeze(cienv$jnp$array(!is.infinite(myLoss_fromGrad)))
      MPList[[2]] <- MPList[[2]]$adjust( AllFinite_DontAdjust  )
      # which(is.na( c(unlist(lapply(cienv$jax$tree$leaves(myGrad_jax), function(zer){cienv$np$array(zer)}))) ) )
      # which(is.infinite( c(unlist(lapply(cienv$jax$tree$leaves(myGrad_jax), function(zer){cienv$np$array(zer)}))) ) )
      
      # get update norm 
      GradNorm_vec[i] <- mean( GradVec <- unlist( lapply(cienv$jax$tree$leaves(GradientUpdatePackage),
                                                         function(zer){ cienv$np$array(cienv$jnp$mean(cienv$jnp$abs(zer) )) }) )  )
      
      # update parameters if finite gradients
      DoUpdate <- !is.na(myLoss_fromGrad) & cienv$np$array(AllFinite_DontAdjust) & 
        !is.infinite(myLoss_fromGrad) & ( GradNorm_vec[i] > 1e-10)
      if(! DoUpdate ){ message("Warning: Not updating parameters due to NA, zero, or non-finite gradients in mixed-precision training...") }
      if( DoUpdate ){
        DoneUpdates <- DoneUpdates + 1
        
        # cast updates to param 
        GradientUpdatePackage <- MPList[[1]]$cast_to_param( GradientUpdatePackage )
        
        # get gradient updates 
        GradientUpdatePackage <- jit_get_update( 
          updates = GradientUpdatePackage,
          state = opt_state,
          params = (cienv$eq$partition(ModelList, cienv$eq$is_array)[[1]] )
        )
        
        # separate updates from state
        opt_state <- GradientUpdatePackage[[2]]
        GradientUpdatePackage <- cienv$eq$combine(GradientUpdatePackage[[1]], 
                                            GradientUpdatePackage_aux)
        
        # perform updates
        #ModelList_tminus1 <- ModelList
        ModelList <- cienv$eq$combine( jit_apply_updates(
          params = cienv$eq$partition(ModelList, cienv$eq$is_array)[[1]],
          updates = GradientUpdatePackage),
          cienv$eq$partition(ModelList, cienv$eq$is_array)[[2]])
        StateList <- StateList_tmp
        if(FALSE){
          LayerWiseParamDiff <- function(params_new, params_old){
            diff_fn <- function(new, old){
              cienv$jnp$mean(cienv$jnp$abs(new - old))
            }
            diff_list <- cienv$jax$tree$map(diff_fn, params_new, params_old)
            rrapply::rrapply(diff_list, how = "flatten")
          }
          
          # Use the function right after parameters update:
          param_diffs <- LayerWiseParamDiff(
            cienv$eq$partition(ModelList, cienv$eq$is_array)[[1]],
            cienv$eq$partition(ModelList_tminus1, cienv$eq$is_array)[[1]]
          )
          GradientUpdatePackage
        }
        suppressWarnings( rm(StateList_tmp, GradientUpdatePackage,BNInfo) )
      }
      i_ <- i ; if( (i %% 25 == 0 | i < 10) & 
                    (length(loss_vec[!is.na(loss_vec) & !is.infinite(loss_vec)]) > 5) ){
        message(sprintf("SGD iteration %s of %s -- Loss: %.2f (%.1f%%) --
                           Total iter time (s): %.2f - Grad iter time (s): %.2f --
                           Grad norm: %.3f -- Grads zero %%: %.1f%% --
                           %.3f tstat on log(iter)",
                       i,  nSGD, loss_vec[i], 100*mean(loss_vec[i] <= loss_vec[1:i],na.rm=T),
                       (Sys.time() - t0)[[1]], (Sys.time() - t1)[[1]],
                       mean(GradVec,na.rm=T), 100*mean(GradVec==0,na.rm=T),
                       coef(summary(lm(loss_vec[1:i]~log(1:i))))[2,3]
                       ) )
        loss_vec <- f2n(loss_vec); loss_vec[is.infinite(loss_vec)] <- NA
        plot( (na.omit(loss_vec)), cex.main = 0.95,ylab = "Loss Function",xlab="SGD Iteration Number")
        if(length(na.omit(loss_vec)) > 10){ points(smooth.spline( (na.omit(loss_vec) ),spar=1,cv=TRUE), col="red",type = "l",lwd=5) }
        plot(GradNorm_vec[!is.infinite(GradNorm_vec) & !is.na(GradNorm_vec)], cex.main = 0.95,ylab = "GradNorm",xlab="SGD Iteration Number")
      }
    }
  } # end for(i in i_:nSGD){
  par(mfrow=c(1,1))
}

