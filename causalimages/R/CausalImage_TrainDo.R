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
  keys2indices_list <- tapply(1:length(imageKeysOfUnits), imageKeysOfUnits, c)
  GradNorm_vec <- loss_vec <- rep(NA,times=nSGD)
  keysUsedInTraining <- c();i_<-1L ; DoneUpdates <- 0L; for(i in i_:nSGD){
    t0 <- Sys.time(); if(i %% 5 == 0 | i == 1){gc(); py_gc$collect()}
    
    if(is.null(TFRecordControl)){ 
      # get next batch 
      ds_next_train <- ds_iterator_train$`next`()
      
      # if we run out of observations, reset iterator
      RestartedIterator <- F; if( is.null(ds_next_train) ){
        print2("Re-setting iterator! (type 1)"); gc(); py_gc$collect()
        ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
        ds_next_train <-  ds_iterator_train$`next`(); gc();py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(dim(ds_next_train[[1]])[1] != batchSize){
        print2("Re-setting iterator! (type 2)"); gc(); py_gc$collect()
        ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
        ds_next_train <-  ds_iterator_train$`next`(); gc(); py_gc$collect()
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
        print2("Re-setting iterator! (type 1)"); gc(); py_gc$collect()
        ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
        ds_next_train_control <-  ds_iterator_train_control$`next`(); gc();py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(dim(ds_next_train_control[[1]])[1] != batchSize){
        print2("Re-setting iterator! (type 2)"); gc(); py_gc$collect()
        ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
        ds_next_train_control <-  ds_iterator_train_control$`next`(); gc(); py_gc$collect()
      } }
      
      # get next batch 
      ds_next_train_treated <- ds_iterator_train_treated$`next`()
      
      # if we run out of observations, reset iterator
      RestartedIterator <- F; if( is.null(ds_next_train_treated) ){
        print2("Re-setting iterator! (type 1)"); gc(); py_gc$collect()
        ds_iterator_train_treated <- reticulate::as_iterator( tf_dataset_train_treated )
        ds_next_train_treated <-  ds_iterator_train_treated$`next`(); gc();py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(dim(ds_next_train_treated[[1]])[1] != batchSize){
        print2("Re-setting iterator! (type 2)"); gc(); py_gc$collect()
        ds_iterator_train_treated <- reticulate::as_iterator( tf_dataset_train_treated )
        ds_next_train_treated <-  ds_iterator_train_treated$`next`(); gc(); py_gc$collect()
      } }
      
      # select batch indices based on keys
      batch_keys <- c(unlist(  lapply( p2l(ds_next_train_control[[3]]$numpy()), as.character) ),
                      unlist(  lapply( p2l(ds_next_train_treated[[3]]$numpy()), as.character) ))
      batch_indices <- sapply(batch_keys,function(key_){ f2n( sample(as.character( keys2indices_list[[key_]] ), 1) ) })
      ds_next_train <- tf$concat(list(ds_next_train_control[[1]],
                                      ds_next_train_treated[[1]]), 0L)
    }
    if(any(!batch_indices %in% keysUsedInTraining)){ keysUsedInTraining <- c(keysUsedInTraining, batch_keys[!batch_keys %in% keysUsedInTraining]) }
    
    # training step
    # rm(GradAndLossAndAux); GradAndLossAndAux <-  eq$filter_jit( eq$filter_value_and_grad( GetLoss, has_aux = T) )
    t1 <- Sys.time()
    GradientUpdatePackage <- GradAndLossAndAux(
      MPList[[1]]$cast_to_compute(ModelList), MPList[[1]]$cast_to_compute(ModelList_fixed),
      #ModelList, ModelList_fixed,
      InitImageProcessFn(jnp$array(ds_next_train),  jax$random$PRNGKey(600L+i), inference = F), # m
      jnp$array(X[batch_indices,], dtype = jnp$float16), # x
      jnp$array(obsW[batch_indices], dtype = jnp$float16), # treat
      jax$random$split(jax$random$PRNGKey( 500L+i ),length(batch_indices)),  # vseed
      StateList, # StateList
      jax$random$PRNGKey( 123L+i ), # seed
      MPList, # MPlist
      F) # inference
    
    # perform gradient updates 
    {
      # get updated state
      StateList_tmp <- GradientUpdatePackage[[1]][[2]] # state
      
      # get loss + grad
      if(image_dtype_char == "float16"){ 
        loss_vec[i] <- myLoss_fromGrad <- np$array( MPList[[2]]$unscale( GradientUpdatePackage[[1]][[1]] ) )# value
      }
      if(image_dtype_char != "float16"){ 
        loss_vec[i] <- myLoss_fromGrad <- np$array( GradientUpdatePackage[[1]][[1]] )# value
      }
      GradientUpdatePackage <- GradientUpdatePackage[[2]] # grads
      GradientUpdatePackage <- eq$partition(GradientUpdatePackage, eq$is_inexact_array)
      GradientUpdatePackage_aux <- GradientUpdatePackage[[2]]; GradientUpdatePackage <- GradientUpdatePackage[[1]]
      
      # unscale + adjust loss scale is some non-finite or NA
      if(i == 1){
        Map2Zero <- eq$filter_jit(function(input){
          jax$tree_map(function(x){ jnp$where(jnp$isnan(x), jnp$array(0), x)}, input) })
        GetGetNorms <- eq$filter_jit(function(input){
          jax$tree_map(function(x){ jnp$mean(jnp$abs(x)) }, input) })
        AllFinite <- jax$jit( jmp$all_finite )
      }
      if(image_dtype_char == "float16"){ 
        GradientUpdatePackage <- Map2Zero( MPList[[2]]$unscale( GradientUpdatePackage ) )
      }
      AllFinite_DontAdjust <- AllFinite( GradientUpdatePackage )  & jnp$squeeze(jnp$array(!is.infinite(myLoss_fromGrad)))
      MPList[[2]] <- MPList[[2]]$adjust( AllFinite_DontAdjust  )
      # which(is.na( c(unlist(lapply(jax$tree_leaves(myGrad_jax), function(zer){np$array(zer)}))) ) )
      # which(is.infinite( c(unlist(lapply(jax$tree_leaves(myGrad_jax), function(zer){np$array(zer)}))) ) )
      
      # get update norm 
      GradNorm_vec[i] <- mean( GradVec <- unlist( lapply(jax$tree_leaves(GradientUpdatePackage),
                                                         function(zer){ np$array(jnp$mean(jnp$abs(zer) )) }) )  )
      
      # update parameters if finite gradients
      DoUpdate <- !is.na(myLoss_fromGrad) & np$array(AllFinite_DontAdjust) & 
        !is.infinite(myLoss_fromGrad) & ( GradNorm_vec[i] > 1e-10)
      if(! DoUpdate ){ print2("Warning: Not updating parameters due to zero or non-finite gradients in mixed-precision training...") }
      if( DoUpdate ){
        DoneUpdates <- DoneUpdates + 1
        
        # cast updates to param 
        GradientUpdatePackage <- MPList[[1]]$cast_to_param( GradientUpdatePackage )
        
        
        # get gradient updates 
        BNInfo <- FilterBN( ModelList )[[2]]
        GradientUpdatePackage <- jit_get_update( 
          updates = FilterBN(GradientUpdatePackage)[[1]],
          state = optax_optimizer$init(   FilterBN(eq$partition(ModelList, eq$is_array)[[1]] )[[1]] ) ,
          params = FilterBN(eq$partition(ModelList, eq$is_array)[[1]] )[[1]])
        #GradientUpdatePackage <- jit_get_update( updates = GradientUpdatePackage, state = opt_state, params = eq$partition(ModelList, eq$is_array)[[1]] )
        
        # separate updates from state
        opt_state <- GradientUpdatePackage[[2]]
        GradientUpdatePackage <- eq$combine(GradientUpdatePackage[[1]], 
                                            GradientUpdatePackage_aux)
        
        # perform updates
        ModelList <- eq$combine( jit_apply_updates(
          #params = GlobalPartition(eq$partition(ModelList, eq$is_array)[[1]], PartFxn)[[1]],
          params = FilterBN(eq$partition(ModelList, eq$is_array)[[1]])[[1]],
          updates = GradientUpdatePackage),
          eq$partition(ModelList, eq$is_array)[[2]])
        ModelList <- eq$combine(ModelList,BNInfo)
        StateList <- StateList_tmp
        suppressWarnings( rm(StateList_tmp, GradientUpdatePackage,BNInfo) )
      }
      i_ <- i ; if(i %% 10 == 0 | i < 10 ){
        print2(sprintf("SGD iteration %s of %s - Loss: %.2f (%.1f%%) - - Total iter time (s): %.2f - Grad iter time (s): %.2f",
                       i,  nSGD, loss_vec[i], 100*mean(loss_vec[i] <= loss_vec[1:i],na.rm=T),
                       (Sys.time() - t0)[[1]], (Sys.time() - t1)[[1]] ) )
        loss_vec <- f2n(loss_vec); loss_vec[is.infinite(loss_vec)] <- NA
        par(mfrow=c(1,2))
        try(plot(rank(na.omit(loss_vec)), cex.main = 0.95,ylab = "Loss Function Rank",xlab="SGD Iteration Number"), T)
        if(i > 10){ try_ <- try(points(smooth.spline( rank(na.omit(loss_vec) )), col="red",type = "l",lwd=5),T) }
        try(plot(na.omit(GradNorm_vec), cex.main = 0.95,ylab = "GradNorm",xlab="SGD Iteration Number"), T)
      }
    }
  } # end for(i in i_:nSGD){
}

