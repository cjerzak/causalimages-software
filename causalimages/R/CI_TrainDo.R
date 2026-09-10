#' Runs a training routine (internal function)
#'
#' Runs trainers defined in TrainDefine(). Internal function.
#' This function takes no parameters and is called within parent functions
#' to execute the training loop.
#'
#' @return Internal function performing model training. 
#'
#' @import reticulate rrapply
#' @noRd
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
      RestartedIterator <- FALSE; if( is.null(ds_next_train) ){
        message2("Re-setting iterator! (type 1)"); gc(); cienv$py_gc$collect()
        ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
        ds_next_train <-  ds_iterator_train$`next`(); gc();cienv$py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(as.integer(ds_next_train[[1]]$shape[[0]]) != batchSize){
        message2("Re-setting iterator! (type 2)"); gc(); cienv$py_gc$collect()
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
      RestartedIterator <- FALSE; if( is.null(ds_next_train_control) ){
        message2("Re-setting iterator! (type 1)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
        ds_next_train_control <-  ds_iterator_train_control$`next`(); gc();cienv$py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(as.integer(ds_next_train_control[[1]]$shape[[0]]) != batchSize){
        message2("Re-setting iterator! (type 2)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
        ds_next_train_control <-  ds_iterator_train_control$`next`(); gc(); cienv$py_gc$collect()
      } }
      
      # get next batch 
      ds_next_train_treated <- ds_iterator_train_treated$`next`()
      
      # if we run out of observations, reset iterator
      RestartedIterator <- F; if( is.null(ds_next_train_treated) ){
        message2("Re-setting iterator! (type 1)"); gc(); cienv$py_gc$collect()
        ds_iterator_train_treated <- reticulate::as_iterator( tf_dataset_train_treated )
        ds_next_train_treated <-  ds_iterator_train_treated$`next`(); gc();cienv$py_gc$collect()
      }
      
      # get a new batch if size mismatch - size mismatches generate new cached compiled fxns
      if(!RestartedIterator){ if(as.integer(ds_next_train_treated[[1]]$shape[[0]]) != batchSize){
        message2("Re-setting iterator! (type 2)"); gc(); cienv$py_gc$collect()
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
    if(any(!batch_indices %in% keysUsedInTraining)){ 
      keysUsedInTraining <- c(keysUsedInTraining, batch_keys[!batch_keys %in% keysUsedInTraining])
    }
    
    # if no treat, define it (unused in GetLoss)
    if(!"obsW" %in% ls()){ obsW <- obsY }
    
    # training step
    if(!justCheckIterators){ 
    t1 <- Sys.time()
    # sanity check 
    if(FALSE){ 
      test_index <- 2
      GetElementFromTfRecordAtIndices(
        uniqueKeyIndices = which(unique(imageKeysOfUnits)==unique(imageKeysOfUnits)[test_index]),
        filename = file,
        readVideo = useVideoIndicator,
        nObs = length(unique(imageKeysOfUnits) ) )
      # unique(imageKeysOfUnits)[test_index]
    }

    # Sanity check for dimension swapping as i varies 
    if(i == 1){ 
      message2(sprintf("Training balance: %s",
                       paste(paste(names(table(obsW[batch_indices])),
                             table(obsW[batch_indices]), sep = " has "),collapse="; ")
                       ))
      if(any(prop.table(table(obsW[batch_indices])) > 0.9) & 
                !is.null(TFRecordControl)){
        stop(  "Stopping - Balanced training not satisfied despite TFRecordControl being defined!"  ) 
      }
    }
    # causalimages::image2(cienv$np$array(InitImageProcessFn(cienv$jnp$array(ds_next_train),  cienv$jax$random$key(600L+sample(1:100,1)), inference = F)[2,,,1]))
    # causalimages::image2(cienv$np$array(InitImageProcessFn(cienv$jnp$array(ds_next_train),  cienv$jax$random$key(600L+sample(1:100,1)), inference = F)[1,,,1]))

    # A single compiled update owns gradients, finite checks and optimizer state.
    step_result <- TrainStep$update(
      training_state, ModelList_fixed,
      InitImageProcessFn(cienv$jnp$array(ds_next_train), cienv$jax$random$key(600L+i), inference = FALSE),
      cienv$jnp$array(if (is.null(X)) matrix(0, length(batch_indices), 1L) else X[batch_indices, , drop = FALSE], dtype = ComputeDtype),
      cienv$jnp$array(as.matrix(obsW[batch_indices]), dtype = ComputeDtype),
      cienv$jnp$array(as.matrix(obsY[batch_indices]), dtype = ComputeDtype),
      cienv$jax$random$split(cienv$jax$random$key(50L+i), length(batch_indices)),
      MPList[[1]]
    )
    training_state <- step_result[[1]]
    metrics <- as.numeric(step_result[[2]])
    rm(step_result)
    loss_vec[i] <- myLoss_fromGrad <- metrics[1]
    GradNorm_vec[i] <- metrics[2]
    GradZeroFraction <- metrics[3]
    DoUpdate <- isTRUE(metrics[4] == 1)
    if (DoUpdate) DoneUpdates <- DoneUpdates + 1L else
      message2("Warning: Not updating parameters due to non-finite or zero gradients/loss")
    {
      i_ <- i ; if( (i %% 25 == 0 | i < 10) & 
                    (length(loss_vec[!is.na(loss_vec) & !is.infinite(loss_vec)]) > 5) ){
        loss_vec_ <- loss_vec
        loss_vec_[is.infinite(loss_vec_)] <- NA
        message2(sprintf("SGD iteration %s of %s -- Loss: %.2f (%.1f%%) --
                           Total iter time (s): %.2f - Grad iter time (s): %.2f --
                           Grad norm: %.3f -- Grads zero %%: %.1f%% --
                           %.3f tstat on log(iter)",
                       i,  nSGD, loss_vec[i], 100*mean(loss_vec[i] <= loss_vec[1:i],na.rm=T),
                       (Sys.time() - t0)[[1]], (Sys.time() - t1)[[1]],
                       GradNorm_vec[i], 100*GradZeroFraction,
                       ifelse("try-error" %in% class(tstat_ <- try(coef(summary(lm(loss_vec[1:i]~log(1:i))))[2,3], T)),yes = NA, no = tstat_)
                       ) )
        loss_vec <- f2n(loss_vec); loss_vec[is.infinite(loss_vec)] <- NA
        plot( (na.omit(loss_vec)), cex.main = 0.95,ylab = "Loss Function",xlab="SGD Iteration Number")
        if(length(na.omit(loss_vec)) > 10){ points(smooth.spline( (na.omit(loss_vec) ),spar=1,cv=TRUE), col="red",type = "l",lwd=5) }
        plot(GradNorm_vec[!is.infinite(GradNorm_vec) & !is.na(GradNorm_vec)], cex.main = 0.95,ylab = "GradNorm",xlab="SGD Iteration Number")
      }
      
      # Early stopping 
      
      if( !is.null(earlyStopThreshold) ){ 
        window <- 25
        patience_limit <- 25
        if(!"patience_counter" %in% ls()){ patience_counter <- 0 }
        if( i > 2*window & i > 100 ){
          first_avg <- mean(loss_vec[1:10], na.rm = TRUE)
          prev_avg <- mean(loss_vec[(i-2*window):(i-window-1)], na.rm = TRUE)
          curr_avg <- mean(loss_vec[(i-window):i], na.rm = TRUE)
          
          se_diff <- sqrt( var(loss_vec[(i-2*window):(i-window-1)], na.rm=TRUE)/window +
                             var(loss_vec[(i-window):i], na.rm=TRUE)/window )
          prev_avg_upper <- curr_avg + (t_es<-2.528)*sqrt( var(loss_vec[(i-2*window):(i-window-1)], na.rm=TRUE)/window )
          curr_avg_lower <- prev_avg - t_es*sqrt( var(loss_vec[(i-window):i], na.rm=TRUE)/window )
          
          if( curr_avg >= prev_avg - t_es*se_diff & curr_avg < 0.8*first_avg ){
            message2("We fail to detect evidence of improvement, early stopping being considered...") 
            patience_counter <- patience_counter + 1
            if(patience_counter >= patience_limit){
              message2("Early stopping triggered - No more meaningful improvement.")
              break
            }
          } else {
            patience_counter <- 0  # reset when any improvement detected
          }
        } 
      }
    }
    }
  } # end for(i in i_:nSGD){
  ModelList <- training_state$model
  StateList <- training_state$model_state
  MPList[[2]] <- training_state$loss_scale
  # Inference needs model state, but not AdaBelief's two parameter-sized slots.
  rm(training_state, TrainStep, optax_optimizer)
  gc(); cienv$py_gc$collect()
  par(mfrow=c(1,1))
}
