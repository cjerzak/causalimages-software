#' Defines an internal training routine. 
#'
#' Defines trainers defined in TrainDefine(). Internal function. 
#'
#' @param . No parameters. 
#' 
#' @return Internal function defining a training sequence. 
#'
#' @import reticulate rrapply
#' @export
#' @md
TrainDefine <- function(){
  message2("Define optimizer and training step...") 
  {
    LR_schedule <- cienv$optax$warmup_cosine_decay_schedule(warmup_steps = (nWarmup <- min(c(max(0.25*nSGD,100L), nSGD))),
                                                      decay_steps = max(c(101L, nSGD-nWarmup)),
                                                      init_value = learningRateMax/100, 
                                                      peak_value = learningRateMax, 
                                                      end_value =  learningRateMax/100)
    plot(cienv$np$array(LR_schedule(cienv$jnp$array(1:nSGD))),xlab = "Iteration", ylab="Learning rate schedule")
    optax_optimizer <-  cienv$optax$chain(
      cienv$optax$adaptive_grad_clip(clipping = 0.15, eps = 0.001),
      cienv$optax$adabelief( learning_rate = LR_schedule )
      #cienv$optax$adam( learning_rate = LR_schedule )
    )
    
    # model partition, setup state, perform parameter count
    opt_state <- optax_optimizer$init(   cienv$eq$partition(ModelList, cienv$eq$is_array)[[1]] )
    message2(sprintf("Total trainable parameter count: %s", 
                    nParamsRep <- nTrainable <- 
                      sum(unlist(lapply(cienv$jax$tree$leaves(cienv$eq$partition(ModelList, 
                                         cienv$eq$is_array)[[1]]), function(zer){zer$size})))))
    
    # jit update fxns
    jit_apply_updates <- cienv$eq$filter_jit( cienv$optax$apply_updates )
    jit_get_update <- cienv$eq$filter_jit( optax_optimizer$update )
  }
}

