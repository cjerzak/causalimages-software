#' Defines an internal training routine (internal function)
#'
#' Defines trainers defined in TrainDefine(). Internal function.
#' This function takes no parameters and is called within parent functions
#' to set up the optimizer and training step.
#'
#' @return Internal function defining a training sequence. 
#'
#' @import reticulate rrapply
#' @noRd
TrainDefine <- function(){
  if (!exists("earlyStopThreshold", inherits = FALSE)) earlyStopThreshold <- NULL
  if (!exists("learningRateBatchScaling", inherits = FALSE)) learningRateBatchScaling <- TRUE
  message2("Define optimizer and training step...")
  {
    # sqrt(batch size) LR scaling; TrainDo concatenates control + treated batches under TFRecordControl
    nEffectiveBatch <- batchSize * if (is.null(TFRecordControl)) 1L else 2L
    learningRatePeak <- ci_sqrt_batch_lr(learningRateMax, nEffectiveBatch, scale = learningRateBatchScaling)
    message2(sprintf("Peak learning rate: %.3g (learningRateMax = %.3g%s)",
                     learningRatePeak, learningRateMax,
                     if (isTRUE(learningRateBatchScaling)) sprintf(" x sqrt(%s/16)", nEffectiveBatch) else ", no batch scaling"))
    LR_schedule <- cienv$optax$warmup_cosine_decay_schedule(
                                                      warmup_steps = (nWarmup <- (0.05*nSGD)),
                                                      decay_steps = nSGD-nWarmup,
                                                      init_value = learningRatePeak/100,
                                                      peak_value = learningRatePeak,
                                                      end_value =  learningRatePeak/100)
    plot(cienv$np$array(LR_schedule(cienv$jnp$array(1:nSGD))),xlab = "Iteration", ylab="Learning rate schedule")
    optax_optimizer <-  cienv$optax$chain(
      cienv$optax$adaptive_grad_clip(clipping = 0.25, eps = 0.001),
      cienv$optax$adabelief( learning_rate = LR_schedule )
    )
    plot(cienv$np$array(LR_schedule(cienv$jnp$array(1:nSGD))), xlab = "Iteration", ylab = "Learning rate")
    
    # model partition, setup state, perform parameter count
    message2(sprintf("Total trainable parameter count: %s", 
                    nParamsRep <- nTrainable <- 
                      sum(unlist(lapply(cienv$jax$tree$leaves(cienv$eq$partition(ModelList, 
                                         cienv$eq$is_array)[[1]]), function(zer){zer$size})))))
    
    TrainStep <- ci_runtime()$TrainingStep(
      GetLoss, optax_optimizer, loss_scaling = identical(image_dtype_char, "float16")
    )
    training_state <- TrainStep$init(ModelList, StateList, MPList[[2]])
    # The compiled step owns these buffers from here until training completes.
    ModelList <- StateList <- NULL
  }
}
