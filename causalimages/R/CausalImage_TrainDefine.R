#' Defines a trainer. 
#'
#' Defines trainers defined in TrainDefine(). 
#'
#' @param . No parameters. 
#' 
#' @return Defines a training sequence
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
TrainDefine <- function(){
  print2("Define optimizer and training step...") 
  {
    LR_schedule <- optax$warmup_cosine_decay_schedule(warmup_steps = (nWarmup <- min(c(100L, nSGD))),
                                                      decay_steps = max(c(101L, nSGD-nWarmup)),
                                                      init_value = learningRateMax/100, 
                                                      peak_value = learningRateMax, 
                                                      end_value =  learningRateMax/100)
    plot(np$array(LR_schedule(jnp$array(1:nSGD))),xlab = "Iteration", ylab="Learning rate")
    optax_optimizer <-  optax$chain(
      optax$clip(1), 
      optax$adaptive_grad_clip(clipping = 0.05),
      optax$adabelief( learning_rate = LR_schedule )
    )
    
    # model partition, setup state, perform parameter count
    opt_state <- optax_optimizer$init(   eq$partition(ModelList, eq$is_array)[[1]] )
    print2(sprintf("Total trainable parameter count: %s", nParamsRep <- nTrainable <- sum(unlist(lapply(jax$tree_leaves(eq$partition(ModelList, eq$is_array)[[1]]), function(zer){zer$size})))))
    # unlist(lapply(jax$tree_leaves(eq$partition(ModelList, eq$is_array)[[1]]), function(zer){zer$size}))
    
    # jit update fxns
    jit_apply_updates <- eq$filter_jit( optax$apply_updates )
    jit_get_update <- eq$filter_jit( optax_optimizer$update )
  }
}

