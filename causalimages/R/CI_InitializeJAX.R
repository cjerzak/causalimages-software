initialize_jax <- function(conda_env = "fastrerandomize", 
                           conda_env_required = TRUE) {
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in fastrr_env
  if (!exists("jax", envir = fastrr_env, inherits = FALSE)) {
    fastrr_env$jax <- reticulate::import("jax")
    fastrr_env$jnp <- reticulate::import("jax.numpy")
    fastrr_env$np  <- reticulate::import("numpy")
  }
  
  # Disable 64-bit computations
  fastrr_env$jax$config$update("jax_enable_x64", FALSE)
  fastrr_env$jaxFloatType <- fastrr_env$jnp$float32
  
  # Setup core JAX functions and store them in fastrr_env
  {
    fastrr_env$InsertOnes <- fastrr_env$jax$jit( function(treat_indices_, zeros_){
      zeros_ <- zeros_$at[treat_indices_]$add(1L)
      return(  zeros_ )
    } )
    fastrr_env$InsertOnesVectorized <- fastrr_env$jax$jit( fastrr_env$jax$vmap(function(treat_indices_, zeros_){
      fastrr_env$InsertOnes(treat_indices_, zeros_)
    }, list(1L,NULL)))
    
    fastrr_env$FastDiffInMeans <- fastrr_env$jax$jit( FastDiffInMeans_R <- function(y_,w_, n0, n1){
      my1 <- fastrr_env$jnp$divide(fastrr_env$jnp$sum(fastrr_env$jnp$multiply(y_,w_)), n1)
      my0 <- fastrr_env$jnp$divide(fastrr_env$jnp$sum(fastrr_env$jnp$multiply(y_, fastrr_env$jnp$subtract(1.,w_))), n0)
      return( diff10 <- fastrr_env$jnp$subtract(my1, my0) )
    })
    
    fastrr_env$VectorizedTakeAxis0 <- fastrr_env$jax$jit( VectorizedTakeAxis0_R <- function(A_, I_){
      fastrr_env$jnp$expand_dims(fastrr_env$jnp$take(A_, I_, axis = 0L), 0L)
    })
    
    fastrr_env$Potential2Obs <- fastrr_env$jax$jit(fastrr_env$Potential2Obs_R <- function(Y0__, Y1__, obsW__){
      fastrr_env$jnp$add( fastrr_env$jnp$multiply(Y0__, fastrr_env$jnp$subtract(1, obsW__)),
               fastrr_env$jnp$multiply(Y1__, obsW__))
    })
    
    fastrr_env$W_VectorizedFastDiffInMeans <- fastrr_env$jax$jit( 
      W_VectorizedFastDiffInMeans_R <- fastrr_env$jax$vmap(function(y_, w_, n0, n1){
          FastDiffInMeans_R(y_, w_, n0, n1)},
      in_axes = list(NULL, 0L, NULL, NULL)) )
    
    fastrr_env$Y_VectorizedFastDiffInMeans <- fastrr_env$jax$jit( 
      Y_VectorizedFastDiffInMeans_R <- fastrr_env$jax$vmap(function(y_, w_, n0, n1){
         FastDiffInMeans_R(y_, w_, n0, n1)},
      in_axes = list(0L, NULL, NULL, NULL)) )
    
    fastrr_env$YW_VectorizedFastDiffInMeans <- fastrr_env$jax$jit( 
      YW_VectorizedFastDiffInMeans_R <- fastrr_env$jax$vmap(function(y_, w_, n0, n1){
        fastrr_env$W_VectorizedFastDiffInMeans(y_, w_, n0, n1)},
      in_axes = list(0L, NULL, NULL, NULL)) )
    
    fastrr_env$WVectorizedFastDiffInMeans <- fastrr_env$jax$jit( 
      fastrr_env$jax$vmap(function(y_, w_, n0, n1){
        fastrr_env$FastDiffInMeans(y_, w_, n0, n1)},
      in_axes = list(NULL, 0L, NULL, NULL)) )
    
    fastrr_env$GreaterEqualMagCompare <- fastrr_env$jax$jit(GreaterEqualMagCompare_R <- function(NULL_, OBS_){
      fastrr_env$jnp$mean(fastrr_env$jnp$greater_equal(fastrr_env$jnp$abs(NULL_),  fastrr_env$jnp$expand_dims(OBS_,1L)), 1L)
    })

    fastrr_env$get_stat_vec_at_tau_pseudo <- fastrr_env$jax$jit( function(treatment_pseudo,
                                                     obsY_array,
                                                     obsW_array,
                                                     tau_pseudo,
                                                     n0_array,
                                                     n1_array){
      Y0_under_null <- fastrr_env$jnp$subtract(obsY_array,  fastrr_env$jnp$multiply(obsW_array, tau_pseudo))
      Y1_under_null_pseudo <- fastrr_env$jnp$add(Y0_under_null,  fastrr_env$jnp$multiply(treatment_pseudo, tau_pseudo))

      Yobs_pseudo <- fastrr_env$jnp$add(fastrr_env$jnp$multiply(Y1_under_null_pseudo, treatment_pseudo),
                             fastrr_env$jnp$multiply(Y0_under_null, fastrr_env$jnp$subtract(1., treatment_pseudo)))
      stat_ <- fastrr_env$FastDiffInMeans(Yobs_pseudo, treatment_pseudo, n0_array, n1_array)
    } )
    
    fastrr_env$vec1_get_stat_vec_at_tau_pseudo <- fastrr_env$jax$jit( fastrr_env$jax$vmap(function(treatment_pseudo,
                                                                   obsY_array,
                                                                   obsW_array,
                                                                   tau_pseudo,
                                                                   n0_array,
                                                                   n1_array){
      fastrr_env$get_stat_vec_at_tau_pseudo(treatment_pseudo,
                                 obsY_array,
                                 obsW_array,
                                 tau_pseudo,
                                 n0_array,
                                 n1_array)
    }, in_axes = list(0L, NULL, NULL, NULL, NULL, NULL)) )
    
    
    fastrr_env$RowBroadcast <- fastrr_env$jax$vmap(function(mat, vec){
      fastrr_env$jnp$multiply(mat, vec)}, in_axes = list(1L, NULL))
      
    fastrr_env$FastHotel2T2 <- ( function(samp_, samp_cov_inv, samp_cov_inv_approx, w_,  n0, n1, approximate_inv = FALSE){
      
      # set up calc
      xbar1 <- fastrr_env$jnp$divide(fastrr_env$jnp$sum(
                            fastrr_env$RowBroadcast(samp_, w_),1L,keepdims = T), n1)
      xbar2 <- fastrr_env$jnp$divide(fastrr_env$jnp$sum(
                            fastrr_env$RowBroadcast(samp_,fastrr_env$jnp$subtract(1.,w_)),1L,keepdims = T), n0)
      CovWts <- fastrr_env$jnp$add(fastrr_env$jnp$reciprocal(n0), fastrr_env$jnp$reciprocal(n1))
      xbar_diff <- fastrr_env$jnp$subtract(xbar1, xbar2)
      CovInv_TIMES_xbar_diff <- fastrr_env$jax$lax$cond(pred = approximate_inv,
                                        
                             # if using diagonal approximation           
                             true_fun = function(){
                               #CovPooled <- fastrr_env$jnp$var(samp_, 0L); 
                               #samp_cov_inv <- fastrr_env$jnp$reciprocal(CovPooled)
                               #CovInv_TIMES_xbar_diff_ <- fastrr_env$jnp$matmul(fastrr_env$jnp$diag( fastrr_env$jnp$reciprocal( CovPooled) ), xbar_diff)
                               CovInv_TIMES_xbar_diff_ <- fastrr_env$jnp$multiply(fastrr_env$jnp$expand_dims(samp_cov_inv_approx, 1L),  xbar_diff)  # no need for matrix casting 
                               
                               return(CovInv_TIMES_xbar_diff_)
                              },
                             
                             # if using no diagonal approximation (avoiding matrix inverse for numerical stability)
                             false_fun = function(){
                               #CovPooled <- fastrr_env$jnp$cov(samp_,rowvar = FALSE); 
                               #samp_cov_inv <- fastrr_env$jnp$linalg$inv(samp_,rowvar = FALSE); 
                               CovInv_TIMES_xbar_diff_ <- fastrr_env$jnp$matmul(samp_cov_inv, xbar_diff)
                               
                               return( CovInv_TIMES_xbar_diff_ )
                              }
                             )
      Tstat <- fastrr_env$jnp$multiply( (n0 * n1) / (n0 + n1), 
                                    fastrr_env$jnp$matmul(fastrr_env$jnp$transpose(xbar_diff), CovInv_TIMES_xbar_diff) )
    })
    
    fastrr_env$VectorizedFastHotel2T2 <- fastrr_env$jax$jit(VectorizedFastHotel2T2_R <- fastrr_env$jax$vmap(
            function(
                     samp_, samp_cov_inv_, samp_cov_inv_approx_,
                     w_, 
                     n0, n1, approximate_inv = FALSE){
      fastrr_env$FastHotel2T2(samp_, samp_cov_inv_, samp_cov_inv_approx_, 
                              w_, 
                              n0, n1, approximate_inv)},
    in_axes = list(NULL, NULL, NULL, 0L, NULL, NULL, NULL)) 
    )
    
    fastrr_env$BatchedVectorizedFastHotel2T2 <- function(
                                               samp_, samp_cov_inv_, samp_cov_inv_approx_,
                                               w_, 
                                               n0, n1, NWBatch, approximate_inv = FALSE){
      N_w <- w_$shape[0]  # Total number of w_ vectors
      num_batches <- as.integer( (N_w + NWBatch - 1) / NWBatch )  # Calculate number of batches
      
      # Function to process a single batch
      process_batch <- function(batch_idx, carry){
        start_idx <- batch_idx * NWBatch
        end_idx <- fastrr_env$jnp$minimum(start_idx + NWBatch, N_w)
        w_batch <- w_[start_idx:end_idx, ]
        
        # Compute the statistics for the current batch
        Tstat_batch <- fastrr_env$VectorizedFastHotel2T2(samp_, samp_cov_inv_, samp_cov_inv_approx_,  
                                                         w_batch, 
                                                         n0, n1,
                                                         approximate_inv)
        
        # Accumulate results
        carry <- fastrr_env$jnp$concatenate(list(carry, Tstat_batch), axis=0)
        return( carry ) 
      }
      
      # Initialize carry with an empty array
      carry_init <- fastrr_env$jnp$array(NULL, dtype=fastrr_env$jnp$float32)
      
      # Loop over batches using lax.fori_loop for efficiency
      Tstats <- fastrr_env$jax$lax$fori_loop(
        lower=0,
        upper=num_batches,
        body_fun=function( i, carry ){ process_batch(i, carry)},
        init_val=carry_init
      )
      
      return( Tstats ) 
    }
  }
}
