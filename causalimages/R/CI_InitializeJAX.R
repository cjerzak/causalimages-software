initialize_jax <- function(conda_env = "cienv", 
                           conda_env_required = TRUE) {
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  # Import Python packages once, storing them in cienv
  if (!exists("jax", envir = cienv, inherits = FALSE)) {
    cienv$jax <- reticulate::import("jax")
    cienv$jnp <- reticulate::import("jax.numpy")
    cienv$np  <- reticulate::import("numpy")
  }
  
  # Disable 64-bit computations
  cienv$jax$config$update("jax_enable_x64", FALSE)
  cienv$jaxFloatType <- cienv$jnp$float32
}
