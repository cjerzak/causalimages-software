initialize_jax <- function(conda_env = "cienv", 
                           conda_env_required = TRUE,
                           Sys.setenv_text = NULL) {
  message2("Establishing connection to computational environment (build via causalimages::BuildBackend())")
  
  library(reticulate)
  #library(tensorflow)
  
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  if(!is.null(Sys.setenv_text)){ 
    eval(parse(text = Sys.setenv_text), envir = .GlobalEnv)
  }
  
  # Import Python packages once, storing them in cienv
  if (!exists("jax", envir = cienv, inherits = FALSE)) {
    cienv$jax <- reticulate::import("jax")
    cienv$jnp <- reticulate::import("jax.numpy")
    cienv$tf <- reticulate::import("tensorflow")
    cienv$np  <- reticulate::import("numpy")
    cienv$jmp  <- reticulate::import("jmp")
    cienv$optax  <- reticulate::import("optax")
    #cienv$oryx  <- reticulate::import("tensorflow_probability.substrates.jax")
    cienv$eq  <- reticulate::import("equinox")
    cienv$py_gc  <- reticulate::import("gc")
  }
  
  # set memory growth for tensorflow 
  for(device_ in cienv$tf$config$list_physical_devices()){ try(cienv$tf$config$experimental$set_memory_growth(device_, T),T) }
  
  # ensure tensorflow doesn't use GPU
  try(cienv$tf$config$set_visible_devices(list(), "GPU"), silent = TRUE)
  
  # Disable 64-bit computations
  cienv$jax$config$update("jax_enable_x64", FALSE)
  cienv$jaxFloatType <- cienv$jnp$float32
}

initialize_torch <- function(conda_env = "cienv", 
                           conda_env_required = TRUE,
                           Sys.setenv_text = NULL) {
  # Import Python packages once, storing them in cienv
  if (!exists("torch", envir = cienv, inherits = FALSE)) {
    cienv$torch <- reticulate::import("torch")
    cienv$transformers <- reticulate::import("transformers")
  }
}
