ci_python_distribution_version <- function(dist_name) {
  tryCatch({
    as.character(
      reticulate::import("importlib.metadata", delay_load = FALSE)$version(dist_name)
    )
  }, error = function(e) {
    NA_character_
  })
}

ci_stop_backend_incompatibility <- function(conda_env, original_error,
                                            dist_names = c("torch", "transformers",
                                                           "huggingface-hub",
                                                           "tokenizers")) {
  detected <- vapply(dist_names, ci_python_distribution_version, character(1))
  detected[is.na(detected)] <- "<not found>"

  stop(
    paste(
      sprintf(
        "The Python backend in conda env '%s' is incompatible with causalimages.",
        conda_env
      ),
      sprintf("Supported backend: %s", ci_supported_backend_summary()),
      sprintf(
        "Detected core packages: %s",
        paste(sprintf("%s=%s", names(detected), detected), collapse = ", ")
      ),
      sprintf(
        "Rebuild the environment with causalimages::BuildBackend(conda_env = '%s').",
        conda_env
      ),
      sprintf("Original Python import error: %s", conditionMessage(original_error)),
      sep = "\n"
    ),
    call. = FALSE
  )
}

initialize_jax <- function(conda_env = "cienv", 
                           conda_env_required = TRUE,
                           Sys.setenv_text = NULL) {
  message2("Establishing connection to computational environment (build via causalimages::BuildBackend())")
  
  if (!requireNamespace("reticulate", quietly = TRUE)) stop("Package 'reticulate' required")
  
  # Load reticulate (Declared in Imports: in DESCRIPTION)
  reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)
  
  if(!is.null(Sys.setenv_text)){ 
    eval(parse(text = Sys.setenv_text), envir = .GlobalEnv)
  }
  
  # Import Python packages once, storing them in cienv
  required_modules <- c("jax", "jnp", "tf", "np", "jmp", "optax", "eq", "py_gc")
  if (!all(required_modules %in% ls(envir = cienv))) {
    cienv$jax <- reticulate::import("jax")
    cienv$jnp <- reticulate::import("jax.numpy")
    cienv$flash_mha <- try(import("flash_attn_jax.flash_mha"),TRUE)
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
  # Import torch BEFORE other libraries (JAX/TF/NumPy) to avoid conflicts
  # This is critical for pretrained models using transformers
  required_modules <- c("torch", "transformers")
  if (!all(required_modules %in% ls(envir = cienv))) {
    message2("Initializing torch (must be imported before JAX/TF/NumPy for pretrained models)")

    if (!requireNamespace("reticulate", quietly = TRUE)) stop("Package 'reticulate' required")

    # Set up conda environment if not already done
    reticulate::use_condaenv(condaenv = conda_env, required = conda_env_required)

    if(!is.null(Sys.setenv_text)){
      eval(parse(text = Sys.setenv_text), envir = .GlobalEnv)
    }

    # Import torch and transformers first
    tryCatch({
      cienv$torch <- reticulate::import("torch")
      cienv$transformers <- reticulate::import("transformers")
    }, error = function(e) {
      loaded_modules <- intersect(required_modules, ls(envir = cienv))
      if (length(loaded_modules) > 0L) {
        rm(list = loaded_modules, envir = cienv)
      }
      ci_stop_backend_incompatibility(conda_env = conda_env, original_error = e)
    })
  }
}

# Helper function to check if a pretrained model requires torch/transformers
pretrained_model_requires_torch <- function(pretrainedModel) {
  if (is.null(pretrainedModel)) return(FALSE)
  pretrainedModel <- as.character(pretrainedModel)
  # Generic transformers-XXX pattern (e.g., "transformers-facebook/dinov2-base")
  if (grepl("^transformers-", pretrainedModel, ignore.case = TRUE)) return(TRUE)
  # These pretrained models use torch/transformers
  torch_patterns <- c("clip", "swin", "vit-base", "clay", "videomae")
  any(sapply(torch_patterns, function(p) grepl(p, pretrainedModel, ignore.case = TRUE)))
}
