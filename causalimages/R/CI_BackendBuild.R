#' Build the environment for CausalImages models. Builds a conda environment in which jax, tensorflow, tensorflow-probability optax, equinox, and jmp are installed.
#'
#' @param conda_env (default = `"CausalImagesEnv"`) Name of the conda environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using `"auto"` allows reticulate to attempt to automatically find an appropriate conda binary.

#' @return Builds the computational environment for `causalimages`. This function requires an Internet connection.
#' You may find out a list of conda Python paths via: `system("which python")`
#'
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/causalimages-software/
#'
#' @export
#' @md

BuildBackend <- function(conda_env = "CausalImagesEnv", conda = "auto") {
  # --- helpers ---------------------------------------------------------------
  os <- Sys.info()[["sysname"]]
  machine <- Sys.info()[["machine"]]
  msg <- function(...) message(sprintf(...))
  
  pip_install <- function(pkgs, ...) {
    reticulate::py_install(
      packages = pkgs,
      envname  = conda_env,
      conda    = conda,
      pip      = TRUE,
      ...
    )
    TRUE
  }
  
  # Find the Python executable inside the target conda env (for manual pip calls)
  env_python_path <- function() {
    # try via conda_list
    cl <- try(reticulate::conda_list(), silent = TRUE)
    if (!inherits(cl, "try-error") && any(cl$name == conda_env)) {
      py <- cl$python[match(conda_env, cl$name)]
      if (length(py) == 1 && !is.na(py) && nzchar(py) && file.exists(py)) return(py)
    }
    # fallback via conda binary location
    cb <- try(reticulate::conda_binary(conda), silent = TRUE)
    prefix <- if (!inherits(cb, "try-error") && nzchar(cb)) dirname(dirname(cb)) else {
      # last-ditch default
      if (os == "Windows") "C:/Miniconda3" else file.path(Sys.getenv("HOME"), "miniconda3")
    }
    if (os == "Windows")
      file.path(prefix, "envs", conda_env, "python.exe")
    else
      file.path(prefix, "envs", conda_env, "bin", "python")
  }
  
  pip_install_from_findlinks <- function(spec, find_links) {
    py <- env_python_path()
    cmd <- sprintf(
      "%s -m pip install --upgrade --no-user -f %s %s",
      shQuote(py), shQuote(find_links), shQuote(spec)
    )
    res <- try(system(cmd, intern = TRUE), silent = TRUE)
    !inherits(res, "try-error")
  }
  
  # --- conda env -------------------------------------------------------------
  reticulate::conda_create(
    envname        = conda_env,
    conda          = conda,
    python_version = "3.13"
  )
  
  # Install numpy early to stabilize BLAS/ABI choices if needed
  pip_install("numpy")
  
  # --- JAX first: hardware-aware selection -----------------------------------
  install_jax <- function() {
    if (os == "Darwin" && machine %in% c("arm64", "aarch64")) {
      # Apple Silicon: Metal backend
      #msg("Apple Silicon detected: installing JAX (Metal).")
      #pip_install(c("jax==0.5.0", "jaxlib==0.5.0", "jax-metal==0.1.1"))
      pip_install(c("jax", "jaxlib"))
      return(invisible(TRUE))
    }
    
    if (identical(os, "Linux")) {
      # Query NVIDIA driver major version (e.g., '535.171.04' -> 535)
      drv <- try(suppressWarnings(
        system("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1",
               intern = TRUE)
      ), silent = TRUE)
      drv_major <- suppressWarnings(as.integer(sub("^([0-9]+).*", "\\1", drv[1])))
      msg("Detected NVIDIA driver: %s", ifelse(length(drv) > 0, drv[1], "none/unknown"))
      
      # Prefer CUDA 13 when possible, fall back to CUDA 12, then CPU
      if (!is.na(drv_major) && drv_major >= 580) {
        msg("Driver >= 580: trying JAX CUDA 13 wheels.")
        ok <- try(pip_install("jax[cuda13]"), silent = TRUE)
        ok <- isTRUE(ok) && !inherits(ok, "try-error")
        if (!ok) {
          msg("CUDA 13 wheels failed; falling back to CUDA 12 extras.")
          ok <- try(pip_install("jax[cuda12]"), silent = TRUE)
          ok <- isTRUE(ok) && !inherits(ok, "try-error")
        }
        if (!ok) {
          msg("CUDA 12 extras failed; trying legacy 'cuda12_pip' via find-links.")
          ok <- pip_install_from_findlinks(
            "jax[cuda12_pip]",
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
          )
        }
        if (!ok) {
          msg("All CUDA wheel attempts failed; installing CPU-only JAX.")
          pip_install("jax")
        }
      } else if (!is.na(drv_major) && drv_major >= 525) {
        msg("Driver >= 525 and < 580: installing JAX CUDA 12 wheels.")
        ok <- try(pip_install("jax[cuda12]"), silent = TRUE)
        ok <- isTRUE(ok) && !inherits(ok, "try-error")
        if (!ok) {
          msg("CUDA 12 extras failed; trying legacy 'cuda12_pip' via find-links.")
          ok <- pip_install_from_findlinks(
            "jax[cuda12_pip]",
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
          )
        }
        if (!ok) {
          msg("CUDA wheels failed; installing CPU-only JAX.")
          pip_install("jax")
        }
      } else {
        msg("No suitable NVIDIA driver found (or too old); installing CPU-only JAX.")
        pip_install("jax")
      }
      return(invisible(TRUE))
    }
    
    # Other OSes: CPU-only JAX
    msg("Non-Linux or non-Apple-Silicon platform; installing CPU-only JAX.")
    pip_install("jax")
  }
  
  install_jax()
  
  # Optionally neutralize LD_LIBRARY_PATH within this env to avoid host overrides
  if (os == "Linux") {
    cb <- try(reticulate::conda_binary(conda), silent = TRUE)
    conda_prefix <- if (!inherits(cb, "try-error") && nzchar(cb)) dirname(dirname(cb)) else {
      file.path(Sys.getenv("HOME"), "miniconda3")
    }
    env_dir <- file.path(conda_prefix, "envs", conda_env)
    actdir <- file.path(env_dir, "etc", "conda", "activate.d")
    dir.create(actdir, recursive = TRUE, showWarnings = FALSE)
    try(writeLines("unset LD_LIBRARY_PATH", file.path(actdir, "00-unset-ld.sh")), silent = TRUE)
  }
  
  # --- Remaining packages (do NOT include 'jax' again to avoid downgrades) ----
  pip_install(c(
    "tensorflow",
    "optax",
    "torch",
    "flax",
    "torchax",
    "transformers",
    "pillow",
    "tf-keras",
    "equinox",
    "jmp"
  ))
  
  # reinstall JAX again on Mac after forced upgrade to breaking version of JAX
  #if(os == "Darwin"){ install_jax() }
  
  done_msg <- sprintf("Done building causalimages backend (env '%s').", conda_env)
  if (exists("message2", mode = "function")) message2(done_msg) else message(done_msg)
}
