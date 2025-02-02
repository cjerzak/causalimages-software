#!/usr/bin/env Rscript
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

BuildBackend <- function(conda_env = "CausalImagesEnv", conda = "auto"){
  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda, python_version = "3.12")

  # Install Python packages within the environment
  reticulate::py_install(c("tensorflow", "optax", "jax","tensorflow-datasets"
                           "torch","transformers","pillow","tf-keras", 
                           "equinox", "jmp", "tensorflow-probability"),
                           conda = conda, 
                           pip = TRUE,
                           envname = conda_env)
  if(Sys.info()["sysname"] == "Linux"){
    # pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
    try_ <- try(system(sprintf("'%s' -m pip install --upgrade --no-user %s",
           gsub(conda, pattern = "bin/python", replacement = sprintf("envs/%s/bin/python",conda_env)),
           "'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" )), T)
    if("try-error" %in% class(try_)){
      print("Failed to establish connection with jax[cuda12_pip], falling back to jax...")
      try_ <- try(reticulate::py_install("jax", conda = conda, pip = TRUE,
                                         envname = conda_env), T)
    }
  }
  message("Done building causalimages backend!")
}

