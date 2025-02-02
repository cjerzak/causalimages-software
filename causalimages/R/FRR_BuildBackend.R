#' A function to build the environment for fastrerandomize. Builds a conda environment in which 'JAX' and 'np' are installed. Users can also create a conda environment where 'JAX' and 'np' are installed themselves. 
#'
#' @param conda_env (default = `"fastrerandomize"`) Name of the conda environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using `"auto"` allows reticulate to attempt to automatically find an appropriate conda binary.

#' @return Invisibly returns NULL; this function is used for its side effects 
#' of creating and configuring a conda environment for `fastrerandomize`. 
#' This function requires an Internet connection.
#' You can find out a list of conda Python paths via: `Sys.which("python")`
#'
#' @examples
#' \dontrun{
#' # Create a conda environment named "fastrerandomize"
#' # and install the required Python packages (jax, numpy, etc.)
#' build_backend(conda_env = "fastrerandomize", conda = "auto")
#'
#' # If you want to specify a particular conda path:
#' # build_backend(conda_env = "fastrerandomize", conda = "/usr/local/bin/conda")
#' }
#'
#' @export
#' @md

build_backend <- function(conda_env = "fastrerandomize", conda = "auto"){
  # Create a new conda environment
  reticulate::conda_create(envname = conda_env,
                           conda = conda,
                           python_version = "3.11")
  
  # Define packages to install 
  Packages2Install <- c("numpy==1.26.4",
                        "jax==0.4.38",
                        "jaxlib==0.4.38")
  
  # Add packages to install in METAL case 
  if( Sys.info()["machine"] == "arm64" & Sys.info()["sysname"] == "Darwin" ){
    Packages2Install <- c(Packages2Install,"jax-metal==0.1.1")
  }
  
  # Install packages 
  reticulate::py_install(Packages2Install, conda = conda, pip = TRUE, envname = conda_env)
}
