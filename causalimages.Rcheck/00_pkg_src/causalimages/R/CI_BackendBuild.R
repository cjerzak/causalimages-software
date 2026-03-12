#' Supported backend package versions for `causalimages`.
#'
#' @return A named character vector of pinned Python and package versions used
#'   by the backend installer.
#'
#' @export
#' @md
ci_supported_backend_versions <- function() {
  c(
    python = "3.13",
    numpy = "2.4.3",
    jax = "0.9.1",
    jaxlib = "0.9.1",
    tensorflow = "2.20.0",
    "tf-keras" = "2.20.1",
    optax = "0.2.6",
    torch = "2.10.0",
    flax = "0.12.5",
    torchax = "0.0.11",
    transformers = "5.3.0",
    "huggingface-hub" = "1.6.0",
    tokenizers = "0.22.2",
    pillow = "12.1.1",
    equinox = "0.13.6",
    jmp = "0.0.4"
  )
}

ci_backend_version_specs <- function(packages = NULL) {
  versions <- ci_supported_backend_versions()
  if (is.null(packages)) {
    packages <- names(versions)[names(versions) != "python"]
  }
  sprintf("%s==%s", packages, unname(versions[packages]))
}

ci_supported_backend_summary <- function() {
  versions <- ci_supported_backend_versions()
  pkgs <- versions[names(versions) != "python"]
  paste(
    c(
      sprintf("python=%s", versions[["python"]]),
      sprintf("%s==%s", names(pkgs), unname(pkgs))
    ),
    collapse = ", "
  )
}

#' Build the environment for CausalImages models.
#'
#' Builds a conda environment in which jax, tensorflow,
#' tensorflow-probability, optax, equinox, and jmp are installed.
#'
#' @param conda_env (default = `"CausalImagesEnv"`) Name of the conda
#'   environment in which to place the backends.
#' @param conda (default = `auto`) The path to a conda executable. Using
#'   `"auto"` allows reticulate to attempt to automatically find an
#'   appropriate conda binary.
#' @return Builds the computational environment for `causalimages`. This
#'   function requires an Internet connection. You may find out a list of conda
#'   Python paths via: `system("which python")`
#' @examples
#' # For a tutorial, see
#' # github.com/cjerzak/causalimages-software/
#' @export
#' @md
BuildBackend <- function(conda_env = "CausalImagesEnv", conda = "auto") {
  # --- helpers ---------------------------------------------------------------
  os <- Sys.info()[["sysname"]]
  machine <- Sys.info()[["machine"]]
  versions <- ci_supported_backend_versions()
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

  pip_check <- function() {
    py <- env_python_path()
    out <- try(system2(py, c("-m", "pip", "check"), stdout = TRUE, stderr = TRUE),
               silent = TRUE)
    if (inherits(out, "try-error")) {
      stop(sprintf(
        "BuildBackend() could not validate Python dependencies in env '%s'.",
        conda_env
      ), call. = FALSE)
    }
    status <- attr(out, "status")
    if (is.null(status)) status <- 0L
    if (status != 0L) {
      stop(sprintf(
        "BuildBackend() installed an inconsistent Python environment in '%s'.\n%s",
        conda_env,
        paste(out, collapse = "\n")
      ), call. = FALSE)
    }
    invisible(TRUE)
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
    if (inherits(res, "try-error")) {
      return(FALSE)
    }
    status <- attr(res, "status")
    if (is.null(status)) status <- 0L
    status == 0L
  }
  
  # --- conda env -------------------------------------------------------------
  reticulate::conda_create(
    envname        = conda_env,
    conda          = conda,
    python_version = versions[["python"]]
  )

  msg("Installing supported backend packages: %s", ci_supported_backend_summary())
  
  # Install numpy early to stabilize BLAS/ABI choices if needed
  pip_install(ci_backend_version_specs("numpy"))
  
  # --- JAX first: hardware-aware selection -----------------------------------
  install_jax <- function() {
    cpu_jax_specs <- ci_backend_version_specs(c("jax", "jaxlib"))
    cuda13_spec <- sprintf("jax[cuda13]==%s", versions[["jax"]])
    cuda12_spec <- sprintf("jax[cuda12]==%s", versions[["jax"]])
    legacy_cuda12_spec <- sprintf("jax[cuda12_pip]==%s", versions[["jax"]])

    if (os == "Darwin" && machine %in% c("arm64", "aarch64")) {
      # Apple Silicon: Metal backend
      pip_install(cpu_jax_specs)
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
        ok <- try(pip_install(cuda13_spec), silent = TRUE)
        ok <- isTRUE(ok) && !inherits(ok, "try-error")
        if (!ok) {
          msg("CUDA 13 wheels failed; falling back to CUDA 12 extras.")
          ok <- try(pip_install(cuda12_spec), silent = TRUE)
          ok <- isTRUE(ok) && !inherits(ok, "try-error")
        }
        if (!ok) {
          msg("CUDA 12 extras failed; trying legacy 'cuda12_pip' via find-links.")
          ok <- pip_install_from_findlinks(
            legacy_cuda12_spec,
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
          )
        }
        if (!ok) {
          msg("All CUDA wheel attempts failed; installing CPU-only JAX.")
          pip_install(cpu_jax_specs)
        }
      } else if (!is.na(drv_major) && drv_major >= 525) {
        msg("Driver >= 525 and < 580: installing JAX CUDA 12 wheels.")
        ok <- try(pip_install(cuda12_spec), silent = TRUE)
        ok <- isTRUE(ok) && !inherits(ok, "try-error")
        if (!ok) {
          msg("CUDA 12 extras failed; trying legacy 'cuda12_pip' via find-links.")
          ok <- pip_install_from_findlinks(
            legacy_cuda12_spec,
            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
          )
        }
        if (!ok) {
          msg("CUDA wheels failed; installing CPU-only JAX.")
          pip_install(cpu_jax_specs)
        }
      } else {
        msg("No suitable NVIDIA driver found (or too old); installing CPU-only JAX.")
        pip_install(cpu_jax_specs)
      }
      return(invisible(TRUE))
    }
    
    # Other OSes: CPU-only JAX
    msg("Non-Linux or non-Apple-Silicon platform; installing CPU-only JAX.")
    pip_install(cpu_jax_specs)
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
  pip_install(ci_backend_version_specs(c(
    "tensorflow",
    "optax",
    "torch",
    "flax",
    "torchax",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "pillow",
    "tf-keras",
    "equinox",
    "jmp"
  )))

  pip_check()
  
  done_msg <- sprintf("Done building causalimages backend (env '%s').", conda_env)
  if (exists("message2", mode = "function")) message2(done_msg) else message(done_msg)
}
