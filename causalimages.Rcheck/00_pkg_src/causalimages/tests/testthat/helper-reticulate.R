if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) {
  target_python <- try(reticulate::conda_python(envname = "CausalImagesEnv"),
                       silent = TRUE)

  if (!inherits(target_python, "try-error") &&
      nzchar(target_python) &&
      file.exists(target_python)) {
    Sys.setenv(RETICULATE_PYTHON = target_python)
  }
}
