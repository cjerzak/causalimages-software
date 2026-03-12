#!/usr/bin/env Rscript

test_that("BuildBackend works", {
  skip_on_cran()
  conda_binary <- try(reticulate::conda_binary("auto"), silent = TRUE)
  skip_if(inherits(conda_binary, "try-error") || !nzchar(conda_binary) ||
            !file.exists(conda_binary), "Conda not found")

  versions <- causalimages:::ci_supported_backend_versions()
  env_name <- sprintf("CausalImagesTestEnv_%s", Sys.getpid())
  on.exit(
    try(reticulate::conda_remove(envname = env_name, conda = conda_binary),
        silent = TRUE),
    add = TRUE
  )

  # setup backend, conda points to location of conda binary
  # note: This function requires an Internet connection
  # you can find out a list of conda paths via:
  # system("which conda")
  causalimages::BuildBackend(conda_env = env_name, conda = conda_binary)

  conda_envs <- reticulate::conda_list(conda = conda_binary)
  env_python <- conda_envs$python[match(env_name, conda_envs$name)]
  skip_if(is.na(env_python) || !file.exists(env_python),
          "Conda env python not found")
  version_cmd <- sprintf(
    "%s -c %s",
    shQuote(env_python),
    shQuote(paste(
      "import importlib.metadata as md",
      "print(md.version('transformers'))",
      "print(md.version('huggingface-hub'))",
      sep = "\n"
    ))
  )
  version_lines <- system(version_cmd, intern = TRUE)
  expect_equal(
    version_lines[[1]],
    unname(versions["transformers"])
  )
  expect_equal(
    version_lines[[2]],
    unname(versions["huggingface-hub"])
  )
  print("Done with BuildBackend() test!")
})
