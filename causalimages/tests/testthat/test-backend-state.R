test_that("partial backend state is repaired before writing tfrecords", {
  skip_on_cran()

  pkg_ns <- asNamespace("causalimages")
  cienv <- get("cienv", envir = pkg_ns)
  ci_backend_ready <- get("ci_backend_ready", envir = pkg_ns)

  backend_names <- ls(envir = cienv, all.names = TRUE)
  backend_snapshot <- if (length(backend_names) > 0L) {
    mget(backend_names, envir = cienv, inherits = FALSE)
  } else {
    list()
  }

  on.exit({
    current_names <- ls(envir = cienv, all.names = TRUE)
    if (length(current_names) > 0L) {
      rm(list = current_names, envir = cienv)
    }
    if (length(backend_snapshot) > 0L) {
      list2env(backend_snapshot, envir = cienv)
    }
  }, add = TRUE)

  current_names <- ls(envir = cienv, all.names = TRUE)
  if (length(current_names) > 0L) {
    rm(list = current_names, envir = cienv)
  }

  cienv$jax <- TRUE
  expect_false(ci_backend_ready())

  data("CausalImagesTutorialData", package = "causalimages", envir = environment())

  acquireImageFromMemory <- function(keys) {
    m_ <- FullImageArray[match(keys, KeysOfImages), , , ]
    if (length(keys) == 1) {
      m_ <- array(m_, dim = c(1L, 35L, 35L, 3L))
    }
    m_
  }

  tfrecord_loc <- tempfile(fileext = ".tfrecord")
  unlink(tfrecord_loc)

  causalimages::WriteTfRecord(
    file = tfrecord_loc,
    uniqueImageKeys = head(unique(KeysOfObservations), 2),
    acquireImageFxn = acquireImageFromMemory,
    conda_env = "CausalImagesEnv",
    conda_env_required = TRUE
  )

  expect_true(file.exists(tfrecord_loc))
  expect_true(ci_backend_ready())
})
