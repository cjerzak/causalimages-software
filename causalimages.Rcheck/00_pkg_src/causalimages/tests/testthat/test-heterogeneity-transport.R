test_that("AnalyzeImageHeterogeneity scores transportability keys from tfrecord", {
  skip_on_cran()

  library(causalimages)
  skip_if(inherits(try(reticulate::import("tensorflow_probability.substrates.jax"), silent = TRUE), "try-error"),
          "tensorflow_probability is not available in the configured backend")
  data(CausalImagesTutorialData)

  test_dir <- if (exists("TEST_DATA_DIR", envir = .GlobalEnv)) {
    get("TEST_DATA_DIR", envir = .GlobalEnv)
  } else {
    tempdir()
  }
  dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)

  acquireImageFromMemory <- function(keys) {
    m_ <- FullImageArray[match(keys, KeysOfImages), 1:35, 1:35, ]
    if (length(keys) == 1L) {
      m_ <- array(m_, dim = c(1L, 35L, 35L, 3L))
    }
    m_
  }

  unique_obs <- match(unique(KeysOfObservations), KeysOfObservations)
  control_obs <- unique_obs[obsW[unique_obs] == 0]
  treated_obs <- unique_obs[obsW[unique_obs] == 1]

  train_indices <- c(control_obs[1:4], treated_obs[1:4])
  transport_indices <- c(control_obs[5], treated_obs[5], control_obs[6])
  train_keys <- KeysOfObservations[train_indices]
  transport_keys <- KeysOfObservations[transport_indices]

  tfrecord_path <- file.path(test_dir, "heterogeneity_transport.tfrecord")
  on.exit(unlink(tfrecord_path), add = TRUE)
  causalimages::WriteTfRecord(
    file = tfrecord_path,
    uniqueImageKeys = c(unique(train_keys), unique(transport_keys)),
    acquireImageFxn = acquireImageFromMemory
  )

  transportabilityMat <- data.frame(
    key = c(transport_keys, transport_keys[1]),
    marker = seq_len(length(transport_keys) + 1L),
    stringsAsFactors = FALSE
  )

  hetero_fit <- causalimages::AnalyzeImageHeterogeneity(
    obsW = obsW[train_indices],
    obsY = obsY[train_indices],
    imageKeysOfUnits = train_keys,
    file = tfrecord_path,
    kClust_est = 1L,
    plotResults = FALSE,
    optimizeImageRep = FALSE,
    imageModelClass = "VisionTransformer",
    nDepth_ImageRep = 1L,
    nWidth_ImageRep = 16L,
    nWidth_Dense = 16L,
    batchSize = 4L,
    nSGD = 2L,
    nMonte_predictive = 2L,
    nMonte_salience = 1L,
    nMonte_variational = 1L,
    transportabilityMat = transportabilityMat,
    seed = 1234L
  )

  expect_equal(nrow(hetero_fit$transportabilityMat), nrow(transportabilityMat))
  expect_true(all(c("mean_k1", "var_k1") %in% colnames(hetero_fit$transportabilityMat)))
  expect_false(anyNA(hetero_fit$transportabilityMat$mean_k1))
  expect_false(anyNA(hetero_fit$transportabilityMat$var_k1))
  expect_equal(
    hetero_fit$transportabilityMat$mean_k1[1],
    hetero_fit$transportabilityMat$mean_k1[nrow(hetero_fit$transportabilityMat)],
    tolerance = 1e-6
  )
  expect_equal(
    hetero_fit$transportabilityMat$var_k1[1],
    hetero_fit$transportabilityMat$var_k1[nrow(hetero_fit$transportabilityMat)],
    tolerance = 1e-6
  )
})
