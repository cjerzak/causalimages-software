test_that("PredictiveRun persists artifacts and scores transport data", {
  skip_on_cran()

  library(causalimages)
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

  X_proc <- apply(X[, -1, drop = FALSE], 2, function(zer) {
    zer[is.na(zer)] <- mean(zer, na.rm = TRUE)
    zer
  })

  unique_obs <- match(unique(KeysOfObservations), KeysOfObservations)
  control_obs <- unique_obs[obsW[unique_obs] == 0]
  treated_obs <- unique_obs[obsW[unique_obs] == 1]

  train_indices <- c(control_obs[1:5], treated_obs[1:5])
  transport_indices <- c(control_obs[6:7], treated_obs[6:7])

  keep_cols <- apply(X_proc[train_indices, , drop = FALSE], 2, sd) > 0
  X_proc <- X_proc[, keep_cols, drop = FALSE]

  train_keys <- KeysOfObservations[train_indices]
  transport_keys <- KeysOfObservations[transport_indices]

  train_file <- file.path(test_dir, "predictive_train.tfrecord")
  transport_file <- file.path(test_dir, "predictive_transport.tfrecord")
  model_path <- file.path(test_dir, "predictive_model.eqx")
  metrics_path <- file.path(test_dir, "predictive_metrics.rds")
  on.exit(unlink(c(train_file, transport_file, model_path, metrics_path, paste0(model_path, ".meta.rds"))), add = TRUE)

  causalimages::WriteTfRecord(
    file = train_file,
    uniqueImageKeys = unique(train_keys),
    acquireImageFxn = acquireImageFromMemory
  )
  causalimages::WriteTfRecord(
    file = transport_file,
    uniqueImageKeys = unique(transport_keys),
    acquireImageFxn = acquireImageFromMemory
  )

  predictive_fit <- causalimages::PredictiveRun(
    obsY = obsY[train_indices],
    X = X_proc[train_indices, , drop = FALSE],
    imageKeysOfUnits = train_keys,
    file = train_file,
    fileTransport = transport_file,
    imageKeysOfUnitsTransport = transport_keys,
    XTransport = X_proc[transport_indices, , drop = FALSE],
    batchSize = 4L,
    nSGD = 3L,
    testFrac = 0.2,
    optimizeImageRep = FALSE,
    imageModelClass = "VisionTransformer",
    nDepth_ImageRep = 1L,
    nWidth_ImageRep = 16L,
    nWidth_Dense = 16L,
    nDepth_Dense = 1L,
    learningRateMax = 0.001,
    plotResults = FALSE,
    modelPath = model_path,
    metricsPath = metrics_path,
    seed = 1234L
  )

  expect_true(file.exists(model_path))
  expect_true(file.exists(paste0(model_path, ".meta.rds")))
  expect_true(file.exists(metrics_path))
  expect_length(predictive_fit$predictedY, length(train_indices))
  expect_length(predictive_fit$predictedY_transport, length(transport_indices))
  expect_false(anyNA(predictive_fit$predictedY))
  expect_false(anyNA(predictive_fit$predictedY_transport))

  roundtrip_preds <- causalimages::PredictiveScore(
    modelPath = model_path,
    file = transport_file,
    imageKeysOfUnits = transport_keys,
    X = X_proc[transport_indices, , drop = FALSE],
    batchSize = 4L,
    seed = 1234L
  )

  expect_equal(roundtrip_preds$predictedY, predictive_fit$predictedY_transport, tolerance = 1e-6)
  expect_equal(roundtrip_preds$imageKeysOfUnits, as.character(transport_keys))
  expect_equal(roundtrip_preds$obsIndex, seq_along(transport_keys))

  expect_error(
    causalimages::PredictiveScore(
      modelPath = model_path,
      file = transport_file,
      imageKeysOfUnits = transport_keys,
      X = NULL,
      batchSize = 4L,
      seed = 1234L
    ),
    "requires X transport covariates"
  )
})
