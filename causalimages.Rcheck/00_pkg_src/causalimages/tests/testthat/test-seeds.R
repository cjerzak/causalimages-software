test_that("ci_seed_int32 validates scalar integer seeds and rejects overflow", {
  expect_identical(causalimages:::ci_seed_int32(123L), 123L)
  expect_identical(causalimages:::ci_seed_int32(123L, 5L), 128L)
  expect_error(causalimages:::ci_int32_scalar(c(1L, 2L)), "scalar")
  expect_error(causalimages:::ci_seed_int32(1.5, 0L), "integer-like")
  expect_error(causalimages:::ci_seed_int32(2147483647L, 1L), "overflowed")
})

test_that("GetImageRepresentations avoids integer coercion warnings for cross-modal seeds", {
  skip_on_cran()

  data(CausalImagesTutorialData)

  take_indices <- c(head(which(obsW == 0), 2L), head(which(obsW == 1), 2L))
  X_full <- apply(X[, -1, drop = FALSE], 2, function(zer) {
    zer[is.na(zer)] <- mean(zer, na.rm = TRUE)
    zer
  })
  X_small <- as.matrix(X_full[take_indices, , drop = FALSE])
  X_small <- X_small[, apply(X_small, 2, sd) > 0, drop = FALSE]

  acquireImageFromMemory <- function(keys) {
    m_ <- FullImageArray[match(keys, KeysOfImages), 1:35, 1:35, ]
    if (length(keys) == 1L) {
      m_ <- array(m_, dim = c(1L, 35L, 35L, 3L))
    }
    m_
  }

  tfrecord_file <- file.path(
    tempdir(),
    sprintf("seed-regression-%s.tfrecord", Sys.getpid())
  )
  on.exit(unlink(tfrecord_file), add = TRUE)

  causalimages::WriteTfRecord(
    file = tfrecord_file,
    uniqueImageKeys = unique(KeysOfObservations[take_indices]),
    acquireImageFxn = acquireImageFromMemory
  )

  warnings_seen <- character()
  result <- withCallingHandlers(
    causalimages::GetImageRepresentations(
      X = X_small,
      file = tfrecord_file,
      imageKeysOfUnits = KeysOfObservations[take_indices],
      imageModelClass = "VisionTransformer",
      nWidth_ImageRep = 16L,
      nDepth_ImageRep = 1L,
      batchSize = 2L,
      getRepresentations = FALSE,
      returnContents = TRUE,
      seed = 123L
    ),
    warning = function(w) {
      warnings_seen <<- c(warnings_seen, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  expect_false(any(grepl("NAs introduced by coercion to integer range", warnings_seen, fixed = TRUE)))
  expect_true(is.list(result))
  expect_true(is.null(result$ImageRepresentations))
})
