test_that("predictive tfrecord key map cache invalidates when files are overwritten", {
  skip_on_cran()

  target_python <- try(reticulate::conda_python(envname = "CausalImagesEnv"),
                       silent = TRUE)
  skip_if(
    inherits(target_python, "try-error") ||
      !nzchar(target_python) ||
      !file.exists(target_python),
    "CausalImagesEnv is not available"
  )

  acquireImageFxn <- function(keys) {
    out <- array(seq_len(length(keys) * 4L), dim = c(length(keys), 2L, 2L, 1L))
    if (length(keys) == 1L) {
      out <- array(out, dim = c(1L, 2L, 2L, 1L))
    }
    out
  }

  tfrecord_path <- file.path(tempdir(), "predictive_key_cache.tfrecord")
  on.exit(unlink(tfrecord_path), add = TRUE)

  causalimages::WriteTfRecord(
    file = tfrecord_path,
    uniqueImageKeys = c("a", "b"),
    acquireImageFxn = acquireImageFxn
  )
  first_map <- causalimages:::ci_tfrecord_key_index_map(tfrecord_path)
  expect_identical(first_map$keys, c("a", "b"))

  Sys.sleep(1.1)
  expect_warning(
    causalimages::WriteTfRecord(
      file = tfrecord_path,
      uniqueImageKeys = c("c", "d"),
      acquireImageFxn = acquireImageFxn
    ),
    "already exists"
  )
  second_map <- causalimages:::ci_tfrecord_key_index_map(tfrecord_path)

  expect_identical(second_map$keys, c("c", "d"))
})
