test_that("internal macro helpers are not exported", {
  exports <- getNamespaceExports("causalimages")

  expect_false("TFRecordManagement" %in% exports)
  expect_false("TrainDefine" %in% exports)
  expect_false("TrainDo" %in% exports)
})

test_that("runtime dependencies used by exported workflows are imports", {
  imports <- packageDescription("causalimages")$Imports

  expect_true(grepl("PRROC", imports, fixed = TRUE))
  expect_true(grepl("gtools", imports, fixed = TRUE))
})

test_that("ci_with_wd restores the caller working directory on errors", {
  old_wd <- getwd()

  expect_error(
    causalimages:::ci_with_wd(tempdir(), {
      expect_identical(normalizePath(getwd()), normalizePath(tempdir()))
      stop("forced failure")
    }),
    "forced failure"
  )
  expect_identical(getwd(), old_wd)
})

test_that("LocalFxnSource evaluates function bodies without parse/deparse", {
  target_env <- new.env(parent = baseenv())
  target_env$x <- 1
  local_fxn <- function() {
    y <- x + 1
  }

  causalimages:::LocalFxnSource(local_fxn, target_env)

  expect_identical(target_env$y, 2)
})

test_that("pretrained cache keys and activation isolate entries", {
  ci_ns <- asNamespace("causalimages")
  ci_env <- get("cienv", envir = ci_ns)
  cache_key <- get("ci_pretrained_cache_key", envir = ci_ns)
  cache_env_fxn <- get("ci_pretrained_cache_env", envir = ci_ns)
  cache_save <- get("ci_pretrained_cache_save", envir = ci_ns)
  cache_activate <- get("ci_pretrained_cache_activate", envir = ci_ns)

  key_a <- cache_key(
    pretrainedModel = "vit-base",
    dataType = "image",
    NORM_MEAN = c(1, 2, 3),
    NORM_SD = c(4, 5, 6),
    rawShape = c(1, 35, 35, 3)
  )
  key_b <- cache_key(
    pretrainedModel = "clip-rsicd",
    dataType = "image",
    NORM_MEAN = c(1, 2, 3),
    NORM_SD = c(4, 5, 6),
    rawShape = c(1, 35, 35, 3)
  )
  key_c <- cache_key(
    pretrainedModel = "vit-base",
    dataType = "image",
    NORM_MEAN = c(10, 2, 3),
    NORM_SD = c(4, 5, 6),
    rawShape = c(1, 35, 35, 3)
  )

  expect_false(identical(key_a, key_b))
  expect_false(identical(key_a, key_c))

  cache_env <- cache_env_fxn()
  if (exists(key_a, envir = cache_env, inherits = FALSE)) {
    rm(list = key_a, envir = cache_env)
  }
  if (exists(key_b, envir = cache_env, inherits = FALSE)) {
    rm(list = key_b, envir = cache_env)
  }
  on.exit({
    rm(
      list = intersect(c(key_a, key_b), ls(envir = cache_env, all.names = TRUE)),
      envir = cache_env
    )
    rm(
      list = intersect("CI_TEST_MODEL", ls(envir = ci_env, all.names = TRUE)),
      envir = ci_env
    )
  }, add = TRUE)

  ci_env$CI_TEST_MODEL <- "model-a"
  cache_save(key_a, names = "CI_TEST_MODEL")
  ci_env$CI_TEST_MODEL <- "model-b"
  cache_save(key_b, names = "CI_TEST_MODEL")

  expect_true(cache_activate(key_a, names = "CI_TEST_MODEL"))
  expect_identical(ci_env$CI_TEST_MODEL, "model-a")
  expect_true(cache_activate(key_b, names = "CI_TEST_MODEL"))
  expect_identical(ci_env$CI_TEST_MODEL, "model-b")
})
