ci_eff_fixture <- function(video = FALSE, count = 6L) {
  causalimages:::initialize_jax()
  keys <- paste0("memory-", seq_len(count))
  dims <- if (video) c(count, 3L, 16L, 16L, 3L) else c(count, 16L, 16L, 3L)
  images <- array(sin(seq_len(prod(dims)) / 19), dims)
  acquire <- if (video) function(k) images[match(k, keys), , , , , drop = FALSE] else
    function(k) images[match(k, keys), , , , drop = FALSE]
  file <- tempfile(fileext = ".tfrecord")
  causalimages::WriteTfRecord(keys, acquireImageFxn = acquire, file = file,
                             writeVideo = video, image_dtype = "float32")
  list(file = file, keys = keys, images = images, video = video)
}

ci_eff_rep <- function(fixture, version = 2L, checkpoint = TRUE, aggregation = "transformer", pretrained = NULL) {
  old <- options(causalimages.activation_checkpointing = checkpoint)
  on.exit(options(old), add = TRUE)
  causalimages::GetImageRepresentations(
    file = fixture$file, imageKeysOfUnits = fixture$keys, batchSize = 2L,
    dataType = if (fixture$video) "video" else "image", temporalAggregation = aggregation,
    pretrainedModel = pretrained, nWidth_ImageRep = 16L, nDepth_ImageRep = 2L,
    nDepth_TemporalRep = 3L, patchEmbedDim = 4L, NORM_MEAN = c(0, 0, 0), NORM_SD = c(1, 1, 1),
    image_dtype = causalimages:::cienv$jnp$float32,
    image_dtype_tf = causalimages:::cienv$tf$float32,
    modelStructureVersion = version, seed = 42L
  )
}

test_that("pretrained cache eviction and cleanup release all cache references", {
  old <- options(causalimages.pretrained_cache_size = 1L)
  on.exit(options(old), add = TRUE)
  env <- causalimages:::cienv
  causalimages:::ci_pretrained_cache_clear()
  on.exit(causalimages:::ci_pretrained_cache_clear(), add = TRUE)
  env$JAX_Weights <- "first"
  causalimages:::ci_pretrained_cache_save("first")
  expect_false(causalimages:::ci_pretrained_cache_activate("second"))
  expect_length(ls(causalimages:::ci_pretrained_cache_env()), 0L)
  env$JAX_Weights <- "second"
  causalimages:::ci_pretrained_cache_save("second")
  expect_true(causalimages:::ci_pretrained_cache_activate("second"))
  expect_identical(env$JAX_Weights, "second")
  causalimages:::ci_pretrained_cache_clear()
  expect_false(exists("JAX_Weights", env, inherits = FALSE))
  expect_length(ls(causalimages:::ci_pretrained_cache_env()), 0L)
})

test_that("no-grad scope restores gradient mode on errors", {
  env <- causalimages:::cienv
  previous <- if (exists("torch", env, inherits = FALSE)) env$torch else NULL
  on.exit({ if (is.null(previous)) rm(list = "torch", envir = env) else env$torch <- previous }, add = TRUE)
  enabled <- TRUE
  env$torch <- list(no_grad = function() list(
    `__enter__` = function() enabled <<- FALSE,
    `__exit__` = function(...) enabled <<- TRUE
  ))
  expect_false(causalimages:::ci_torch_inference(enabled))
  expect_true(enabled)
  expect_error(causalimages:::ci_torch_inference(stop("scope test")), "scope test")
  expect_true(enabled)
})

test_that("new image trees remove inactive heads and preserve legacy outputs", {
  skip_on_cran()
  f <- ci_eff_fixture()
  on.exit(unlink(f$file), add = TRUE)
  legacy <- ci_eff_rep(f, version = 1L, checkpoint = FALSE)
  compact <- ci_eff_rep(f)
  expect_equal(compact$ImageRepresentations, legacy$ImageRepresentations, tolerance = 2e-3)
  model <- compact$ImageModel_And_State_And_MPPolicy_List[[1]]
  expect_false(any(c("MixCLSPool", "PoolQuery", "FinalProj") %in% names(model$SpatialTransformerSupp)))
  expect_lt(compact$nParamsRep, legacy$nParamsRep)
  env <- causalimages:::cienv
  expect_true(all(vapply(env$jax$tree$leaves(model), function(x) all(is.finite(env$np$array(x))), logical(1))))
  expect_false(exists("ModelList", environment(compact$ImageRepArm_batch_R), inherits = FALSE))
})

test_that("temporal attention has independent layers and receives gradients", {
  skip_on_cran()
  f <- ci_eff_fixture(video = TRUE)
  on.exit(unlink(f$file), add = TRUE)
  rep <- ci_eff_rep(f)
  uncheckpointed <- ci_eff_rep(f, checkpoint = FALSE)
  env <- causalimages:::cienv
  bundle <- rep$ImageModel_And_State_And_MPPolicy_List
  model <- bundle[[1]]
  expect_identical(as.integer(model$TemporalTransformer$Multihead$W_q$shape[[1]]), 3L)
  expect_identical(as.integer(model$SpatialTransformer$Multihead$W_q$shape[[1]]), 2L)
  expect_true("PoolMultihead" %in% names(model$TemporalTransformerSupp))
  expect_false(any(grepl("^Temporal_d", names(model))))
  expect_equal(dim(rep$ImageRepresentations), c(6L, 16L))
  expect_equal(rep$ImageRepresentations, uncheckpointed$ImageRepresentations, tolerance = 1e-5)
  images <- env$jnp$array(f$images[1:2, , , , , drop = FALSE], dtype = env$jnp$float32)
  x <- env$jnp$zeros(list(2L, 1L))
  keys <- env$jax$random$split(env$jax$random$key(3L), 2L)
  mp <- bundle[[3]]
  mp[[1]] <- env$jmp$Policy(param_dtype = "float32", compute_dtype = "float32", output_dtype = "float32")
  loss <- function(model) env$jnp$sum(env$jnp$square(
    rep$ImageRepArm_batch_R(model, images, x, bundle[[2]], keys, mp, FALSE)[[1]][, 1:3]
  ))
  plain_loss <- function(model) env$jnp$sum(env$jnp$square(
    uncheckpointed$ImageRepArm_batch_R(model, images, x, bundle[[2]], keys, mp, FALSE)[[1]][, 1:3]
  ))
  gradients <- env$eq$filter_grad(loss)(model)
  plain_gradients <- env$eq$filter_grad(plain_loss)(model)
  leaves <- env$jax$tree$leaves(gradients)
  plain_leaves <- env$jax$tree$leaves(plain_gradients)
  for (i in seq_along(leaves)) expect_equal(env$np$array(leaves[[i]]), env$np$array(plain_leaves[[i]]), tolerance = 1e-4)
  expect_gt(sum(abs(env$np$array(gradients$TemporalTransformer$FF$FFNarrow$weight))), 0)
  concat <- ci_eff_rep(f, aggregation = "concatenate")
  expect_false(any(grepl("Temporal", names(concat$ImageModel_And_State_And_MPPolicy_List[[1]]))))
  expect_equal(dim(concat$ImageRepresentations), c(6L, 48L))
})

test_that("pretrained image embeddings do not allocate an unused trainable transformer", {
  skip_on_cran()
  f <- ci_eff_fixture()
  on.exit(unlink(f$file), add = TRUE)
  env <- causalimages:::cienv
  previous_torch <- if (exists("torch", env, inherits = FALSE)) env$torch else NULL
  on.exit({
    causalimages:::ci_pretrained_cache_clear()
    if (is.null(previous_torch)) rm(list = "torch", envir = env) else env$torch <- previous_torch
  }, add = TRUE)
  causalimages:::ci_pretrained_cache_clear()
  env$torch <- TRUE  # Cached JAX fixture avoids external model downloads.
  env$JAX_Weights <- env$jnp$array(0.)
  reticulate::py_run_string("import jax.numpy as jnp\ndef ci_cached_encoder(weights, args, kwargs):\n    return (jnp.ones((args[0].shape[0], 1, 768)), jnp.ones((args[0].shape[0], 768)))")
  env$JAX_Model <- reticulate::py$ci_cached_encoder
  for (name in c("MEAN_RESCALER", "NORM_MEAN_array_inner")) env[[name]] <- env$jnp$zeros(list(1L, 3L, 1L, 1L))
  for (name in c("SD_RESCALER", "NORM_SD_array_inner")) env[[name]] <- env$jnp$ones(list(1L, 3L, 1L, 1L))
  env$NORM_MEAN_array_inner <- env$jnp$zeros(list(1L, 1L, 1L, 3L))
  env$NORM_SD_array_inner <- env$jnp$ones(list(1L, 1L, 1L, 3L))
  env$nParameters_Pretrained <- 42L
  key <- causalimages:::ci_pretrained_cache_key("vit-base", "image", c(0, 0, 0), c(1, 1, 1), c(1, 16, 16, 3))
  causalimages:::ci_pretrained_cache_save(key)
  rep <- ci_eff_rep(f, pretrained = "vit-base")
  expect_length(rep$ImageModel_And_State_And_MPPolicy_List[[1]], 0L)
  expect_equal(dim(rep$ImageRepresentations), c(6L, 768L))
})

test_that("v1 predictive artifacts still load after compact model construction", {
  skip_on_cran()
  f <- ci_eff_fixture()
  path <- tempfile(fileext = ".eqx")
  old_path <- tempfile(fileext = ".eqx")
  on.exit(unlink(c(f$file, path, paste0(path, ".meta.rds"), old_path, paste0(old_path, ".meta.rds"))), add = TRUE)
  causalimages::PredictiveRun(
    obsY = seq_along(f$keys), file = f$file, imageKeysOfUnits = f$keys,
    nWidth_ImageRep = 16L, nWidth_Dense = 16L, nDepth_ImageRep = 1L,
    batchSize = 2L, nSGD = 2L, testFrac = .5, patchEmbedDim = 4L,
    dropoutRate = 0, droppathRate = 0, useTrainingPerturbations = FALSE,
    plotResults = FALSE, image_dtype = "float32", seed = 42L, modelPath = path, metricsPath = NULL
  )
  manifest <- readRDS(paste0(path, ".meta.rds"))
  expect_identical(manifest$model_structure_version, 2L)
  manifest$artifact_version <- "predictive-v1"
  manifest$model_structure_version <- manifest$representation_width <- NULL
  prepared_X <- causalimages:::ci_predictive_prepare_x(NULL, length(f$keys),
    list(X_mean = manifest$X_mean, X_sd = manifest$X_sd), manifest$x_ncol)$X
  template <- causalimages:::ci_predictive_build_template_bundle(
    manifest, f$file, f$keys, prepared_X, "CausalImagesEnv", TRUE, NULL
  )
  expect_true("MixCLSPool" %in% names(template$ModelList$SpatialTransformerSupp))
  causalimages:::cienv$eq$tree_serialise_leaves(old_path, list(template$ModelList, template$StateList))
  saveRDS(manifest, paste0(old_path, ".meta.rds"))
  expected <- causalimages:::ci_predictive_score_existing_model(
    config = manifest, ModelList = template$ModelList, StateList = template$StateList,
    ModelList_fixed = template$ModelList_fixed, MPList = template$MPList,
    ImageRepArm_batch_R = template$ImageRepArm_batch_R, InitImageProcessFn = template$InitImageProcessFn,
    X = prepared_X, XisNull = TRUE, imageKeysOfUnits = f$keys, file = f$file
  )
  actual <- causalimages::PredictiveScore(old_path, f$file, f$keys, batchSize = 2L)
  expect_equal(actual$predictedY, expected$predictedY, tolerance = 1e-6)
})

test_that("confounding training uses bounded inputs with the requested precision", {
  skip_on_cran()
  f <- ci_eff_fixture(count = 12L)
  on.exit(unlink(f$file), add = TRUE)
  expect_warning(fit <- causalimages::AnalyzeImageConfounding(
    obsW = rep(c(0, 1), 6L), obsY = sin(seq_len(12L)) + rep(c(0, 1), 6L),
    imageKeysOfUnits = f$keys, file = f$file, figuresPath = tempdir(),
    plotResults = FALSE, optimizeImageRep = TRUE, nWidth_ImageRep = 16L,
    nWidth_Dense = 16L, nDepth_ImageRep = 1L, patchEmbedDim = 4L,
    batchSize = 4L, nSGD = 2L, nBoot = 3L, kFolds = 2L,
    dropoutRate = 0, droppathRate = 0, useTrainingPerturbations = FALSE,
    useScalePerturbations = FALSE, image_dtype = "float32", seed = 123L
  ), "nSGD = 2 is low")
  expect_length(fit$SGD_loss_vec, 2L)
  expect_true(all(is.finite(fit$SGD_loss_vec)))
  expect_length(fit$prW_est, length(f$keys))
  expect_true(all(is.finite(fit$prW_est)))
  expect_true(all(fit$prW_est >= 0 & fit$prW_est <= 1))
})
