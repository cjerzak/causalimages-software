# Internal helper utilities for predictive model scoring and persistence.

ci_predictive_meta_path <- function(modelPath) {
  paste0(modelPath, ".meta.rds")
}

ci_resolve_tfrecord_path <- function(filename) {
  tf_record_name <- path.expand(as.character(filename))
  if (!grepl(tf_record_name, pattern = "/")) {
    tf_record_name <- paste("./", tf_record_name, sep = "")
  }
  tf_record_name <- strsplit(tf_record_name, split = "/")[[1]]
  list(
    workdir = paste(tf_record_name[-length(tf_record_name)], collapse = "/"),
    basename = tf_record_name[length(tf_record_name)]
  )
}

parse_tfr_key_element <- function(element) {
  dict_init_val <- list()
  key_feature_description <- dict(
    "index" = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
    "key" = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$string)
  )

  content <- cienv$tf$io$parse_single_example(element, key_feature_description)
  key <- cienv$tf$io$parse_tensor(content[["key"]], out_type = cienv$tf$string)
  list(content[["index"]], key)
}

ci_tfrecord_key_index_map <- function(filename) {
  if (!exists("tfrecord_key_index_cache", envir = cienv, inherits = FALSE)) {
    cienv$tfrecord_key_index_cache <- new.env(parent = emptyenv())
  }

  cache_key <- normalizePath(path.expand(filename), winslash = "/", mustWork = FALSE)
  cache_env <- cienv$tfrecord_key_index_cache
  if (exists(cache_key, envir = cache_env, inherits = FALSE)) {
    return(get(cache_key, envir = cache_env, inherits = FALSE))
  }

  tf_path <- ci_resolve_tfrecord_path(filename)
  orig_wd <- getwd()
  on.exit(try(setwd(orig_wd), silent = TRUE), add = TRUE)
  setwd(tf_path$workdir)

  dataset <- cienv$tf$data$TFRecordDataset(tf_path$basename)
  dataset <- dataset$map(function(x) {
    parse_tfr_key_element(x)
  })
  dataset_iterator <- reticulate::as_iterator(dataset)

  keys <- character(0)
  repeat {
    element <- reticulate::iter_next(dataset_iterator)
    if (is.null(element)) {
      break
    }

    key_value <- unlist(lapply(p2l(element[[2]]$numpy()), as.character))
    keys <- c(keys, key_value)
  }

  if (anyDuplicated(keys)) {
    duplicate_keys <- unique(keys[duplicated(keys)])
    stop(
      sprintf(
        "TFRecord contains duplicated keys, which predictive transport does not support. Examples: %s",
        paste(utils::head(duplicate_keys, 5L), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  key_index_map <- seq_along(keys)
  names(key_index_map) <- keys

  out <- list(keys = keys, key_index_map = key_index_map, n_keys = length(keys))
  assign(cache_key, out, envir = cache_env)
  out
}

ci_predictive_ensure_parent_dir <- function(path_) {
  dir_name <- dirname(path.expand(path_))
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
  }
}

ci_predictive_prepare_x <- function(X, n_obs, x_stats = NULL, x_ncol = NULL) {
  XisNull <- is.null(X)

  if (!XisNull && !"matrix" %in% class(X)) {
    X <- as.matrix(X)
  }

  if (XisNull) {
    if (is.null(x_ncol)) {
      x_ncol <- 2L
    }
    X_raw <- matrix(0, nrow = n_obs, ncol = as.integer(x_ncol))
  } else {
    X_raw <- X
  }

  if (!is.null(x_stats)) {
    if (ncol(X_raw) != length(x_stats$X_mean)) {
      stop(
        sprintf(
          "X has %d columns, but the predictive artifact expects %d columns.",
          ncol(X_raw),
          length(x_stats$X_mean)
        ),
        call. = FALSE
      )
    }
    X_mean <- as.numeric(x_stats$X_mean)
    X_sd <- as.numeric(x_stats$X_sd)
  } else {
    X_mean <- colMeans(X_raw)
    X_sd <- apply(X_raw, 2, stats::sd)
  }

  X_std <- t((t(X_raw) - X_mean) / (0.001 + X_sd))

  list(
    X = X_std,
    X_raw = X_raw,
    XisNull = XisNull,
    X_mean = X_mean,
    X_sd = X_sd,
    x_ncol = ncol(X_raw)
  )
}

ci_predictive_dtype_info <- function(image_dtype_char) {
  if (image_dtype_char == "float16") {
    return(list(image_dtype_tf = cienv$tf$float16, ComputeDtype = cienv$jnp$float16))
  }
  if (image_dtype_char == "bfloat16") {
    return(list(image_dtype_tf = cienv$tf$bfloat16, ComputeDtype = cienv$jnp$bfloat16))
  }
  if (image_dtype_char == "float32") {
    return(list(image_dtype_tf = cienv$tf$float32, ComputeDtype = cienv$jnp$float32))
  }

  stop(
    sprintf("Unsupported image_dtype '%s'.", image_dtype_char),
    call. = FALSE
  )
}

ci_predictive_noop_perturbation <- function(im_, key) {
  im_
}

ci_predictive_init_image_process <- function(
    NORM_MEAN_array,
    NORM_SD_array,
    inputAvePoolingSize,
    dataType,
    useTrainingPerturbations,
    useScalePerturbations,
    trainingPerturbations = ci_predictive_noop_perturbation,
    scalePerturbations = ci_predictive_noop_perturbation) {
  cienv$jax$jit(function(im, key, inference) {
    if (dataType == "image" && length(im$shape) == 3L) {
      im <- cienv$jnp$expand_dims(im, 0L)
    }
    if (dataType == "video" && length(im$shape) == 4L) {
      im <- cienv$jnp$expand_dims(im, 0L)
    }

    im <- (im - NORM_MEAN_array) / NORM_SD_array

    if (inputAvePoolingSize > 1L && dataType == "image") {
      im <- cienv$jax$vmap(function(imm) {
        cienv$jnp$transpose(
          cienv$eq$nn$AvgPool2d(
            kernel_size = as.integer(c(inputAvePoolingSize, inputAvePoolingSize)),
            stride = as.integer(c(inputAvePoolingSize, inputAvePoolingSize))
          )(cienv$jnp$transpose(imm, c(2L, 0L, 1L))),
          c(1L, 2L, 0L)
        )
      }, 0L)(im)
    }

    if (useTrainingPerturbations) {
      im <- cienv$jax$lax$cond(
        inference,
        true_fun = function() { im },
        false_fun = function() {
          trainingPerturbations(im, cienv$jax$random$split(key, im$shape[[1]]))
        }
      )
    }

    if (useScalePerturbations) {
      im <- cienv$jax$lax$cond(
        inference,
        true_fun = function() { im },
        false_fun = function() {
          scalePerturbations(im, cienv$jax$random$split(key, im$shape[[1]]))
        }
      )
    }

    im
  })
}

ci_predictive_build_dense_modules <- function(config, x_ncol, XisNull, seed_base) {
  DenseList <- DenseStateList <- replicate(config$nDepth_Dense, list())
  for (d_ in seq_len(config$nDepth_Dense)) {
    in_features <- ifelse(
      d_ == 1L,
      yes = config$nWidth_ImageRep + ifelse(XisNull, yes = 0L, no = x_ncol * (!config$XCrossModal)),
      no = config$nWidth_Dense
    )
    out_features <- ifelse(d_ == config$nDepth_Dense, yes = 1L, no = config$nWidth_Dense)
    DenseProj_d <- cienv$eq$nn$Linear(
      in_features = in_features,
      out_features = out_features,
      use_bias = TRUE,
      key = ci_jax_key(seed = seed_base, offset = d_ + 44L, label = "Predictive dense layer seed")
    )
    LayerBN_d <- cienv$jnp$array(1)
    DenseStateList[[d_]] <- list("BNState" = cienv$eq$nn$State(LayerBN_d))
    DenseList[[d_]] <- list("DenseProj" = DenseProj_d, "BN" = LayerBN_d)
  }
  names(DenseList) <- names(DenseStateList) <- paste0("Dense", seq_len(config$nDepth_Dense))

  list(DenseList = DenseList, DenseStateList = DenseStateList)
}

ci_predictive_create_mp_list <- function(ComputeDtype) {
  list(
    cienv$jmp$Policy(
      compute_dtype = ComputeDtype,
      param_dtype = "float32",
      output_dtype = ComputeDtype
    ),
    cienv$jmp$DynamicLossScale(
      loss_scale = cienv$jnp$array(2^15, dtype = ComputeDtype),
      min_loss_scale = cienv$jnp$array(2^1., dtype = ComputeDtype),
      period = 50L
    )
  )
}

ci_predictive_build_template_bundle <- function(
    config,
    file,
    imageKeysOfUnits,
    X,
    conda_env,
    conda_env_required,
    Sys.setenv_text) {
  dtype_info <- ci_predictive_dtype_info(config$image_dtype)
  ComputeDtype <- dtype_info$ComputeDtype

  InitImageProcessFn <- ci_predictive_init_image_process(
    NORM_MEAN_array = cienv$jnp$array(config$NORM_MEAN, dtype = ComputeDtype),
    NORM_SD_array = cienv$jnp$array(config$NORM_SD, dtype = ComputeDtype),
    inputAvePoolingSize = config$inputAvePoolingSize,
    dataType = config$dataType,
    useTrainingPerturbations = config$useTrainingPerturbations,
    useScalePerturbations = config$useScalePerturbations
  )

  unique_keys <- unique(as.character(imageKeysOfUnits))
  init_keys <- unique_keys[seq_len(min(length(unique_keys), max(2L, 2L * config$batchSize)))]

  image_context <- GetImageRepresentations(
    X = X,
    file = file,
    dataType = config$dataType,
    temporalAggregation = config$temporalAggregation,
    InitImageProcess = InitImageProcessFn,
    NORM_MEAN = config$NORM_MEAN,
    NORM_SD = config$NORM_SD,
    nWidth_ImageRep = config$nWidth_ImageRep,
    nDepth_ImageRep = config$nDepth_ImageRep,
    strides = config$strides,
    nonLinearScaler = config$nonLinearScaler,
    dropoutRate = config$dropoutRate,
    droppathRate = config$droppathRate,
    nDepth_TemporalRep = config$nDepth_TemporalRep,
    patchEmbedDim = config$patchEmbedDim,
    batchSize = config$batchSize,
    imageModelClass = config$imageModelClass,
    pretrainedModel = config$pretrainedModel,
    image_dtype = ComputeDtype,
    image_dtype_tf = dtype_info$image_dtype_tf,
    optimizeImageRep = config$optimizeImageRep,
    kernelSize = config$kernelSize,
    inputAvePoolingSize = config$inputAvePoolingSize,
    TfRecords_BufferScaler = 3L,
    XCrossModal = config$XCrossModal,
    XForceModal = config$XForceModal,
    imageKeysOfUnits = init_keys,
    getRepresentations = TRUE,
    returnContents = TRUE,
    initializingFxns = TRUE,
    bn_momentum = 0.99,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    Sys.setenv_text = Sys.setenv_text,
    seed = ci_seed_int32(config$seed_template, label = "Predictive template seed")
  )

  dense_modules <- ci_predictive_build_dense_modules(
    config = config,
    x_ncol = ncol(X),
    XisNull = !config$training_has_X,
    seed_base = ci_seed_int32(config$seed_template, offset = 2000L, label = "Predictive dense template seed")
  )

  ModelList <- c(
    image_context[["ImageModel_And_State_And_MPPolicy_List"]][[1]],
    "DenseList" = list(dense_modules$DenseList)
  )
  StateList <- c(
    image_context[["ImageModel_And_State_And_MPPolicy_List"]][[2]],
    "DenseStateList" = list(dense_modules$DenseStateList)
  )
  MPList <- ci_predictive_create_mp_list(ComputeDtype = ComputeDtype)
  ModelList <- MPList[[1]]$cast_to_param(ModelList)
  ModelList_fixed <- MPList[[1]]$cast_to_param(cienv$jnp$array(0.))

  list(
    ModelList = ModelList,
    StateList = StateList,
    ModelList_fixed = ModelList_fixed,
    MPList = MPList,
    ImageRepArm_batch_R = image_context[["ImageRepArm_batch_R"]],
    InitImageProcessFn = image_context[["InitImageProcess"]]
  )
}

ci_predictive_dense_batch_jit <- function(config, XisNull) {
  GetDense_OneObs <- function(ModelList, ModelList_fixed, m, x, seed,
                              StateList, MPList, inference) {
    if (!config$XCrossModal && !XisNull) {
      m <- cienv$jnp$concatenate(list(m, x))
    }

    for (d__ in seq_len(config$nDepth_Dense)) {
      DenseList_d <- ModelList$DenseList[[paste0("Dense", d__)]]
      StateDenseList_d <- StateList$DenseStateList[[paste0("Dense", d__)]]

      m <- DenseList_d$DenseProj(m)

      if (d__ < config$nDepth_Dense) {
        m <- DenseList_d$BN(m, state = StateDenseList_d, inference = inference)
        StateList$DenseStateList[[paste0("Dense", d__)]] <- m[[2]]
        m <- cienv$jax$nn$swish(m[[1]])
      }
    }

    list(m, StateList)
  }

  GetDense_batch <- cienv$jax$vmap(
    function(ModelList, ModelList_fixed, m, x, seed, StateList, MPList, inference) {
      GetDense_OneObs(ModelList, ModelList_fixed, m, x, seed, StateList, MPList, inference)
    },
    in_axes = list(NULL, NULL, 0L, 0L, 0L, NULL, NULL, NULL),
    axis_name = "batch",
    out_axes = list(0L, NULL)
  )

  cienv$eq$filter_jit(GetDense_batch)
}

ci_predictive_score_existing_model <- function(
    config,
    ModelList,
    StateList,
    ModelList_fixed,
    MPList,
    ImageRepArm_batch_R,
    InitImageProcessFn,
    X,
    XisNull,
    imageKeysOfUnits,
    file) {
  if (length(imageKeysOfUnits) == 0L) {
    stop("imageKeysOfUnits must contain at least one key for predictive scoring.", call. = FALSE)
  }

  imageKeysOfUnits <- as.character(imageKeysOfUnits)
  tfrecord_info <- ci_tfrecord_key_index_map(file)
  requested_unique_keys <- unique(imageKeysOfUnits)
  requested_indices <- tfrecord_info$key_index_map[requested_unique_keys]

  if (any(is.na(requested_indices))) {
    missing_keys <- requested_unique_keys[is.na(requested_indices)]
    stop(
      sprintf(
        "Some requested keys are missing from the tfrecord. Examples: %s",
        paste(utils::head(missing_keys, 5L), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  key_request_table <- data.frame(
    key = requested_unique_keys,
    tf_idx = as.integer(requested_indices),
    stringsAsFactors = FALSE
  )
  key_request_table <- key_request_table[order(key_request_table$tf_idx), , drop = FALSE]

  obs_indices_by_key <- split(seq_along(imageKeysOfUnits), factor(imageKeysOfUnits, levels = requested_unique_keys))
  useVideoIndicator <- config$dataType == "video"
  dtype_info <- ci_predictive_dtype_info(config$image_dtype)
  image_dtype_tf <- dtype_info$image_dtype_tf
  ComputeDtype <- dtype_info$ComputeDtype
  inner_batch_size <- ai(max(2L, config$batchSize))
  outer_batch_size <- ai(max(2L, 2L * config$batchSize))

  ImageRepArm_batch_jit <- cienv$eq$filter_jit(ImageRepArm_batch_R)
  GetDense_batch_jit <- ci_predictive_dense_batch_jit(config = config, XisNull = XisNull)

  batch_starts <- seq(1L, nrow(key_request_table), by = outer_batch_size)
  passedIterator <- NULL
  Results_by_keys <- list()
  result_counter <- 0L
  inference_counter <- 0L

  for (b in seq_along(batch_starts)) {
    batch_rows <- batch_starts[b]:min(batch_starts[b] + outer_batch_size - 1L, nrow(key_request_table))
    batch_request <- key_request_table[batch_rows, , drop = FALSE]

    ds_next_in <- GetElementFromTfRecordAtIndices(
      uniqueKeyIndices = batch_request$tf_idx,
      filename = file,
      iterator = passedIterator,
      readVideo = useVideoIndicator,
      image_dtype = image_dtype_tf,
      nObs = tfrecord_info$n_keys,
      return_iterator = TRUE
    )
    tmp_updated_iterator <- ds_next_in[[2]]
    outerBatchKeys <- unlist(lapply(p2l(ds_next_in[[1]][[3]]$numpy()), as.character))
    ds_next_in <- cienv$jnp$array(ds_next_in[[1]][[1]])

    if (length(ds_next_in$shape) == 3L && config$dataType == "image") {
      ds_next_in <- cienv$jnp$expand_dims(ds_next_in, 0L)
    }
    if (length(ds_next_in$shape) == 4L && config$dataType == "video") {
      ds_next_in <- cienv$jnp$expand_dims(ds_next_in, 0L)
    }

    if (!identical(as.character(outerBatchKeys), as.character(batch_request$key))) {
      stop("Key pairing mismatch in predictive inference; check tfrecord key ordering.", call. = FALSE)
    }

    keyNames_xIndicesValues <- do.call(
      rbind,
      lapply(batch_request$key, function(key__) {
        values__ <- obs_indices_by_key[[key__]]
        cbind("key" = rep(key__, times = length(values__)), "value" = values__)
      })
    )
    keyNames_xIndicesValues_names <- keyNames_xIndicesValues[, 1]
    keyNames_xIndicesValues <- f2n(keyNames_xIndicesValues[, 2])
    names(keyNames_xIndicesValues) <- keyNames_xIndicesValues_names

    batchStarts_inner <- seq(1L, length(keyNames_xIndicesValues), by = inner_batch_size)
    for (bi_ in seq_along(batchStarts_inner)) {
      inference_counter <- inference_counter + 1L

      idx_start_inner <- batchStarts_inner[bi_]
      idx_end_inner <- min(idx_start_inner + inner_batch_size - 1L, length(keyNames_xIndicesValues))

      in_xbatch_indices <- idx_start_inner:idx_end_inner
      x_indices <- keyNames_xIndicesValues[in_xbatch_indices]
      x_indices <- x_indices[!is.na(x_indices)]
      realSize_inner <- length(x_indices)
      if (realSize_inner == 0L) {
        next
      }

      position_indices <- seq_len(realSize_inner)
      if (realSize_inner < inner_batch_size) {
        position_indices <- c(position_indices, rep(realSize_inner, inner_batch_size - realSize_inner))
      }

      m_indices <- match(names(x_indices), outerBatchKeys)
      image_batch <- cienv$jnp$take(
        ds_next_in,
        cienv$jnp$array(ai(m_indices - 1L)),
        axis = 0L
      )
      m <- InitImageProcessFn(
        image_batch,
        ci_jax_key(seed = config$seed_template, offset = 600L + inference_counter, label = "Predictive inference init seed"),
        inference = TRUE
      )
      if (realSize_inner < inner_batch_size) {
        m <- cienv$jnp$take(m, cienv$jnp$array(ai(position_indices - 1L)), axis = 0L)
      }

      x_indices_padded <- if (realSize_inner < inner_batch_size) {
        c(x_indices, rep(x_indices[realSize_inner], inner_batch_size - realSize_inner))
      } else {
        x_indices
      }
      x_batch <- cienv$jnp$array(X[x_indices_padded, , drop = FALSE], dtype = ComputeDtype)

      m_ImageRep <- ImageRepArm_batch_jit(
        ModelList,
        m,
        x_batch,
        StateList,
        cienv$jax$random$split(
          ci_jax_key(seed = config$seed_template, offset = 900L + inference_counter, label = "Predictive inference representation seed"),
          inner_batch_size
        ),
        MPList,
        TRUE
      )[[1]]

      predictions <- GetDense_batch_jit(
        ModelList,
        ModelList_fixed,
        m_ImageRep,
        x_batch,
        cienv$jax$random$split(
          ci_jax_key(seed = config$seed_template, offset = 1200L + inference_counter, label = "Predictive inference dense seed"),
          inner_batch_size
        ),
        StateList,
        MPList,
        TRUE
      )[[1]]

      if (config$is_binary) {
        predictions <- cienv$jax$nn$sigmoid(predictions)
      }

      predictions <- as.matrix(cienv$np$array(predictions))[seq_len(realSize_inner), , drop = FALSE]
      result_counter <- result_counter + 1L
      Results_by_keys[[result_counter]] <- list(
        "PredY" = predictions,
        "obsIndex" = as.matrix(x_indices[seq_len(realSize_inner)]),
        "key" = as.matrix(names(x_indices[seq_len(realSize_inner)]))
      )
    }

    passedIterator <- tmp_updated_iterator
  }

  Results_by_keys <- do.call(rbind.data.frame, Results_by_keys)
  Results_by_keys <- Results_by_keys[order(f2n(Results_by_keys$obsIndex)), , drop = FALSE]

  predictedY <- f2n(Results_by_keys$PredY)
  if (any(is.na(predictedY))) {
    warning("NAs in predictions. Imputing them with the mean predicted value.", call. = FALSE)
    predictedY[is.na(predictedY)] <- mean(predictedY, na.rm = TRUE)
  }

  list(predictedY = predictedY, details = Results_by_keys)
}

ci_predictive_make_manifest <- function(
    config,
    NORM_MEAN,
    NORM_SD,
    X_mean,
    X_sd,
    x_ncol,
    training_has_X) {
  c(
    config,
    list(
      NORM_MEAN = as.numeric(NORM_MEAN),
      NORM_SD = as.numeric(NORM_SD),
      X_mean = as.numeric(X_mean),
      X_sd = as.numeric(X_sd),
      x_ncol = as.integer(x_ncol),
      training_has_X = isTRUE(training_has_X),
      package_version = as.character(utils::packageVersion("causalimages")),
      artifact_version = "predictive-v1"
    )
  )
}

ci_predictive_validate_transport_x <- function(training_has_X, XTransport, context) {
  if (training_has_X && is.null(XTransport)) {
    stop(
      sprintf("%s requires X transport covariates because the predictive model was trained with X.", context),
      call. = FALSE
    )
  }
  if (!training_has_X && !is.null(XTransport)) {
    stop(
      sprintf("%s received X transport covariates, but the predictive model was trained without X.", context),
      call. = FALSE
    )
  }
}

#' Score a saved predictive model on a tfrecord
#'
#' Loads a predictive artifact written by `PredictiveRun()` and returns outcome
#' predictions for the requested image keys.
#'
#' @param modelPath Path to a predictive artifact generated by `PredictiveRun()`.
#' @param file Path to a tfrecord file generated by `WriteTfRecord()`.
#' @param imageKeysOfUnits A vector of image keys to score, one per requested observation.
#' @param X Optional numeric matrix of transport covariates. Required when the saved
#'   predictive model was trained with tabular covariates.
#' @param batchSize Optional batch size for scoring. Defaults to the value saved in the artifact.
#' @param conda_env A `conda` environment where the computational backend lives.
#' @param conda_env_required A Boolean stating whether use of the specified conda environment is required.
#' @param Sys.setenv_text Optional string for setting environment variables before Python initialization.
#' @param atError String specifying behavior on error. Options are `"stop"` (default) or `"debug"`.
#' @param seed Optional integer for reproducibility.
#'
#' @return Returns a list containing:
#' \itemize{
#'   \item `predictedY` Predicted values for the requested observations.
#'   \item `obsIndex` Observation indices corresponding to the supplied `imageKeysOfUnits`.
#'   \item `imageKeysOfUnits` The scored image keys, returned in observation order.
#' }
#'
#' @examples
#' \dontrun{
#' preds <- PredictiveScore(
#'   modelPath = "./trained_model.eqx",
#'   file = "./new_data.tfrecord",
#'   imageKeysOfUnits = new_keys
#' )
#' }
#'
#' @export
#' @md
PredictiveScore <- function(
    modelPath,
    file,
    imageKeysOfUnits,
    X = NULL,
    batchSize = NULL,
    conda_env = "CausalImagesEnv",
    conda_env_required = TRUE,
    Sys.setenv_text = NULL,
    atError = "stop",
    seed = NULL) {
  if (is.null(modelPath) || !file.exists(path.expand(modelPath))) {
    stop("modelPath must point to an existing predictive artifact.", call. = FALSE)
  }

  meta_path <- ci_predictive_meta_path(modelPath)
  if (!file.exists(meta_path)) {
    stop("The predictive artifact metadata sidecar is missing.", call. = FALSE)
  }

  manifest <- readRDS(meta_path)
  if (!is.list(manifest) || is.null(manifest$artifact_version)) {
    stop("The predictive artifact metadata is invalid.", call. = FALSE)
  }

  if (pretrained_model_requires_torch(manifest$pretrainedModel)) {
    if (!"torch" %in% ls(envir = cienv)) {
      initialize_torch(
        conda_env = conda_env,
        conda_env_required = conda_env_required,
        Sys.setenv_text = Sys.setenv_text
      )
    }
  }
  if (!ci_backend_ready()) {
    initialize_jax(
      conda_env = conda_env,
      conda_env_required = conda_env_required,
      Sys.setenv_text = Sys.setenv_text
    )
  }

  manifest$batchSize <- ifelse(is.null(batchSize), yes = manifest$batchSize, no = ai(batchSize))
  if (is.null(seed)) {
    seed <- manifest$seed_template
  }
  manifest$seed_template <- ci_int32_scalar(seed, "PredictiveScore seed")

  ci_predictive_validate_transport_x(
    training_has_X = manifest$training_has_X,
    XTransport = X,
    context = "PredictiveScore"
  )

  prepared_X <- ci_predictive_prepare_x(
    X = X,
    n_obs = length(imageKeysOfUnits),
    x_stats = list(X_mean = manifest$X_mean, X_sd = manifest$X_sd),
    x_ncol = manifest$x_ncol
  )

  template_bundle <- ci_predictive_build_template_bundle(
    config = manifest,
    file = file,
    imageKeysOfUnits = imageKeysOfUnits,
    X = prepared_X$X,
    conda_env = conda_env,
    conda_env_required = conda_env_required,
    Sys.setenv_text = Sys.setenv_text
  )
  loaded_bundle <- cienv$eq$tree_deserialise_leaves(
    path.expand(modelPath),
    list(template_bundle$ModelList, template_bundle$StateList)
  )

  score_out <- try(
    ci_predictive_score_existing_model(
      config = manifest,
      ModelList = loaded_bundle[[1]],
      StateList = loaded_bundle[[2]],
      ModelList_fixed = template_bundle$ModelList_fixed,
      MPList = template_bundle$MPList,
      ImageRepArm_batch_R = template_bundle$ImageRepArm_batch_R,
      InitImageProcessFn = template_bundle$InitImageProcessFn,
      X = prepared_X$X,
      XisNull = !manifest$training_has_X,
      imageKeysOfUnits = imageKeysOfUnits,
      file = file
    ),
    silent = TRUE
  )
  if ("try-error" %in% class(score_out)) {
    if (atError == "debug") {
      browser()
    }
    stop(conditionMessage(attr(score_out, "condition")), call. = FALSE)
  }

  list(
    predictedY = score_out$predictedY,
    obsIndex = f2n(score_out$details$obsIndex),
    imageKeysOfUnits = as.character(score_out$details$key)
  )
}
