# Internal memory controls. Options are read when a pipeline/model is built.
ci_runtime <- function() {
  if (!exists("runtime", envir = cienv, inherits = FALSE)) {
    path <- system.file("python", package = "causalimages")
    cienv$runtime <- reticulate::import_from_path("causalimages_runtime", path = path)
  }
  cienv$runtime
}

ci_memory_option <- function(name, default) {
  value <- getOption(paste0("causalimages.", name), default)
  if (length(value) != 1L || !is.numeric(value) || !is.finite(value) || value < 1) {
    stop(sprintf("causalimages.%s must be a positive number", name), call. = FALSE)
  }
  floor(value)
}

ci_bounded_shuffle <- function(dataset, buffer_size, reshuffle_each_iteration = TRUE) {
  ci_runtime()$bounded_shuffle(
    dataset, as.integer(buffer_size),
    ci_memory_option("input_buffer_bytes", 64 * 1024^2),
    reshuffle_each_iteration
  )
}

ci_bounded_prefetch <- function(dataset) {
  ci_runtime()$bounded_prefetch(
    dataset, as.integer(ci_memory_option("prefetch_batches", 1L)),
    as.integer(ci_memory_option("input_threads", 2L))
  )
}

ci_torch_inference <- function(expr) {
  context <- cienv$torch$no_grad()
  context$`__enter__`()
  on.exit(context$`__exit__`(NULL, NULL, NULL), add = TRUE)
  force(expr)
}

ci_fold_keys <- function(keys, index) ci_runtime()$fold_keys(keys, index)

ci_transformer_parameters <- function(type, width, depth, seed_key, nonLinearScaler,
                                      total_depth, video, patch_size, channels,
                                      x_projection = NULL) {
  wide <- as.integer(width * 3.5)
  inverse_softplus <- function(y) {
    y <- cienv$jnp$clip(y, 1e-6, 1e6)
    y + cienv$jnp$log1p(-cienv$jnp$exp(-y))
  }
  rand_array <- function(x, key) {
    x <- cienv$jnp$array(x)
    x + cienv$jax$random$normal(shape = x$shape, dtype = x$dtype, key = key) * 0.0001
  }
  offset <- if (type == "Spatial") 0L else 10000L
  key_for <- function(n) seed_key(offset = offset + n)
  create_layer <- function(key, layer) {
    if (!is.null(nonLinearScaler)) {
      scale <- rand_array(rep(nonLinearScaler, width), key)
    } else if (video) {
      scale <- rand_array(rep((2 * total_depth)^(-1/2), width), key)
    } else {
      scale <- cienv$jnp$broadcast_to(cienv$jnp$array(depth, cienv$jnp$float32)^(-1/(2*layer-2)), as.integer(width))
    }
    key <- cienv$jax$random$split(key)[[1L]]
    norm <- list(NormScaler1 = rand_array(t(rep(1, width)), key),
                 NormScaler2 = rand_array(t(rep(1, width)), key))
    key <- cienv$jax$random$split(key)[[1L]]
    key <- cienv$jax$random$split(key, 4L)
    attention <- list(W_q = wt_init(list(width, width), key[0L]),
                      W_k = wt_init(list(width, width), key[1L]),
                      W_v = wt_init(list(width, width), key[2L]),
                      W_o = wt_init(list(width, width), key[3L]))
    key <- cienv$jax$random$split(key[[1L]])[[1L]]
    key <- cienv$jax$random$split(key, 3L)
    ff <- list(
      FFWide1 = cienv$eq$nn$Linear(width, wide, use_bias = FALSE, key = key[0L]),
      FFWide2 = cienv$eq$nn$Linear(width, wide, use_bias = FALSE, key = key[1L]),
      FFNarrow = cienv$eq$nn$Linear(wide, width, use_bias = FALSE, key = key[2L])
    )
    list(Multihead = attention, FF = ff,
         ResidualWts = list(RightWt1 = inverse_softplus(scale), RightWt2 = inverse_softplus(scale)),
         TransformerRenormer = norm)
  }
  layers <- cienv$jax$vmap(create_layer, in_axes = list(0L, 0L), out_axes = 0L)(
    cienv$jax$random$split(key_for(0L), depth), cienv$jnp$array(as.matrix(seq_len(depth)))
  )
  supp <- list(
    StartEmbed = cienv$jax$random$uniform(key_for(333324L), minval = -sqrt(6/width), maxval = sqrt(6/width), shape = list(1L, width)),
    StopEmbed = cienv$jax$random$uniform(key_for(33326124L), minval = -sqrt(6/width), maxval = sqrt(6/width), shape = list(1L, width))
  )
  if (type == "Spatial") {
    supp$PatchEmbedder <- cienv$eq$nn$Conv(
      kernel_size = as.integer(c(patch_size, patch_size)), num_spatial_dims = 2L,
      stride = as.integer(c(patch_size, patch_size)), padding_mode = "ZEROS",
      in_channels = channels, out_channels = width, use_bias = TRUE, key = key_for(1044L)
    )
    if (!is.null(x_projection)) supp$XProj <- x_projection
  }
  supp$PoolProject <- cienv$eq$nn$Linear(width, width, use_bias = TRUE, key = key_for(332415L))
  supp$PoolMultihead <- list(
    W_q = wt_init(list(width, width), key_for(3325L)),
    W_k = wt_init(list(width, width), key_for(3415L)),
    W_v = wt_init(list(width, width), key_for(32415L)),
    W_o = wt_init(list(width, width), key_for(32416L))
  )
  supp$FinalNormScaler <- cienv$jnp$ones(list(width), dtype = cienv$jnp$float32)
  stats::setNames(list(layers, supp), paste0(type, c("Transformer", "TransformerSupp")))
}
