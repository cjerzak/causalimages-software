#' Get the spatial point of long/lat coordinates
#'
#' Convert longitude and latitude coordinates to a different coordinate reference
#' system (CRS).
#'
#' @param long Vector of numeric longitudes.
#' @param lat Vector of numeric latitudes.
#' @param CRS_ref A CRS into which the long-lat point should be projected.
#'
#' @return Numeric vector of length two giving the coordinates of the supplied
#'   location in the CRS defined by `CRS_ref`.
#'
#' @importFrom grDevices dev.off hcl.colors pdf
#' @importFrom graphics abline axis layout legend mtext par points
#' @importFrom stats coef cor lm na.omit resid sd smooth.spline var
#' @importFrom utils capture.output setTxtProgressBar str txtProgressBar
#'
#' @examples
#' # (Not run)
#' #spatialPt <- LongLat2CRS(long = 49.932,
#' #                 lat = 35.432,
#' #                 CRS_ref = sf::st_crs("+proj=lcc +lat_1=48 +lat_2=33 +lon_0=-100 +ellps=WGS84"))
#' @export
#' @md
#'
LongLat2CRS <- function(long, lat, CRS_ref){
  point_longlat <- sf::st_as_sf(
    data.frame(long = as.numeric(long), lat = as.numeric(lat)),
    coords = c("long", "lat"),
    crs = 4326
  )
  point_longlat_ref <- sf::st_transform(point_longlat, crs = sf::st_crs(CRS_ref))
  coords_ <- sf::st_coordinates(point_longlat_ref)[1, ]
  return(coords_)
}

LongLat2CRS_extent <- function(point_longlat,
                               CRS_ref,
                               target_km_diameter = 10){
  target_km <- target_km_diameter
  offset <- 1/111 * (target_km/2)
  point_longlat1 <- c(long = as.numeric(point_longlat[1]) - offset,
                      lat = as.numeric(point_longlat[2]) - offset)
  point_longlat2 <- c(long = as.numeric(point_longlat[1]) + offset,
                      lat = as.numeric(point_longlat[2]) + offset)
  pts <- sf::st_as_sf(rbind(point_longlat1, point_longlat2),
                      coords = c("long", "lat"), crs = 4326)
  pts_ref <- sf::st_transform(pts, crs = sf::st_crs(CRS_ref))
  coords_ <- sf::st_coordinates(pts_ref)
  return(raster::extent(min(coords_[,1]), max(coords_[,1]),
                        min(coords_[,2]), max(coords_[,2])))
}

# converts python builtin to list
p2l <- function(zer){
  if("python.builtin.bytes" %in% class(zer)){ zer <- list(zer) }
  return( zer )
}

# zips two lists
rzip<-function(l1,l2){  fl<-list(); for(aia in 1:length(l1)){ fl[[aia]] <- list(l1[[aia]], l2[[aia]]) }; return( fl  ) }

# reshapes
reshape_fxn_DEPRECIATED <- function(input_){
    ## DEPRECIATED
    cienv$tf$reshape(input_, list(cienv$tf$shape(input_)[1],
                            cienv$tf$reduce_prod(cienv$tf$shape(input_)[2:5])))
}

fixZeroEndings <- function(zr,roundAt=2){
  unlist( lapply(strsplit(as.character(zr),split="\\."),function(l_){
    if(length(l_) == 1){ retl <- paste(l_, paste(rep("0",times=roundAt),collapse=""),sep=".") }
    if(length(l_) == 2){
      retl <- paste(l_[1], paste(l_[2], paste(rep("0",times=roundAt-nchar(l_[2])),collapse=""),sep=""),
                    sep = ".") }
    return( retl  )
  }) ) }

r2const <- function(x, dtype){
  if("tensorflow.tensor" %in% class( x )){ x <- cienv$tf$cast(x, dtype = dtype  ) }
  if(!"tensorflow.tensor" %in% class( x )){ x <- cienv$tf$constant(x, dtype = dtype  ) }
  return( x )
}

#' print2  print() with timestamps
#'
#' A function that prints a string with date and time.
#'
#' @param text Character string to be printed, with date and time.
#' @param quiet Logical. If TRUE, suppresses the print output. Default is FALSE.
#'
#' @return Prints with date and time.
#'
#' @examples
#' print2("Hello world")
#' @export
#' @md
#'
print2 <- function(text, quiet = F){
  if(!quiet){ print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) ) }
}

#' message2 message() with timestamps
#'
#' A function that displays a message with date and time. 
#'
#' @param text Character string to be displayed as message, with date and time. 
#' @param quiet Logical. If TRUE, suppresses the message output. Default is FALSE.
#'
#' @return Displays message with date and time to stderr. 
#'
#' @examples
#' message2("Hello world")
#' message2("Process completed", quiet = FALSE)
#' @export
#' @md
#'
message2 <- function(text, quiet = FALSE){
  if(!quiet){ 
    message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), text)) 
  }
}

# LE <- function(l_, name_){ return( unlist(l_)[[name_]] ) }
# l_ <- DenseList;name <-"Tau_d1"
LE <- function(l_, key) {
  # Recursive helper function
  search_recursive <- function(list_element, key) {
    # Check if the current element is a list
    if (is.list(list_element)) {
      # If it's a list, check if the key exists in this list
      if (key %in% names(list_element)) {
        return(list_element[[key]])
      }
      # Otherwise, iterate over its elements
      for (item in list_element) {
        found <- search_recursive(item, key)
        if (!is.null(found)) {
          return(found)
        }
      }
    }
    return(NULL)
  }

  # Start the recursive search
  return(search_recursive(l_, key))
}

LE_index <- function(l_, key) {
  # Recursive helper function
  search_recursive <- function(list_element, key, path) {
    # Check if the current element is a list
    if(is.list(list_element)){

      # If it's a list, check if the key exists in this list
      if(key %in% names(list_element) & (length(names(list_element)) == 1)){
        return( c(path,
                  ifelse("list" %in% class(list_element), yes = 1, no = NULL)) )
      }
      if(key %in% names(list_element) & (length(names(list_element)) > 1)){
        return( c(path,
                  which(names(list_element) == key),
                  ifelse("list" %in% class(list_element), yes = 1, no = NULL) ) )
      }

      # Otherwise, iterate over its elements
      for (i in seq_along(list_element)) {
        new_path <- c(path, i)
        found <- search_recursive(list_element[[i]], key, new_path)
        if (!is.null(found)) { return( found ) }
      }
    }
    return(NULL)
  }

  # Start the recursive search
  return(search_recursive(l_, key, c()))
}

GlobalPartition <- function(zer, eq_fxn){
  yes_branches <- rrapply::rrapply(zer,f=function(zerr){
    unlist(ifelse(eq_fxn(zerr),yes = list(zerr), no = list(NULL))[[1]])
  },how="list")
  no_branches <- rrapply::rrapply(zer,f=function(zerrr){
    unlist(ifelse(eq_fxn(zerrr),yes = list(NULL), no = list(zerrr))[[1]])
  },how="list")
  list(yes_branches,no_branches)
}
PartFxn <- function(zerz){ !"first_time_index" %in% names(zerz)}

AddQuotes <- function(text) { gsub("\\[\\[([A-Za-z]\\w*)", "\\[\\['\\1'", text) }
LinearizeNestedList <- function (NList,
                                 LinearizeDataFrames = FALSE,
                                 NameSep = "/",
                                 ForceNames = FALSE){
  stopifnot(is.character(NameSep), length(NameSep) == 1)
  stopifnot(is.logical(LinearizeDataFrames), length(LinearizeDataFrames) == 1)
  stopifnot(is.logical(ForceNames), length(ForceNames) == 1)
  if (!is.list(NList))
    return(NList)
  if (is.null(names(NList)) | ForceNames == TRUE)
    names(NList) <- as.character(1:length(NList))
  if (is.data.frame(NList) & LinearizeDataFrames == FALSE)
    return(NList)
  if (is.data.frame(NList) & LinearizeDataFrames == TRUE)
    return(as.list(NList))
  A <- 1
  B <- length(NList)
  while (A <= B) {
    Element <- NList[[A]]
    EName <- names(NList)[A]
    if (is.list(Element)) {
      Before <- if (A == 1)
        NULL
      else NList[1:(A - 1)]
      After <- if (A == B)
        NULL
      else NList[(A + 1):B]
      if (is.data.frame(Element)) {
        if (LinearizeDataFrames == TRUE) {
          Jump <- length(Element)
          NList[[A]] <- NULL
          if(is.null(names(Element)) | ForceNames == TRUE)
            names(Element) <- as.character(1:length(Element))
          Element <- as.list(Element)
          names(Element) <- paste(EName, names(Element),
                                  sep = NameSep)
          NList <- c(Before, Element, After)
        }
        Jump <- 1
      }
      else {
        NList[[A]] <- NULL
        if (is.null(names(Element)) | ForceNames == TRUE)
          names(Element) <- as.character(1:length(Element))
        Element <- LinearizeNestedList(Element, LinearizeDataFrames,
                                       NameSep, ForceNames)
        names(Element) <- AddQuotes( paste(EName,
                                           names(Element),
                                           sep = NameSep) )
        Jump <- length(Element)
        NList <- c(Before, Element, After)
      }
    }
    else {
      Jump <- 1
    }
    A <- A + Jump
    B <- length(NList)
  }
  return(NList)
}


ai <- as.integer

ci_int32_scalar <- function(value, label = "value") {
  if (length(value) != 1L) {
    stop(sprintf("%s must be a scalar.", label), call. = FALSE)
  }

  value_num <- suppressWarnings(as.numeric(value))
  if (is.na(value_num) || !is.finite(value_num)) {
    stop(sprintf("%s must be a finite numeric scalar.", label), call. = FALSE)
  }

  rounded_value <- round(value_num)
  if (!isTRUE(all.equal(value_num, rounded_value, tolerance = sqrt(.Machine$double.eps)))) {
    stop(sprintf("%s must be an integer-like value, got %s.", label, format(value_num)),
         call. = FALSE)
  }

  if (rounded_value < -2147483648 || rounded_value > 2147483647) {
    stop(
      sprintf(
        "%s=%s is outside the supported 32-bit integer range [%s, %s].",
        label,
        format(rounded_value, scientific = FALSE, trim = TRUE),
        "-2147483648",
        "2147483647"
      ),
      call. = FALSE
    )
  }

  as.integer(rounded_value)
}

ci_seed_int32 <- function(seed, offset = 0L, label = "seed") {
  seed_value <- as.numeric(ci_int32_scalar(seed, label = label))
  offset_value <- as.numeric(ci_int32_scalar(offset, label = sprintf("%s offset", label)))
  combined_value <- seed_value + offset_value

  if (combined_value < -2147483648 || combined_value > 2147483647) {
    stop(
      sprintf(
        "%s + offset overflowed the supported 32-bit integer range: %s + %s = %s.",
        label,
        format(seed_value, scientific = FALSE, trim = TRUE),
        format(offset_value, scientific = FALSE, trim = TRUE),
        format(combined_value, scientific = FALSE, trim = TRUE)
      ),
      call. = FALSE
    )
  }

  as.integer(combined_value)
}

ci_jax_key <- function(seed, offset = 0L, label = "seed") {
  cienv$jax$random$key(ci_seed_int32(seed = seed, offset = offset, label = label))
}

ci_image_dtype_info <- function(image_dtype_char) {
  image_dtype_char <- as.character(image_dtype_char)
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

ci_with_wd <- function(path, expr) {
  old_wd <- getwd()
  on.exit(try(setwd(old_wd), silent = TRUE), add = TRUE)
  setwd(path.expand(as.character(path)))
  force(expr)
}

ci_cache_value_text <- function(x) {
  if (is.null(x)) {
    return("<NULL>")
  }

  numeric_value <- suppressWarnings(try(as.numeric(x), silent = TRUE))
  if (!inherits(numeric_value, "try-error") && length(numeric_value) > 0L &&
      all(is.finite(numeric_value) | is.na(numeric_value))) {
    return(paste(format(signif(numeric_value, 12L), scientific = TRUE), collapse = ","))
  }

  if (exists("np", envir = cienv, inherits = FALSE)) {
    numeric_value <- suppressWarnings(try(as.numeric(cienv$np$array(x)), silent = TRUE))
    if (!inherits(numeric_value, "try-error") && length(numeric_value) > 0L) {
      return(paste(format(signif(numeric_value, 12L), scientific = TRUE), collapse = ","))
    }
  }

  paste(capture.output(str(x, max.level = 1L)), collapse = " ")
}

ci_pretrained_cache_names <- function() {
  c(
    "EXPECTED_IMAGE_SIZE",
    "FeatureExtractor",
    "GENERIC_TRANSFORMERS_MODEL",
    "JAX_CLIP_Feature_Model",
    "JAX_CLIP_Feature_Weights",
    "JAX_Model",
    "JAX_Weights",
    "MEAN_RESCALER",
    "NORM_MEAN_array_inner",
    "NORM_SD_array_inner",
    "RunDtype",
    "RunOnDevice",
    "SD_RESCALER",
    "ScaleResizeTranspose",
    "TRANSFORMERS_MODEL_NAME",
    "TransformersModel",
    "TransformersProcessor",
    "ClayModel",
    "nParameters_Pretrained",
    "nWidth_ImageRep"
  )
}

ci_pretrained_cache_env <- function() {
  if (!exists("pretrained_model_cache", envir = cienv, inherits = FALSE)) {
    cienv$pretrained_model_cache <- new.env(parent = emptyenv())
  }
  cienv$pretrained_model_cache
}

ci_pretrained_cache_key <- function(pretrainedModel, dataType, NORM_MEAN, NORM_SD,
                                    rawShape = NULL) {
  paste(
    "pretrained",
    as.character(pretrainedModel),
    as.character(dataType),
    ci_cache_value_text(rawShape),
    ci_cache_value_text(NORM_MEAN),
    ci_cache_value_text(NORM_SD),
    sep = "|"
  )
}

ci_pretrained_cache_activate <- function(cache_key,
                                         names = ci_pretrained_cache_names()) {
  loaded_names <- intersect(names, ls(envir = cienv, all.names = TRUE))
  if (length(loaded_names) > 0L) {
    rm(list = loaded_names, envir = cienv)
  }

  cienv$active_pretrained_cache_key <- cache_key
  cache_env <- ci_pretrained_cache_env()
  if (!exists(cache_key, envir = cache_env, inherits = FALSE)) {
    return(invisible(FALSE))
  }

  entry <- get(cache_key, envir = cache_env, inherits = FALSE)
  for (nm in names(entry)) {
    assign(nm, entry[[nm]], envir = cienv)
  }

  invisible(TRUE)
}

ci_pretrained_cache_save <- function(cache_key,
                                     names = ci_pretrained_cache_names()) {
  loaded_names <- intersect(names, ls(envir = cienv, all.names = TRUE))
  entry <- lapply(loaded_names, function(nm) get(nm, envir = cienv, inherits = FALSE))
  names(entry) <- loaded_names
  assign(cache_key, entry, envir = ci_pretrained_cache_env())
  invisible(entry)
}

se <- function(x){ x <- c(na.omit(x)); return(sqrt(var(x)/length(x)))}

LocalFxnSource <- function(fxn, evaluation_environment){
  eval(body(fxn), envir = evaluation_environment)
}

FilterBN <- function(l_){ cienv$eq$partition(l_, function(l__){"first_time_index" %in% names(l__)}) }

cienv <- new.env( parent = emptyenv() )


dropout_layer_init <- function(p) {
  if (p == 0) { return( function(x, key, inference) { return( x ) } )  }
  if (p != 0) {
    keep_prob <- (1 - p)
    return( 
      function(x, key, inference) {
        # Efficient dynamic branch: skip dropout at inference time
        cienv$jax$lax$cond(
          pred = inference, 
          true_fun = function(args){ return(args[[1]]) },
          false_fun = function(args){
            mask <- cienv$jax$lax$stop_gradient(
              #cienv$jax$random$bernoulli(key = args[[2]], p = keep_prob, shape = args[[1]]$shape)$astype(args[[1]])
              cienv$jax$random$bernoulli(key = args[[2]], p = keep_prob, shape = args[[1]]$shape)$astype( args[[1]]$dtype )
            )
            return( args[[1]] * mask / keep_prob )
          },
          operand = list(x, key) # pack arguments
        ) }
    )
  }}

wt_init <- function(shape, seed_key){
  init_std <- sqrt(2.0 / as.numeric(shape[[1]] + shape[[1]]))
  cienv$jax$random$normal(
    key = seed_key,
    shape = shape
  ) * cienv$jnp$array(init_std)$astype( cienv$jaxFloatType )
}
