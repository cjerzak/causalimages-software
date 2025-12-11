#' Write an image corpus as a .tfrecord file
#'
#' Writes an image corpus to a `.tfrecord` file for rapid reading of images into memory for fast ML training. 
#' Specifically, this function serializes an image or video corpus into a `.tfrecord` file, enabling efficient data loading for machine learning tasks, particularly for image-based causal inference training.
#' It requires that users define an `acquireImageFxn` function that accepts keys and returns the corresponding image or video as an array of dimensions `(length(keys), nSpatialDim1, nSpatialDim2, nChannels)` for images or `(length(keys), nTimeSteps, nSpatialDim1, nSpatialDim2, nChannels)` for video sequences.

#'
#' @param file A character string naming a file for writing.
#' @param uniqueImageKeys A vector specifying the unique image keys of the corpus. 
#' A key grabs an image/video array via `acquireImageFxn(key)`. 
#' @param acquireImageFxn A function whose input is an observation keys and whose output is an array with dimensions `(length(keys), nSpatialDim1, nSpatialDim2, nChannels)` for images and  `(length(keys), nTimeSteps, nSpatialDim1, nSpatialDim2, nChannels)` for image sequence data. 
#' @param conda_env (default = `"CausalImagesEnv"`) A `conda` environment where computational environment lives, usually created via `causalimages::BuildBackend()`
#' @param conda_env_required (default = `T`) A Boolean stating whether use of the specified conda environment is required.
#' @param writeVideo (default = `FALSE`) Should we assume we're writing image sequence data of form batch by time by height by width by channels?
#'
#' @return Writes a unique key-referenced `.tfrecord` from an image/video corpus for use in image-based causal inference training.
#'
#' @examples
#' # Example usage (not run):
#' #WriteTfRecord(
#' #  file = "./NigeriaConfoundApp.tfrecord",
#' #  uniqueImageKeys = 1:n,
#' #  acquireImageFxn = acquireImageFxn)
#'
#' @export
#' @md
WriteTfRecord <- function(file,
                          uniqueImageKeys,
                          acquireImageFxn,
                          writeVideo = F,
                          image_dtype = "float16",
                          conda_env = "CausalImagesEnv",
                          conda_env_required = T,
                          Sys.setenv_text = NULL){
  if(!"jax" %in% ls(envir = cienv)) {
      initialize_jax(conda_env = conda_env, 
                     conda_env_required = conda_env_required,
                     Sys.setenv_text = Sys.setenv_text) 
  }

  if(length(uniqueImageKeys) != length(unique(uniqueImageKeys))){
    stop("Stopping because length(uniqueImageKeys) != length(unique(uniqueImageKeys)) \n
         Remember: Input to WriteTFRecord is uniqueImageKeys, not imageKeysOfUnits where redundancies may live")
  }

  # helper fxns
  message2("Initializing tfrecord helpers...")
  {
    # see https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
    my_bytes_feature <- function(value){
      #"""Returns a bytes_list from a string / byte."""
      #if(class(value) == class(cienv$tf$constant(0))){ # if value ist tensor
      value = value$numpy() # get value of tensor
      #}
      return( cienv$tf$train$Feature(bytes_list=cienv$tf$train$BytesList(value=list(value))))
    }

    my_simple_bytes_feature <- function(value){
      return( cienv$tf$train$Feature(bytes_list = cienv$tf$train$BytesList(value = list(value$numpy()))) )
    }

    my_int_feature <- function(value){
      #"""Returns an int64_list from a bool / enum / int / uint."""
      return( cienv$tf$train$Feature(int64_list=cienv$tf$train$Int64List(value=list(value))) )
    }

    my_serialize_array <- function(array){return( cienv$tf$io$serialize_tensor(array) )}

    parse_single_image <- function(image, index, key){
       if(writeVideo == F){
          data <- dict(
                      "height"    = my_int_feature( image$shape[[1]] ), # note: zero indexed 
                      "width"     = my_int_feature( image$shape[[2]] ),
                      "channels"  = my_int_feature( image$shape[[3]] ),
                      "raw_image" = my_bytes_feature( my_serialize_array( image ) ),
                      "index"     = my_int_feature( index ),
                      "key"       = my_bytes_feature( my_serialize_array(key) ))
       }
      if(writeVideo == T){
        data <- dict(
          "time"      = my_int_feature( image$shape[[1]] ), # note: zero indexed 
          "height"    = my_int_feature( image$shape[[2]] ),
          "width"     = my_int_feature( image$shape[[3]] ),
          "channels"  = my_int_feature( image$shape[[4]] ),
          "raw_image" = my_bytes_feature( my_serialize_array( image ) ),
          "index"     = my_int_feature( index ),
          "key"       = my_bytes_feature( my_serialize_array(key) ) )
      }
        out <- cienv$tf$train$Example(  features = cienv$tf$train$Features(feature = data)  )
        return( out )
  }
  }

  # for clarity, set file to tf_record_name
  message2("Starting save run...")
  tf_record_name <- file
  if( !grepl(tf_record_name, pattern = "/") ){
    tf_record_name <- paste("./",tf_record_name, sep = "")
  }

  orig_wd <- getwd()
  tf_record_name <- strsplit(tf_record_name,split="/")[[1]]
  new_wd <- paste(tf_record_name[- length(tf_record_name) ],collapse = "/")
  setwd( new_wd )
  tf_record_writer = cienv$tf$io$TFRecordWriter( tf_record_name[  length(tf_record_name)  ] ) #create a writer that'll store our data to disk
  setwd(  orig_wd )
  for(irz in 1:length(uniqueImageKeys)){
    if(irz %% 10 == 0 | irz == 1){ print( sprintf("[%s] At index %s of %s [%.3f%%]",
                                                  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                                                  irz, length(uniqueImageKeys), 
                                                  100*irz/length(uniqueImageKeys)) ) }
    tf_record_write_output <- parse_single_image(image = r2const(acquireImageFxn( uniqueImageKeys[irz]  ),
                                                          eval(parse(text = sprintf("cienv$tf$%s",image_dtype)))),
                                                 index = irz,
                                                 key = as.character(uniqueImageKeys[irz] ) )
    tf_record_writer$write( tf_record_write_output$SerializeToString()  )
  }
  print("Finalizing tfrecords....")
  tf_record_writer$close()
  print("Done writing tfrecord!")
}

#!/usr/bin/env Rscript
#' Reads unique key indices from a `.tfrecord` file.
#'
#' Reads unique key indices from a `.tfrecord` file saved via a call to `causalimages::WriteTfRecord`.
#'
#' @param uniqueKeyIndices (integer vector) Unique image indices to be retrieved from a `.tfrecord`
#' @param filename (character string) A character string stating the path to a `.tfrecord`
#' @param nObs (integer) Total number of observations in the tfrecord file
#' @param readVideo (default = `FALSE`) A Boolean indicating whether to read video/sequence data
#' @param conda_env (Default = `NULL`) A `conda` environment where tensorflow v2 lives. Used only if a version of tensorflow is not already active.
#' @param conda_env_required (default = `FALSE`) A Boolean stating whether use of the specified conda environment is required.
#' @param image_dtype (default = `"float16"`) A string specifying the image data type
#' @param iterator (default = `NULL`) An optional iterator object for sequential reading
#' @param return_iterator (default = `FALSE`) A Boolean indicating whether to return the iterator for subsequent calls
#'
#' @return Returns content from a `.tfrecord` associated with `uniqueKeyIndices`
#'
#' @examples
#' # Example usage (not run):
#' #GetElementFromTfRecordAtIndices(
#'   #uniqueKeyIndices = 1:10,
#'   #filename = "./NigeriaConfoundApp.tfrecord",
#'   #nObs = 100)
#'
#' @export
#' @md
GetElementFromTfRecordAtIndices <- function(uniqueKeyIndices, filename, nObs, readVideo = F,
                                            conda_env = NULL, conda_env_required = F, image_dtype = "float16",
                                            iterator = NULL, return_iterator = F){
  # consider passing iterator as input to function to speed up large-batch execution
  image_dtype_ <- try(eval(parse(text = sprintf("cienv$tf$%s",image_dtype))), T)
  if("try-error" %in% class(image_dtype_)){ 
    image_dtype_ <- try(eval(parse(text = sprintf("cienv$tf$%s",image_dtype$name))), T) 
  }
  image_dtype <- image_dtype_

  if(is.null(iterator)){
    orig_wd <- getwd()
    tf_record_name <- filename
    if( !grepl(tf_record_name, pattern = "/") ){
      tf_record_name <- paste("./",tf_record_name, sep = "")
    }
    tf_record_name <- strsplit(tf_record_name,split="/")[[1]]
    new_wd <- paste(tf_record_name[-length(tf_record_name)],collapse = "/")
    setwd( new_wd )

    # Load the TFRecord file
    dataset = cienv$tf$data$TFRecordDataset( tf_record_name[length(tf_record_name)]  )

    # Parse the tf.Example messages
    dataset <- dataset$map( function(x){ parse_tfr_element(x, 
                                                           readVideo = readVideo, 
                                                           image_dtype = image_dtype) }) # return

    index_counter <- last_in_ <- 0L
    return_list <- replicate(length( dataset$element_spec), {list(replicate(length(uniqueKeyIndices), list()))})
  }

  if(!is.null(iterator)){
    dataset_iterator <- iterator[[1]]
    last_in_ <- iterator[[2]] # note: last_in_ is 0 indexed
    index_counter <- 0L
    return_list <- replicate(length( dataset_iterator$element_spec),
                             {list(replicate(length(uniqueKeyIndices), list()))})
  }

  # uniqueKeyIndices made 0 indexed
  uniqueKeyIndices <- as.integer( uniqueKeyIndices - 1L )

  for(in_ in (indices_sorted <- sort(uniqueKeyIndices))){
    index_counter <- index_counter + 1

    # Skip the first `uniqueKeyIndices` elements, shifted by current loc thru data set
    if( index_counter == 1 & is.null(iterator) ){
      dataset <- dataset$skip(  as.integer(in_)  )#$prefetch(buffer_size = 5L)
      dataset_iterator <- reticulate::as_iterator( dataset$take( as.integer(nObs - as.integer(in_)  ) ))
      element <- dataset_iterator$`next`()
    }

    # Take the next element, then
    # Get the only element in the dataset (as a tuple of features)
    if(index_counter > 1 | !is.null(iterator)){
      needThisManyUnsavedIters <- (in_ - last_in_ - 1L)
      if(length(needThisManyUnsavedIters) > 0){ if(needThisManyUnsavedIters > 0){
        for(fari in 1:needThisManyUnsavedIters){ dataset_iterator$`next`() }
      } }
      element <- dataset_iterator$`next`()
    }
    last_in_ <- in_

    # form final output
    if(length(uniqueKeyIndices) == 1){ return_list <- element }
    if(length(uniqueKeyIndices) > 1){
      for(li_ in 1:length(element)){
        return_list[[li_]][[index_counter]] <- cienv$tf$expand_dims(element[[li_]],0L)
      }
    }
    if(index_counter %% 5==0){ try(cienv$py_gc$collect(),T) }
  }

  if(index_counter > 1){ for(li_ in 1:length(element)){
    return_list[[li_]] <- eval(parse(text =
                      paste("cienv$tf$concat( list(", paste(paste("return_list[[li_]][[", 1:length(uniqueKeyIndices), "]]"),
                                                            collapse = ","), "), 0L)", collapse = "") ))
    if(  any(diff(uniqueKeyIndices)<0)  ){ # re-order if needed
      return_list[[li_]] <- cienv$tf$gather(return_list[[li_]],
                                      indices = as.integer(match(uniqueKeyIndices,indices_sorted)-1L),
                                      axis = 0L)
    }
  }}

  if(is.null(iterator)){ setwd(  orig_wd  ) }

  if(return_iterator == T){
    return_list <- list(return_list, list(dataset_iterator, last_in_))
  }

  return(  return_list  )
}

parse_tfr_element <- function(element, readVideo, image_dtype){
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  image_dtype_ <- try(eval(parse(text = sprintf("cienv$tf$%s",image_dtype))), T)
  if("try-error" %in% class(image_dtype_)){ 
    image_dtype_ <- try(eval(parse(text = sprintf("cienv$tf$%s",image_dtype$name))), T) 
  }
  image_dtype <- image_dtype_

  dict_init_val <- list()
  if(!readVideo){
    im_feature_description <- dict(
      'height' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'width' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'channels' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'raw_image' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$string),
      'index' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'key' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$string)
    )
  }

  if(readVideo){
    im_feature_description <- dict(
      'time' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'height' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'width' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'channels' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'raw_image' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$string),
      'index' = cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$int64),
      'key'= cienv$tf$io$FixedLenFeature(dict_init_val, cienv$tf$string)
    )
  }

  # parse tf record
  content <- cienv$tf$io$parse_single_example(element, 
                                              im_feature_description)

  # get 'feature' (e.g., image/image sequence)
  feature <- cienv$tf$io$parse_tensor( content[['raw_image']],
                                       out_type = image_dtype )

  # get the key
  key <- cienv$tf$io$parse_tensor( content[['key']],
                             out_type = cienv$tf$string )

  #  and reshape it appropriately
  if(!readVideo){
    feature = cienv$tf$reshape(  feature, shape = c(content[['height']],
                                                    content[['width']],
                                                    content[['channels']])  )
  }
  if(readVideo){
    feature = cienv$tf$reshape(  feature, shape = c(content[['time']],
                                                    content[['height']],
                                                    content[['width']],
                                                    content[['channels']])  )
  }

  return(    list(feature, content[['index']], key)    )
}
