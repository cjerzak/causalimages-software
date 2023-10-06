#!/usr/bin/env Rscript
#' Write an image corpus as a .tfrecord file
#'
#' Writes an image corpus to a `.tfrecord` file for rapid reading of images into memory for fast ML training.
#'
#' @usage
#'
#' WriteTfRecord(file,imageKeysOfUnits,acquireImageFxn,conda_env)
#'
#' @param file A character string naming a file for writing.
#' @param imageKeysOfUnits A vector specifying the image keys of the corpus. A key grabs an image via acquireImageFxn(key)
#' @param acquireImageFxn A function whose input is an observation index and whose output is an image.
#' @param writeVideo (default = `FALSE`) Should we assume we're writing image sequence data of form batch by time by height by width by channels?
#' @param conda_env A `conda` environment where tensorflow v2 lives. Used only if a version of tensorflow is not already active.
#' @param conda_env_required (default = `F`) A Boolean stating whether use of the specified conda environment is required.
#'
#' @return Writes an key- and index-referenced `.tfrecord` from an image corpus for use in image-based causal inference training.
#'
#' @examples
#' # Example usage (not run):
#' #WriteTfRecord(
#' #  file = "./NigeriaConfoundApp.tfrecord",
#' #  keys = 1:n,
#' #  acquireImageFxn = acquireImageFxn,
#' #  conda_env = "tensorflow_m1")
#'
#' @export
#' @md
WriteTfRecord <- function(file,
                          imageKeysOfUnits,
                          acquireImageFxn,
                          writeVideo = F,
                          conda_env = NULL,
                          attemptRestart = F,
                          conda_env_required = F){
  #if(! (try(as.numeric(tf$sqrt(1.)),T) == 1)){ # note: calling tf$sqrt here induces downstream errors (i think a wrong version of python gets initialized)
  {
    print("Loading Python environment (requires tensorflow)")
    # conda_env <- 'tensorflow_m1'; conda_env_required <- T
    library(tensorflow) ; # library(keras)
    if(!is.null(conda_env)){
      try(tensorflow::use_condaenv(conda_env, required = conda_env_required),T)
    }
     try(tf$square(1.),T);

    # import python garbage collectors
    py_gc <- reticulate::import("gc")
    gc(); py_gc$collect()
  }

  # helper fxns
  {
    # see https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c
    my_bytes_feature <- function(value){
      #"""Returns a bytes_list from a string / byte."""
      #if(class(value) == class(tf$constant(0))){ # if value ist tensor
      value = value$numpy() # get value of tensor
      #}
      return( tf$train$Feature(bytes_list=tf$train$BytesList(value=list(value))))
    }

    my_simple_bytes_feature <- function(value){
      return( tf$train$Feature(bytes_list = tf$train$BytesList(value = list(value$numpy()))) )
    }

    my_float_feature <- function(value){
      #"""Returns a floast_list from a float / double."""
      return( tf$train$Feature(float_list=tf$train$FloatList(value=list(value)) ))
    }

    my_int_feature <- function(value){
      #"""Returns an int64_list from a bool / enum / int / uint."""
      return( tf$train$Feature(int64_list=tf$train$Int64List(value=list(value))) )
    }

    my_serialize_array <- function(array){return( tf$io$serialize_tensor(array) )}

    parse_single_image <- function(image, index, key){
       if(writeVideo == F){
          data = dict(
                      "height" = my_int_feature(image$shape[[2]]),
                      "width" = my_int_feature(image$shape[[3]]),
                      "depth" = my_int_feature(image$shape[[4]]),
                      "raw_image" = my_bytes_feature( my_serialize_array( image ) ),
                      "index" = my_int_feature(index),
                      "key" = my_int_feature(key))
       }
      if(writeVideo == T){
        data = dict(
          "time" = my_int_feature(image$shape[[2]]),
          "height" = my_int_feature(image$shape[[3]]),
          "width" = my_int_feature(image$shape[[4]]),
          "depth" = my_int_feature(image$shape[[5]]),
          "raw_image" = my_bytes_feature( my_serialize_array( image ) ),
          "index" = my_int_feature(index),
          "key" = my_int_feature(key))
      }
        out = tf$train$Example(  features = tf$train$Features(feature = data)  )
        return( out )
  }
  }

  # for clarity, set file to tf_record_name
  tf_record_name <- file
  if( !grepl(tf_record_name, pattern = "/") ){
    tf_record_name <- paste("./",tf_record_name, sep = "")
  }

  orig_wd <- getwd()
  tf_record_name <- strsplit(tf_record_name,split="/")[[1]]
  new_wd <- paste(tf_record_name[- length(tf_record_name) ],collapse = "/")
  setwd( new_wd )
  tf_record_writer = tf$io$TFRecordWriter( tf_record_name[  length(tf_record_name)  ] ) #create a writer that'll store our data to disk
  setwd(  orig_wd )
  for(irz in 1:length(imageKeysOfUnits)){
    if(irz %% 10 == 0 | irz == 1){ print( sprintf("At index %s", irz ) ) }
    tf_record_write_output <- parse_single_image(image = r2const(acquireImageFxn( imageKeysOfUnits[irz]  ), tf$float32),
                                                 index = as.integer( irz ),
                                                 key = as.integer( imageKeysOfUnits[irz]  ) )
    tf_record_writer$write( tf_record_write_output$SerializeToString()  )
  }
  print("Done! Finalizing tfrecords....")
  tf_record_writer$close()
}

#!/usr/bin/env Rscript
#' Reads indices from a `.tfrecord` file.
#'
#' Reads indices from a `.tfrecord` file saved via a call to `causalimages::WriteTfRecord`. Assumes a tensorflow environment has been activated in R.
#'
#' @usage
#'
#' GetElementFromTfRecordAtIndices(indices, file)
#' @param indices (integer vector) Observation indices to be retrieved from a `.tfrecord`
#' @param file (character string) A character string stating the path to a `.tfrecord`
#' @param conda_env (Default = `NULL`) A `conda` environment where tensorflow v2 lives. Used only if a version of tensorflow is not already active.
#' @param conda_env_required (default = `F`) A Boolean stating whether use of the specified conda environment is required.
#'
#' @return Returns content from a `.tfrecord` associated with `indices`
#'
#' @examples
#' # Example usage (not run):
#' #GetElementFromTfRecordAtIndices(
#'   #indices = 1:10,
#'   #file = "./NigeriaConfoundApp.tfrecord")
#'
#' @export
#' @md
GetElementFromTfRecordAtIndices <- function(indices, filename, nObs, readVideo = F,
                                            conda_env = NULL, conda_env_required = F,
                                            iterator = NULL, return_iterator = F){
  # consider passing iterator as input to function to speed up large-batch execution
  #if(! (try(as.numeric(tf$sqrt(1.)),T) == 1)){
  if(T == F){  # function assumes tensorflow initialized
    print("Loading Python environment (requires tensorflow)")
    library(tensorflow); library(keras)
    if(!is.null(conda_env)){
      try(tensorflow::use_condaenv(conda_env, required = T),T)
    }
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]],T),T)
    tf$config$set_soft_device_placement( T )
    tf$random$set_seed(  c(1000L ) ); tf$keras$utils$set_random_seed( 4L )

    # import python garbage collectors
    py_gc <- reticulate::import("gc")
    gc(); py_gc$collect()
  }

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
    dataset = tf$data$TFRecordDataset( tf_record_name[length(tf_record_name)]  )

    # Parse the tf.Example messages
    #dataset <- dataset$map(   parse_tfr_element   )
    dataset <- dataset$map( function(x){parse_tfr_element(x, readVideo = readVideo)} ) # return

    if(T == F){
      dataset <- dataset$skip(  as.integer(in_)  )#$prefetch(buffer_size = 5L)
      dataset_iterator <- reticulate::as_iterator( dataset$take( as.integer(nObs - as.integer(in_)  ) ))
      element <- reticulate::iter_next( dataset_iterator )
    }

    index_counter <- last_in_ <- 0L
    return_list <- replicate(length( dataset$element_spec),
                             {list(replicate(length(indices), list()))})
  }

  if(!is.null(iterator)){
    dataset_iterator <- iterator[[1]]
    last_in_ <- iterator[[2]] # note: last_in_ is 0 indexed
    index_counter <- 0L
    return_list <- replicate(length( dataset_iterator$element_spec),
                             {list(replicate(length(indices), list()))})
  }

  # indices is 0 indexed
  indices <- as.integer( indices - 1L )

  for(in_ in (indices_sorted <- sort(indices))){
    index_counter <- index_counter + 1

    # Skip the first `indices` elements, shifted by current loc thru data set
    #dataset <- dataset$skip(  as.integer( in_   - last_in_)  )#$prefetch(buffer_size = 5L)
    if(index_counter == 1 & is.null(iterator)){
      dataset <- dataset$skip(  as.integer(in_)  )#$prefetch(buffer_size = 5L)
      dataset_iterator <- reticulate::as_iterator( dataset$take( as.integer(nObs - as.integer(in_)  ) ))
      element <- reticulate::iter_next( dataset_iterator )
    }
    #tmp <- tmp$skip(2L)
    #tmp2 <- reticulate::as_iterator( tmp )
    #system.time( reticulate::iter_next( dataset_iterator  ) )

    # Take the next element, then
    # Get the only element in the dataset (as a tuple of features)
    #element <- reticulate::iter_next( reticulate::as_iterator( dataset$take( 1L  ) ) )
    if(index_counter > 1 | !is.null(iterator)){
      needThisManyUnsavedIters <- (in_ - last_in_ - 1L)
      if(length(needThisManyUnsavedIters) > 0){ if(needThisManyUnsavedIters > 0){
          for(fari in 1:needThisManyUnsavedIters){ reticulate::iter_next( dataset_iterator ) }
      } }
      element <- reticulate::iter_next( dataset_iterator )
    }
    last_in_ <- in_

    # form final output
    if(length(indices) == 1){ return_list <- element }
    if(length(indices) > 1){
      for(li_ in 1:length(element)){
        return_list[[li_]][[index_counter]] <- tf$expand_dims(element[[li_]],0L)
      }
    }
    if(index_counter %% 5==0){ try(py_gc$collect(),T) }
  }

  if(index_counter > 1){ for(li_ in 1:length(element)){
    return_list[[li_]] <- eval(parse(text =
                      paste("tf$concat( list(", paste(paste("return_list[[li_]][[", 1:length(indices), "]]"),collapse = ","), "), 0L)", collapse = "") ))
    if(  any(diff(indices)<0)  ){ # re-order if needed
      #indices_sorted; indices
      #indices_sorted[ match(indices,indices_sorted) ]
      return_list[[li_]] <- tf$gather(return_list[[li_]],
                                      indices = as.integer(match(indices,indices_sorted)-1L),
                                      axis = 0L)
    }
  }}

  #for(li_ in 1:length(element)){ return_list[[li_]] <- tf$concat( list(return_list[[li_]], element[[li_]]), 0L) }
  if(is.null(iterator)){ setwd(  orig_wd  ) }

  if(return_iterator == T){
    return_list <- list(return_list,list(dataset_iterator, last_in_))
  }

  return(  return_list  )
}

# parse tf elements
parse_tfr_element <- function(element, readVideo = F){
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  dict_init_val <- list()
  if(!readVideo){
    im_feature_description <- dict(
      'height'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'width'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'depth'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'raw_image'= tf$io$FixedLenFeature(dict_init_val, tf$string),
      'index'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'key'= tf$io$FixedLenFeature(dict_init_val, tf$int64)
    )
  }

  if(readVideo){
    im_feature_description <- dict(
      'time'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'height'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'width'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'depth'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'raw_image'= tf$io$FixedLenFeature(dict_init_val, tf$string),
      'index'= tf$io$FixedLenFeature(dict_init_val, tf$int64),
      'key'= tf$io$FixedLenFeature(dict_init_val, tf$int64)
    )
  }

  # parse tf record
  content = tf$io$parse_single_example(element, im_feature_description)

  height = content[['height']]
  width = content[['width']]
  depth = content[['depth']]
  key = content[['key']]
  index = content[['index']]
  raw_image = content[['raw_image']]

  #get our 'feature' (our image)...
  feature = tf$io$parse_tensor( raw_image, out_type = tf$float32 )

  #  and reshape it appropriately
  if(!readVideo){
    feature = tf$reshape(  feature, shape = c(height, width, depth)  )
  }
  if(readVideo){
    time = content[['time']]
    feature = tf$reshape(  feature, shape = c(time, height, width, depth)  )
  }

  return(    list(feature, index, key)    )
}