#!/usr/bin/env Rscript
#' Write an image corpus as a .tfrecord file
#'
#' Writes an image corpus to a `.tfrecord` file for rapid reading of images into memory for fast ML training.
#'
#' @usage
#'
#' WriteTfRecord(file,acquireImageRepFxn,conda_env)
#'
#' @param file A character string naming a file for writing.
#' @param acquireImageRepFxn A function whose input is an observation index and whose output is an image.
#' @param imageKeys A vector specifying the image keys of the corpus. A key grabs an image via acquireImageRepFxn(key)
#' @param conda_env A `conda` environment where tensorflow v2 lives.
#'
#' @return Writes an index-referenced `.tfrecord` from an image corpus for use in image-based causal inference training.
#'
#' @examples
#' # Example usage:
#' #WriteTfRecord(
#' #  file = "./NigeriaConfoundApp.tfrecord",
#' #  acquireImageRepFxn = acquireImageRepFxn,
#' #  conda_env = "tensorflow_m1")
#'
#' @export
#' @md
WriteTfRecord <- function(file,
                          imageKeys,
                          acquireImageRepFxn,
                          conda_env = NULL){
  if(! (try(as.numeric(tf$sqrt(1.)),T) == 1)){
    #conda_env <- "tensorflow_m1"
    library(tensorflow); library(keras)
    try(tensorflow::use_condaenv(conda_env, required = T),T)
    Sys.sleep(1.); try(tf$square(1.),T); Sys.sleep(1.)
    try(tf$config$experimental$set_memory_growth(tf$config$list_physical_devices('GPU')[[1]],T),T)
    tf$config$set_soft_device_placement( T )
    tfp <- tf_probability()
    tfd <- tfp$distributions
    #tfa <- reticulate::import("tensorflow_addons")

    tf$random$set_seed(  c(1000L ) )
    tf$keras$utils$set_random_seed( 4L )

    py_gc <- reticulate::import("gc")
    gc(); py_gc$collect()
  }

  #
  tf_record_name <- file

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
  }
  parse_single_image <- function(image, index){
    data = dict("height" = my_int_feature(image$shape[[1]]),
                "width" = my_int_feature(image$shape[[2]]),
                "depth" = my_int_feature(image$shape[[3]]),
                "raw_image" = my_bytes_feature( my_serialize_array( image ) ),
                "index" = my_int_feature(index))
    out = tf$train$Example(features=tf$train$Features(feature = data))
  }

  #image <- ImageCorpus[1,,,]
  tf_record_writer = tf$io$TFRecordWriter( tf_record_name ) #create a writer that'll store our data to disk
  for(irz in imageKeys){
    #for(irz in 1:20){
    if(irz %% 10 == 0 | irz == 1){print( irz )}
    tf_record_write_output <- parse_single_image(image = acquireImageRepFxn(irz),
                                                 index = as.integer(irz))
    tf_record_writer$write(tf_record_write_output$SerializeToString())
  }
  tf_record_writer$close()
}
