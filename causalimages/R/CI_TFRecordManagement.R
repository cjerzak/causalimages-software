#' Defines an internal TFRecord management routine (internal function)
#'
#' Defines management defined in TFRecordManagement(). Internal function. 
#'
#' @param . No parameters. 
#' 
#' @return Internal function defining a tfrecord management sequence. 
#'
#' @import reticulate rrapply
#' @export
#' @md
TFRecordManagement <- function(){
  
  if(is.null(file)){stop("No file specified for tfrecord!")}
  changed_wd <- F; if(  !is.null(  file  )  ){
    message2("Establishing connection with tfrecord")
    tf_record_name <- file
    if( !grepl(tf_record_name, pattern = "/") ){
      tf_record_name <- paste("./",tf_record_name, sep = "")
    }
    tf_record_name <- strsplit(tf_record_name,split="/")[[1]]
    new_wd <- paste(tf_record_name[-length(tf_record_name)], collapse = "/")
    message2(sprintf("Temporarily re-setting the wd to %s", new_wd ) )
    changed_wd <- T; setwd( new_wd )
    
    # define video indicator 
    useVideoIndicator <- dataType == "video"
    
    # define tf record 
    tf_dataset <- cienv$tf$data$TFRecordDataset(  tf_record_name[length(tf_record_name)] )
    
    # helper functions
    getParsed_tf_dataset_inference <- function(tf_dataset){
      dataset <- tf_dataset$map( function(x){parse_tfr_element(x, 
                                                               readVideo = useVideoIndicator, 
                                                               image_dtype = image_dtype_tf)} )
      return( dataset <- dataset$batch( ai(max(2L,round(batchSize/2L)  ))) )
    }
    
    message2("Setting up iterators...") # skip the first test_size observations 
    getParsed_tf_dataset_train_Select <- function( tf_dataset ){
      return( tf_dataset$map( function(x){ parse_tfr_element(x, 
                                                             readVideo = useVideoIndicator, 
                                                             image_dtype = image_dtype_tf)},
                              num_parallel_calls = cienv$tf$data$AUTOTUNE) ) 
    }
    getParsed_tf_dataset_train_BatchAndShuffle <- function( tf_dataset ){
      tf_dataset <- tf_dataset$shuffle(buffer_size = cienv$tf$constant(ai(TfRecords_BufferScaler*batchSize),
                                                                       dtype=cienv$tf$int64),
                                       reshuffle_each_iteration = T) 
      tf_dataset <- tf_dataset$batch(  ai(batchSize)   )
      tf_dataset <- tf_dataset$prefetch( cienv$tf$data$AUTOTUNE ) 
      return( tf_dataset )
    }
    if(!is.null(TFRecordControl)){
      tf_dataset_train_control <- getParsed_tf_dataset_train_Select(
        tf_dataset$skip(  test_size <-  ai( TFRecordControl$nTest )  )
      )$take( ai(TFRecordControl$nControl) )$`repeat`(-1L)  
      
      tf_dataset_train_treated <- getParsed_tf_dataset_train_Select(
        tf_dataset$skip( test_size <-  ai( TFRecordControl$nTest ) )
      )$skip( ai(TFRecordControl$nControl)+1L)$`repeat`(-1L) 
      
      tf_dataset_train_treated <- getParsed_tf_dataset_train_BatchAndShuffle( tf_dataset_train_treated )
      tf_dataset_train_control <- getParsed_tf_dataset_train_BatchAndShuffle( tf_dataset_train_control )
      
      ds_iterator_train_treated <- reticulate::as_iterator( tf_dataset_train_treated )
      ds_iterator_train_control <- reticulate::as_iterator( tf_dataset_train_control )
      ds_iterator_train <- reticulate::as_iterator( tf_dataset_train_control )
    }
    if(is.null(TFRecordControl)){
      getParsed_tf_dataset_train <- function( tf_dataset ){
        dataset <- tf_dataset$map( function(x){ parse_tfr_element(x, readVideo = useVideoIndicator, image_dtype = image_dtype_tf)},
                                   num_parallel_calls = cienv$tf$data$AUTOTUNE)
        dataset <- dataset$shuffle(buffer_size = cienv$tf$constant(ai(TfRecords_BufferScaler*batchSize), dtype=cienv$tf$int64),
                                   reshuffle_each_iteration = FALSE) # set FALSE so same train/test split each re-initialization
        dataset <- dataset$batch(  ai(batchSize)   )
        dataset <- dataset$prefetch( cienv$tf$data$AUTOTUNE ) 
        return( dataset  )
      }
      
      # shuffle (generating different train/test splits)
      tf_dataset <- cienv$tf$data$Dataset$shuffle(  tf_dataset, 
                                                    buffer_size = cienv$tf$constant(ai(10L*TfRecords_BufferScaler*batchSize),
                                                                                    dtype=cienv$tf$int64), reshuffle_each_iteration = F )
      tf_dataset_train <- getParsed_tf_dataset_train( 
        tf_dataset$skip(test_size <-  as.integer(round(testFrac * length(unique(imageKeysOfUnits)) )) ) )$`repeat`(  -1L )
      ds_iterator_train <- reticulate::as_iterator( tf_dataset_train )
    }
    
    # define inference iterator 
    tf_dataset_inference <- getParsed_tf_dataset_inference( tf_dataset )
    ds_iterator_inference <- reticulate::as_iterator( tf_dataset_inference )
    
    # Other helper functions
    getParsed_tf_dataset_train_Shuffle <- function( tf_dataset ){
      tf_dataset <- tf_dataset$shuffle(buffer_size = cienv$tf$constant(ai(TfRecords_BufferScaler*batchSize),
                                                                       dtype=cienv$tf$int64),
                                       reshuffle_each_iteration = FALSE )
      return(tf_dataset)
    }
  }
}