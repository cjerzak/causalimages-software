#!/usr/bin/env Rscript

test_that("WriteTfRecord works", {
  skip_on_cran()

  ################################
  # Image confounding tutorial using causalimages
  # and tfrecords for faster results
  ################################

  # remote install latest version of the package if needed
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

  # local install for development team
  # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

  # build backend you haven't ready:
  # causalimages::BuildBackend()

  # load in package
  library( causalimages  )

  # load in tutorial data
  data(  CausalImagesTutorialData )

  # example acquire image function (loading from memory)
  # in general, you'll want to write a function that returns images
  # that saved disk associated with keys
  acquireImageFromMemory <- function(keys){
    # here, the function input keys
    # refers to the unit-associated image keys
    m_ <- FullImageArray[match(keys, KeysOfImages),,,]

    # For multiple keys, ensure batch dimension is first
    # For single key, return (H, W, C) - WriteTfRecord iterates one key at a time
    if(length(keys) == 1){
      m_ <- array(m_, dim = c(35L, 35L, 3L))
    }
    return( m_ )
  }

  dim( acquireImageFromMemory(KeysOfImages[1]) )
  dim( acquireImageFromMemory(KeysOfImages[1:2]) )

  # drop first column
  X <- X[,-1]

  # mean imputation for simplicity
  X <- apply(X,2,function(zer){
    zer[is.na(zer)] <- mean( zer,na.rm = T )
    return( zer )
  })

  # select observation subset to make tutorial analyses run faster
  # select 50 treatment and 50 control observations
  set.seed(1.)
  take_indices <- unlist( tapply(1:length(obsW),obsW,function(zer){sample(zer, 50)}) )

  # !!! important note !!!
  # when using tf recordings, it is essential that the data inputs be pre-shuffled like is done here.
  # you can use a seed for reproducing the shuffle (so the tfrecord is correctly indexed and you don't need to re-make it)
  # tf records read data quasi-sequentially, so systematic patterns in the data ordering
  # reduce performance

  # uncomment for a larger n analysis
  #take_indices <- 1:length( obsY )

  # set tfrecord save location (safest using absolute path)
  tfrecord_loc <- "./ExampleRecord.tfrecord"

  # you may use relative paths like this:
  # tfrecord_loc <- "./Downloads/test1/test2/test3/ExampleRecord.tfrecord"

  # or absolute paths like this:
  # tfrecord_loc <- "~/Downloads/test1/test2/test3/ExampleRecord.tfrecord"

  # write a tf records repository
  causalimages::WriteTfRecord(  file = tfrecord_loc,
                  uniqueImageKeys = unique( KeysOfObservations[ take_indices ] ),
                  acquireImageFxn = acquireImageFromMemory )

  expect_true(file.exists(tfrecord_loc))
})

