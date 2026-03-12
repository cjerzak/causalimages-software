test_that("GetImageRepresentations works", {
  skip_on_cran()

################################
# Image and image-sequence embeddings tutorial using causalimages
################################

# remote install latest version of the package if needed
# devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

# local install for development team
# install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

# build backend you haven't ready:
# causalimages::BuildBackend()

# load in package
library( causalimages  ); options(error = NULL)

clear_cached_pretrained_state <- function() {
  cached_names <- c(
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
    "SD_RESCALER",
    "TRANSFORMERS_MODEL_NAME",
    "TransformersModel",
    "TransformersProcessor",
    "nParameters_Pretrained"
  )
  rm(
    list = intersect(cached_names, ls(envir = causalimages:::cienv, all.names = TRUE)),
    envir = causalimages:::cienv
  )
}

# load in tutorial data
data(  CausalImagesTutorialData )

# example acquire image function (loading from memory)
# in general, you'll want to write a function that returns images
# that saved disk associated with keys
acquireImageFromMemory <- function(keys){
  # here, the function input keys
  # refers to the unit-associated image keys
  # use consistent 35x35x3 dimensions
  m_ <- FullImageArray[match(keys, KeysOfImages),1:35,1:35,]

  # Always return (batch, H, W, C) shape as per documentation
  # For single key, keep batch dimension as 1
  if(length(keys) == 1){
    m_ <- array(m_, dim = c(1L, 35L, 35L, 3L))
  }
  return( m_ )
}

# drop first column
X <- X[,-1]

# mean imputation for simplicity
X <- apply(X,2,function(zer){
  zer[is.na(zer)] <- mean( zer,na.rm = T )
  return( zer )
})

# select observation subset to make tutorial analyses run faster
set.seed(4321L)
take_indices <- unlist( tapply(1:length(obsW),obsW,function(zer){ sample(zer, 3) }) )

# write tf record
# Use TEST_DATA_DIR if set by test suite, otherwise default to ~/Downloads
TfRecord_name <- if (exists("TEST_DATA_DIR")) {
  file.path(TEST_DATA_DIR, "CausalImagesTutorialDat.tfrecord")
} else {
  file.path(tempdir(), "CausalImagesTutorialDat.tfrecord")
}
on.exit(unlink(TfRecord_name), add = TRUE)
causalimages::WriteTfRecord(  file =  TfRecord_name,
                uniqueImageKeys = unique( KeysOfObservations[ take_indices ] ),
                acquireImageFxn = acquireImageFromMemory  )

model_cases <- list(
  list(name = "clip-rsicd", expected_width = 512L),
  list(name = "vit-base", expected_width = 768L),
  list(name = "swin", expected_width = 768L)
)

for(model_case in model_cases){
  clear_cached_pretrained_state()

  # obtain image representation
  MyImageEmbeddings <- NULL
  expect_error(
    MyImageEmbeddings <- causalimages::GetImageRepresentations(
      file  = TfRecord_name,
      imageModelClass = "VisionTransformer",
      pretrainedModel = model_case$name,
      batchSize = 2L,
      imageKeysOfUnits = KeysOfObservations[ take_indices ]
    ),
    NA,
    info = model_case$name
  )

  # each row in MyImageEmbeddings$ImageRepresentations corresponds to an observation
  # each column represents an embedding dimension associated with the imagery for that location
  expect_true(is.matrix(MyImageEmbeddings$ImageRepresentations), info = model_case$name)
  expect_equal(
    dim(MyImageEmbeddings$ImageRepresentations),
    c(length(KeysOfObservations[take_indices]), model_case$expected_width),
    info = model_case$name
  )
  expect_false(anyNA(MyImageEmbeddings$ImageRepresentations), info = model_case$name)
}

clear_cached_pretrained_state()

print("Done with image representations test!")
})
