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

  # For multiple keys, ensure batch dimension is first
  # For single key, return (H, W, C) - WriteTfRecord iterates one key at a time
  if(length(keys) == 1){
    m_ <- array(m_, dim = c(35L, 35L, 3L))
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
take_indices <- unlist( tapply(1:length(obsW),obsW,function(zer){ sample(zer, 50) }) )

# write tf record
# Use TEST_DATA_DIR if set by test suite, otherwise default to ~/Downloads
TfRecord_name <- if (exists("TEST_DATA_DIR")) {
  file.path(TEST_DATA_DIR, "CausalImagesTutorialDat.tfrecord")
} else {
  "~/Downloads/CausalImagesTutorialDat.tfrecord"
}
causalimages::WriteTfRecord(  file =  TfRecord_name,
                uniqueImageKeys = unique( KeysOfObservations[ take_indices ] ),
                acquireImageFxn = acquireImageFromMemory  )

# obtain image representation
MyImageEmbeddings <- causalimages::GetImageRepresentations(
  file  = TfRecord_name,
  imageModelClass = "VisionTransformer",
  pretrainedModel = "clip-rsicd",
  #pretrainedModel = "vit-base",
  imageKeysOfUnits = KeysOfObservations[ take_indices ] 
)

# each row in MyImageEmbeddings$ImageRepresentations corresponds to an observation
# each column represents an embedding dimension associated with the imagery for that location
dim(  MyImageEmbeddings$ImageRepresentations )
plot( MyImageEmbeddings$ImageRepresentations  )

# other output quantities include the image model functions and model parameters
names(  MyImageEmbeddings  )[-1]

print("Done with image representations test!")

expect_true(TRUE)
})
