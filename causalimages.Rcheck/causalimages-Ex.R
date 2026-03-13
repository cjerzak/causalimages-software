pkgname <- "causalimages"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('causalimages')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("AnalyzeImageConfounding")
### * AnalyzeImageConfounding

flush(stderr()); flush(stdout())

### Name: AnalyzeImageConfounding
### Title: Perform causal estimation under image confounding
### Aliases: AnalyzeImageConfounding

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




cleanEx()
nameEx("AnalyzeImageHeterogeneity")
### * AnalyzeImageHeterogeneity

flush(stderr()); flush(stdout())

### Name: AnalyzeImageHeterogeneity
### Title: Decompose treatment effect heterogeneity by image or image
###   sequence
### Aliases: AnalyzeImageHeterogeneity

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




cleanEx()
nameEx("BuildBackend")
### * BuildBackend

flush(stderr()); flush(stdout())

### Name: BuildBackend
### Title: Build the environment for CausalImages models.
### Aliases: BuildBackend

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/



cleanEx()
nameEx("CausalImagesTutorialData")
### * CausalImagesTutorialData

flush(stderr()); flush(stdout())

### Name: CausalImagesTutorialData
### Title: CausalImages Tutorial Data
### Aliases: CausalImagesTutorialData FullImageArray KeysOfImages
###   KeysOfObservations LongLat obsW obsY X
### Keywords: datasets

### ** Examples

# Load the tutorial data
data(CausalImagesTutorialData)

# View dimensions of the image array
dim(FullImageArray)

# Check the number of observations
length(obsY)

# View treatment distribution
table(obsW)




cleanEx()
nameEx("CheckDataQuality")
### * CheckDataQuality

flush(stderr()); flush(stdout())

### Name: CheckDataQuality
### Title: Check data quality for causalimages analyses
### Aliases: CheckDataQuality

### ** Examples

## Not run: 
##D # Check data before writing TfRecord
##D result <- CheckDataQuality(
##D   uniqueImageKeys = unique(imageKeys),
##D   acquireImageFxn = myImageLoader,
##D   file = "data.tfrecord",
##D   checkContext = "pre_write"
##D )
##D 
##D # Check data before analysis
##D result <- CheckDataQuality(
##D   obsW = treatment,
##D   obsY = outcome,
##D   imageKeysOfUnits = imageKeys,
##D   file = "data.tfrecord",
##D   checkContext = "pre_analysis"
##D )
## End(Not run)




cleanEx()
nameEx("GetAndSaveGeolocatedImages")
### * GetAndSaveGeolocatedImages

flush(stderr()); flush(stdout())

### Name: GetAndSaveGeolocatedImages
### Title: Getting and saving geo-located images from a pool of .tif's
### Aliases: GetAndSaveGeolocatedImages

### ** Examples


# Example use (not run):
#MASTER_IMAGE_POOL_FULL_DIR <- c("./LargeTifs/tif1.tif","./LargeTifs/tif2.tif")
#GetAndSaveGeolocatedImages(
                       #long = GeoKeyMat$geo_long,
                       #lat = GeoKeyMat$geo_lat,
                       #image_pixel_width = 500L,
                       #keys = row.names(GeoKeyMat),
                       #tif_pool = MASTER_IMAGE_POOL_FULL_DIR,
                       #save_folder = "./Data/Uganda2000_processed",
                       #save_as = "csv",
                       #lyrs = NULL)




cleanEx()
nameEx("GetElementFromTfRecordAtIndices")
### * GetElementFromTfRecordAtIndices

flush(stderr()); flush(stdout())

### Name: GetElementFromTfRecordAtIndices
### Title: Reads unique key indices from a '.tfrecord' file.
### Aliases: GetElementFromTfRecordAtIndices

### ** Examples

# Example usage (not run):
#GetElementFromTfRecordAtIndices(
  #uniqueKeyIndices = 1:10,
  #filename = "./NigeriaConfoundApp.tfrecord",
  #nObs = 100)




cleanEx()
nameEx("GetImageRepresentations")
### * GetImageRepresentations

flush(stderr()); flush(stdout())

### Name: GetImageRepresentations
### Title: Generates image and video representations useful in earth
###   observation tasks for causal inference.
### Aliases: GetImageRepresentations

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




cleanEx()
nameEx("GetMoments")
### * GetMoments

flush(stderr()); flush(stdout())

### Name: GetMoments
### Title: Get moments for normalization (internal function)
### Aliases: GetMoments

### ** Examples

# (Not run)
# GetMoments(iterator, dataType, image_dtype, momentCalIters = 34L)



cleanEx()
nameEx("LongLat2CRS")
### * LongLat2CRS

flush(stderr()); flush(stdout())

### Name: LongLat2CRS
### Title: Get the spatial point of long/lat coordinates
### Aliases: LongLat2CRS

### ** Examples

# (Not run)
#spatialPt <- LongLat2CRS(long = 49.932,
#                 lat = 35.432,
#                 CRS_ref = sf::st_crs("+proj=lcc +lat_1=48 +lat_2=33 +lon_0=-100 +ellps=WGS84"))



cleanEx()
nameEx("PredictiveRun")
### * PredictiveRun

flush(stderr()); flush(stdout())

### Name: PredictiveRun
### Title: Perform predictive modeling using images or videos
### Aliases: PredictiveRun

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




cleanEx()
nameEx("PredictiveScore")
### * PredictiveScore

flush(stderr()); flush(stdout())

### Name: PredictiveScore
### Title: Score a saved predictive model on a tfrecord
### Aliases: PredictiveScore

### ** Examples

## Not run: 
##D preds <- PredictiveScore(
##D   modelPath = "./trained_model.eqx",
##D   file = "./new_data.tfrecord",
##D   imageKeysOfUnits = new_keys
##D )
## End(Not run)




cleanEx()
nameEx("WriteTfRecord")
### * WriteTfRecord

flush(stderr()); flush(stdout())

### Name: WriteTfRecord
### Title: Write an image corpus as a .tfrecord file
### Aliases: WriteTfRecord

### ** Examples

# Example usage (not run):
#WriteTfRecord(
#  file = "./NigeriaConfoundApp.tfrecord",
#  uniqueImageKeys = 1:n,
#  acquireImageFxn = acquireImageFxn)




cleanEx()
nameEx("image2")
### * image2

flush(stderr()); flush(stdout())

### Name: image2
### Title: Visualizing matrices as heatmaps with correct
###   north-south-east-west orientation
### Aliases: image2

### ** Examples

#set seed
set.seed(1)

#Generate data
x <- matrix(rnorm(50*50), ncol = 50)
diag(x) <- 3

# create plot
image2(x, main = "Example Text", cex.main = 2)




cleanEx()
nameEx("message2")
### * message2

flush(stderr()); flush(stdout())

### Name: message2
### Title: message2 message() with timestamps
### Aliases: message2

### ** Examples

message2("Hello world")
message2("Process completed", quiet = FALSE)



cleanEx()
nameEx("print2")
### * print2

flush(stderr()); flush(stdout())

### Name: print2
### Title: print2 print() with timestamps
### Aliases: print2

### ** Examples

print2("Hello world")



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
