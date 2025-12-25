pkgname <- "causalimages"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
base::assign(".ExTimings", "causalimages-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('causalimages')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("AnalyzeImageConfounding")
### * AnalyzeImageConfounding

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: AnalyzeImageConfounding
### Title: Perform causal estimation under image confounding
### Aliases: AnalyzeImageConfounding

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("AnalyzeImageConfounding", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("AnalyzeImageHeterogeneity")
### * AnalyzeImageHeterogeneity

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: AnalyzeImageHeterogeneity
### Title: Decompose treatment effect heterogeneity by image or image
###   sequence
### Aliases: AnalyzeImageHeterogeneity

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("AnalyzeImageHeterogeneity", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("BuildBackend")
### * BuildBackend

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: BuildBackend
### Title: Build the environment for CausalImages models. Builds a conda
###   environment in which jax, tensorflow, tensorflow-probability optax,
###   equinox, and jmp are installed.
### Aliases: BuildBackend

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("BuildBackend", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("CausalImagesTutorialData")
### * CausalImagesTutorialData

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
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




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("CausalImagesTutorialData", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("CheckDataQuality")
### * CheckDataQuality

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
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




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("CheckDataQuality", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("GetAndSaveGeolocatedImages")
### * GetAndSaveGeolocatedImages

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
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




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("GetAndSaveGeolocatedImages", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("GetElementFromTfRecordAtIndices")
### * GetElementFromTfRecordAtIndices

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: GetElementFromTfRecordAtIndices
### Title: Reads unique key indices from a '.tfrecord' file.
### Aliases: GetElementFromTfRecordAtIndices

### ** Examples

# Example usage (not run):
#GetElementFromTfRecordAtIndices(
  #uniqueKeyIndices = 1:10,
  #filename = "./NigeriaConfoundApp.tfrecord",
  #nObs = 100)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("GetElementFromTfRecordAtIndices", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("GetImageRepresentations")
### * GetImageRepresentations

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: GetImageRepresentations
### Title: Generates image and video representations useful in earth
###   observation tasks for causal inference.
### Aliases: GetImageRepresentations

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("GetImageRepresentations", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("GetMoments")
### * GetMoments

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: GetMoments
### Title: Get moments for normalization (internal function)
### Aliases: GetMoments

### ** Examples

# (Not run)
# GetMoments(iterator, dataType, image_dtype, momentCalIters = 34L)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("GetMoments", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("LongLat2CRS")
### * LongLat2CRS

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: LongLat2CRS
### Title: Get the spatial point of long/lat coordinates
### Aliases: LongLat2CRS

### ** Examples

# (Not run)
#spatialPt <- LongLat2CRS(long = 49.932,
#                 lat = 35.432,
#                 CRS_ref = sf::st_crs("+proj=lcc +lat_1=48 +lat_2=33 +lon_0=-100 +ellps=WGS84"))



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("LongLat2CRS", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("PredictiveRun")
### * PredictiveRun

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: PredictiveRun
### Title: Perform predictive modeling using images or videos
### Aliases: PredictiveRun

### ** Examples

# For a tutorial, see
# github.com/cjerzak/causalimages-software/




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("PredictiveRun", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("WriteTfRecord")
### * WriteTfRecord

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: WriteTfRecord
### Title: Write an image corpus as a .tfrecord file
### Aliases: WriteTfRecord

### ** Examples

# Example usage (not run):
#WriteTfRecord(
#  file = "./NigeriaConfoundApp.tfrecord",
#  uniqueImageKeys = 1:n,
#  acquireImageFxn = acquireImageFxn)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("WriteTfRecord", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("image2")
### * image2

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
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




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("image2", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("message2")
### * message2

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: message2
### Title: message2 message() with timestamps
### Aliases: message2

### ** Examples

message2("Hello world")
message2("Process completed", quiet = FALSE)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("message2", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("print2")
### * print2

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: print2
### Title: print2 print() with timestamps
### Aliases: print2

### ** Examples

print2("Hello world")



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("print2", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
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
