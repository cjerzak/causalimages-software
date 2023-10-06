#' Getting and saving geo-located images from a pool of .tif's
#'
#' A function that finds the image slice associated with the `long` and `lat` values, saves images by band (if `save_as = "csv"`) in save_folder.
#'
#' @usage
#'
#' GetAndSaveGeolocatedImages(long, lat, keys, tif_pool, save_folder)
#'
#' @param long Vector of numeric longitudes.
#' @param lat Vector of numeric latitudes.
#' @param keys The image keys associated with the long/lat coordinates.
#' @param tif_pool A character vector specifying the fully qualified path to a corpus of .tif files.
#' @param image_pixel_width An even integer specifying the pixel width (and height) of the saved images.
#' @param save_folder (default = `"."`) What folder should be used to save the output? Example: `"~/Downloads"`
#' @param save_as (default = `".csv"`) What format should the output be saved as? Only one option currently (`.csv`)
#' @param lyrs (default = NULL) Integer (vector) specifying the layers to be extracted. Default is for all layers to be extracted.
#'
#' @return Finds the image slice associated with the `long` and `lat` values, saves images by band (if `save_as = "csv"`) in save_folder.
#' The save format is: `sprintf("%s/Key%s_BAND%s.csv", save_folder, keys[i], band_)`
#'
#' @examples
#'
#' # Example use (not run):
#' #MASTER_IMAGE_POOL_FULL_DIR <- c("./LargeTifs/tif1.tif","./LargeTifs/tif2.tif")
#' #GetAndSaveGeolocatedImages(
#'                        #long = GeoKeyMat$geo_long,
#'                        #lat = GeoKeyMat$geo_lat,
#'                        #image_pixel_width = 500L,
#'                        #keys = row.names(GeoKeyMat),
#'                        #tif_pool = MASTER_IMAGE_POOL_FULL_DIR,
#'                        #save_folder = "./Data/Uganda2000_processed",
#'                        #save_as = "csv",
#'                        #lyrs = NULL)
#'
#' @import raster
#' @export
#' @md
#'
GetAndSaveGeolocatedImages <- function(
    long,
    lat,
    keys,
    tif_pool,
    image_pixel_width = 250L,
    save_folder = ".",
    save_as = "csv",
    lyrs = NULL){


  if(attemptRestart == T){

    browser()
  }

  library(raster)
  RADIUS_CELLS <- (DIAMETER_CELLS <- image_pixel_width) / 2
  bad_indices <- c();observation_indices <- 1:length(long)
  counter_b <- 0 ; for(i in observation_indices){
    counter_b <- counter_b + 1
    if(counter_b %% 10 == 0){print(sprintf("Iter %s of %s",counter_b,length(observation_indices)))}
    SpatialTarget_longlat <- c(long[i],lat[i])
    # SpatialTarget_longlat <- c(32.821752, 1.827300)
    # rev(SpatialTarget_longlat)

    found_<-F;counter_ <- 0; while(found_ == F){
      counter_ <- counter_ + 1
      if(is.na(tif_pool[counter_])){ found_ <- bad_ <- T }
      if(!is.na(tif_pool[counter_])){
        MASTER_IMAGE_ <- try(brick(tif_pool[counter_] ), T)
        SpatialTarget_utm <- LongLat2CRS(
          long = SpatialTarget_longlat[1],
          lat = SpatialTarget_longlat[2],
          CRS_ref = raster::crs(MASTER_IMAGE_))
        # check inverse of LongLat2CRS
        # SpatialTarget_longlat
        # SpatialPoints(spTransform(SpatialTarget_utm, CRS_longlat),CRS_longlat)

        # exact spatial target - alternative extraction method for pseudo rgb plotting
        if(T == F){
          my_extent <- LongLat2CRS_extent(SpatialTarget_longlat,raster::crs(MASTER_IMAGE_))

          cropped_raster <- raster::crop(MASTER_IMAGE_, y = my_extent)
          #raster::plotRGB(cropped_raster)
          matrix(getValuesBlock(cropped_raster[[band_]],
                                row = 1, nrows = nrow(cropped_raster),
                                col = 1, ncols = ncol(cropped_raster),
                                format = "matrix", lyrs = 1L),
                 ncol = nrow(cropped_raster), byrow=T)
        }
        SpatialTargetCellLoc <- raster::cellFromXY(
          object = MASTER_IMAGE_,
          xy = SpatialTarget_utm)
        SpatialTargetRowCol <- raster::rowColFromCell(
          object = MASTER_IMAGE_,
          cell = SpatialTargetCellLoc )
        if(!is.na(sum(SpatialTargetRowCol))){found_<-T;bad_ <- F}
        if(counter_ > 1000000){stop("ERROR! Target not found anywhere in pool!")}
      }
    }
    if(bad_){
      print(sprintf("Bad at %s. Apparently, no .tif contains the reference point",i))
      bad_indices <- c(bad_indices,i)
    }
    if(!bad_){
      print(sprintf("Good at %s. Extracting and saving image", i))
      # available rows/cols
      rows_available <- nrow( MASTER_IMAGE_ )
      cols_available <- ncol( MASTER_IMAGE_ )


      # define start row/col
      start_row <- SpatialTargetRowCol[1,"row"] - RADIUS_CELLS
      start_col <- SpatialTargetRowCol[1,"col"] - RADIUS_CELLS

      # find end row/col
      end_row <- start_row + DIAMETER_CELLS
      end_col <- start_col + DIAMETER_CELLS

      # perform checks to deal with spilling over image
      if(start_row <= 0){start_row <- 1}
      if(start_col <= 0){start_col <- 1}
      if(end_row > rows_available){ start_row <- rows_available - DIAMETER_CELLS }
      if(end_col > cols_available){ start_col <- cols_available - DIAMETER_CELLS }

      for(iof in 0:0){
        if(is.null(lyrs)){lyrs <- 1:dim(MASTER_IMAGE_)[3] }
        band_iters <- ifelse(grepl(x = save_as, pattern ="csv"),
                             yes = list(lyrs), no = list(1L) )[[1]]
        for(band_ in band_iters){
          if(iof > 0){
            start_row <- sample(1:(nrow(MASTER_IMAGE_)-DIAMETER_CELLS-1),1)
            start_col <- sample(1:(ncol(MASTER_IMAGE_)-DIAMETER_CELLS-1),1)
          }
          SpatialTargetImage_ <- getValuesBlock(MASTER_IMAGE_[[band_]],
                                                row = start_row, nrows = DIAMETER_CELLS,
                                                col = start_col, ncols = DIAMETER_CELLS,
                                                format = "matrix", lyrs = 1L)
          #SpatialTargetImage_ <- matrix(SpatialTargetImage_,ncol = DIAMETER_CELLS, byrow = T)
          if(length(unique(c(SpatialTargetImage_)))<5){ bad_indices <- c(bad_indices,i) }
          check_ <- dim(SpatialTargetImage_) - c(DIAMETER_CELLS,DIAMETER_CELLS)
          if(any(check_ < 0)){print("WARNING: CHECKS FAILED"); browser()}
          # Tests:
          #pdf("~/Downloads/test.pdf");image2(SpatialTargetImage_, main = paste(round(SpatialTarget_longlat,4L),collapse = ",") );dev.off()
          # SpatialTarget_longlat
          if(grepl(x = save_as, pattern ="tif")){
            # in progress
          }
          if(grepl(x = save_as, pattern ="csv")){
            if(iof == 0){
              data.table::fwrite(file = sprintf("%s/Key%s_BAND%s.csv",
                                                save_folder, keys[i], band_),
                                 data.table::as.data.table(SpatialTargetImage_))
            }
            #if(iof > 0){
            #data.table::fwrite(file = sprintf("./Data/Uganda2000_processed_comparisons/Key%s_%s_BAND%s.csv",
            #row.names(GeoKeyMat)[i],iof,band_),
            #data.table::as.data.table(SpatialTargetImage_)) }
          }
        }
      }
    }
  }
}
