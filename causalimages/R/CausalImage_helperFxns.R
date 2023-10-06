#' Get the spatial point of long/lat coordinates
#'
#' A function converts long/lat coordinates into a spatial points object defined by a coordinate reference system (CRS).
#'
#' @usage
#'
#' LongLat2CRS(long, lat, CRS_ref)
#'
#' @param long Vector of numeric longitudes.
#' @param lat Vector of numeric latitudes.
#' @param CRS_ref A CRS into which the long-lat point should be projected.
#'
#' @return Returns the long/lat location as a spatial point in the new CRS defined by `CRS_ref`
#'
#' @examples
#' # (Not run)
#' #spatialPt <- LongLat2CRS(long = 49.932,
#'                   #lat = 35.432,
#'                   #CRS_ref = sp::CRS("+proj=lcc +lat_1=48 +lat_2=33 +lon_0=-100 +ellps=WGS84"))
#' @export
#' @md
#'
LongLat2CRS <- function(long, lat, CRS_ref){
  library(sp)
  CRS_longlat <- CRS("+proj=longlat +datum=WGS84")
  point_longlat <- c(long,lat)
  point_longlat <- data.frame(ID = 1,
                              X = as.numeric(point_longlat[1]),
                              Y = as.numeric(point_longlat[2]))
  coordinates(point_longlat) <- c("X", "Y")
  proj4string(point_longlat) <- CRS_longlat
  SpatialTarget_utm <- SpatialPoints(spTransform(point_longlat, CRS_ref),
                                     CRS_ref)
  return( SpatialTarget_utm )
}

LongLat2CRS_extent <- function(point_longlat,
                               CRS_ref,
                               target_km_diameter = 10){
  CRS_longlat <- CRS("+proj=longlat +datum=WGS84")
  target_km <- 10
  point_longlat1 <- data.frame(ID = 1,
                               X = as.numeric(point_longlat[1])-1/111*(target_km/2),
                               Y = as.numeric(point_longlat[2])-1/111*(target_km/2))
  point_longlat2 <- data.frame(ID = 2,
                               X = as.numeric(point_longlat[1])+1/111*(target_km/2),
                               Y = as.numeric(point_longlat[2])+1/111*(target_km/2))
  point_longlat_mat <- rbind(point_longlat1,point_longlat2)
  coordinates(point_longlat_mat) <- c("X", "Y")
  proj4string(point_longlat_mat) <- CRS_longlat
  point_longlat_mat_ref <- SpatialPoints(spTransform(point_longlat_mat, CRS_ref), CRS_ref)
  return( raster::extent(point_longlat_mat_ref) )
}

# zips two lists
rzip<-function(l1,l2){  fl<-list(); for(aia in 1:length(l1)){ fl[[aia]] <- list(l1[[aia]], l2[[aia]]) }; return( fl  ) }

# reshapes
reshape_fxn_DEPRECIATED <- function(input_){
    ## DEPRECIATED
    tf$reshape(input_, list(tf$shape(input_)[1],
                            tf$reduce_prod(tf$shape(input_)[2:5])))
}

fixZeroEndings <- function(zr,roundAt=2){
  unlist( lapply(strsplit(as.character(zr),split="\\."),function(l_){
    if(length(l_) == 1){ retl <- paste(l_, paste(rep("0",times=roundAt),collapse=""),sep=".") }
    if(length(l_) == 2){
      retl <- paste(l_[1], paste(l_[2], paste(rep("0",times=roundAt-nchar(l_[2])),collapse=""),sep=""),
                    sep = ".") }
    return( retl  )
  }) ) }

r2const <- function(x, dtype){
  if("tensorflow.tensor" %in% class( x )){ }
  if(!"tensorflow.tensor" %in% class( x )){ x <- tf$constant(  x, dtype = dtype  ) }
  return( x )
}
