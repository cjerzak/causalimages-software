#' Get the spatial point of long/lat coordinates
#'
#' Convert longitude and latitude coordinates to a different coordinate reference
#' system (CRS).
#'
#' @param long Vector of numeric longitudes.
#' @param lat Vector of numeric latitudes.
#' @param CRS_ref A CRS into which the long-lat point should be projected.
#'
#' @return Numeric vector of length two giving the coordinates of the supplied
#'   location in the CRS defined by `CRS_ref`.
#'
#' @examples
#' # (Not run)
#' #spatialPt <- LongLat2CRS(long = 49.932,
#' #                 lat = 35.432,
#' #                 CRS_ref = sf::st_crs("+proj=lcc +lat_1=48 +lat_2=33 +lon_0=-100 +ellps=WGS84"))
#' @export
#' @md
#'
LongLat2CRS <- function(long, lat, CRS_ref){
  point_longlat <- sf::st_as_sf(
    data.frame(long = as.numeric(long), lat = as.numeric(lat)),
    coords = c("long", "lat"),
    crs = 4326
  )
  point_longlat_ref <- sf::st_transform(point_longlat, crs = sf::st_crs(CRS_ref))
  coords_ <- sf::st_coordinates(point_longlat_ref)[1, ]
  return(coords_)
}

LongLat2CRS_extent <- function(point_longlat,
                               CRS_ref,
                               target_km_diameter = 10){
  target_km <- target_km_diameter
  offset <- 1/111 * (target_km/2)
  point_longlat1 <- c(long = as.numeric(point_longlat[1]) - offset,
                      lat = as.numeric(point_longlat[2]) - offset)
  point_longlat2 <- c(long = as.numeric(point_longlat[1]) + offset,
                      lat = as.numeric(point_longlat[2]) + offset)
  pts <- sf::st_as_sf(rbind(point_longlat1, point_longlat2),
                      coords = c("long", "lat"), crs = 4326)
  pts_ref <- sf::st_transform(pts, crs = sf::st_crs(CRS_ref))
  coords_ <- sf::st_coordinates(pts_ref)
  return(raster::extent(min(coords_[,1]), max(coords_[,1]),
                        min(coords_[,2]), max(coords_[,2])))
}

# converts python builtin to list
p2l <- function(zer){
  if("python.builtin.bytes" %in% class(zer)){ zer <- list(zer) }
  return( zer )
}

# zips two lists
rzip<-function(l1,l2){  fl<-list(); for(aia in 1:length(l1)){ fl[[aia]] <- list(l1[[aia]], l2[[aia]]) }; return( fl  ) }

# reshapes
reshape_fxn_DEPRECIATED <- function(input_){
    ## DEPRECIATED
    cienv$tf$reshape(input_, list(cienv$tf$shape(input_)[1],
                            cienv$tf$reduce_prod(cienv$tf$shape(input_)[2:5])))
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
  if("tensorflow.tensor" %in% class( x )){ x <- cienv$tf$cast(x, dtype = dtype  ) }
  if(!"tensorflow.tensor" %in% class( x )){ x <- cienv$tf$constant(x, dtype = dtype  ) }
  return( x )
}

#' print2  print() with timestamps
#'
#' A function prints a string with date and time. 
#'
#' @param x Character string to be printed, with date and time. 
#'
#' @return Prints with date and time. 
#'
#' @examples
#' message("Hello world")
#' @export
#' @md
#'
print2 <- function(text, quiet = F){
  if(!quiet){ print( sprintf("[%s] %s" ,format(Sys.time(), "%Y-%m-%d %H:%M:%S"),text) ) }
}

#' message2 message() with timestamps
#'
#' A function that displays a message with date and time. 
#'
#' @param text Character string to be displayed as message, with date and time. 
#' @param quiet Logical. If TRUE, suppresses the message output. Default is FALSE.
#'
#' @return Displays message with date and time to stderr. 
#'
#' @examples
#' message2("Hello world")
#' message2("Process completed", quiet = FALSE)
#' @export
#' @md
#'
message2 <- function(text, quiet = FALSE){
  if(!quiet){ 
    message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), text)) 
  }
}

# LE <- function(l_, name_){ return( unlist(l_)[[name_]] ) }
# l_ <- DenseList;name <-"Tau_d1"
LE <- function(l_, key) {
  # Recursive helper function
  search_recursive <- function(list_element, key) {
    # Check if the current element is a list
    if (is.list(list_element)) {
      # If it's a list, check if the key exists in this list
      if (key %in% names(list_element)) {
        return(list_element[[key]])
      }
      # Otherwise, iterate over its elements
      for (item in list_element) {
        found <- search_recursive(item, key)
        if (!is.null(found)) {
          return(found)
        }
      }
    }
    return(NULL)
  }

  # Start the recursive search
  return(search_recursive(l_, key))
}

LE_index <- function(l_, key) {
  # Recursive helper function
  search_recursive <- function(list_element, key, path) {
    # Check if the current element is a list
    if(is.list(list_element)){

      # If it's a list, check if the key exists in this list
      if(key %in% names(list_element) & (length(names(list_element)) == 1)){
        return( c(path,
                  ifelse("list" %in% class(list_element), yes = 1, no = NULL)) )
      }
      if(key %in% names(list_element) & (length(names(list_element)) > 1)){
        return( c(path,
                  which(names(list_element) == key),
                  ifelse("list" %in% class(list_element), yes = 1, no = NULL) ) )
      }

      # Otherwise, iterate over its elements
      for (i in seq_along(list_element)) {
        new_path <- c(path, i)
        found <- search_recursive(list_element[[i]], key, new_path)
        if (!is.null(found)) { return( found ) }
      }
    }
    return(NULL)
  }

  # Start the recursive search
  return(search_recursive(l_, key, c()))
}

GlobalPartition <- function(zer, eq_fxn){
  yes_branches <- rrapply::rrapply(zer,f=function(zerr){
    unlist(ifelse(eq_fxn(zerr),yes = list(zerr), no = list(NULL))[[1]])
  },how="list")
  no_branches <- rrapply::rrapply(zer,f=function(zerrr){
    unlist(ifelse(eq_fxn(zerrr),yes = list(NULL), no = list(zerrr))[[1]])
  },how="list")
  list(yes_branches,no_branches)
}
PartFxn <- function(zerz){ !"first_time_index" %in% names(zerz)}

AddQuotes <- function(text) { gsub("\\[\\[([A-Za-z]\\w*)", "\\[\\['\\1'", text) }
LinearizeNestedList <- function (NList,
                                 LinearizeDataFrames = FALSE,
                                 NameSep = "/",
                                 ForceNames = FALSE){
  stopifnot(is.character(NameSep), length(NameSep) == 1)
  stopifnot(is.logical(LinearizeDataFrames), length(LinearizeDataFrames) == 1)
  stopifnot(is.logical(ForceNames), length(ForceNames) == 1)
  if (!is.list(NList))
    return(NList)
  if (is.null(names(NList)) | ForceNames == TRUE)
    names(NList) <- as.character(1:length(NList))
  if (is.data.frame(NList) & LinearizeDataFrames == FALSE)
    return(NList)
  if (is.data.frame(NList) & LinearizeDataFrames == TRUE)
    return(as.list(NList))
  A <- 1
  B <- length(NList)
  while (A <= B) {
    Element <- NList[[A]]
    EName <- names(NList)[A]
    if (is.list(Element)) {
      Before <- if (A == 1)
        NULL
      else NList[1:(A - 1)]
      After <- if (A == B)
        NULL
      else NList[(A + 1):B]
      if (is.data.frame(Element)) {
        if (LinearizeDataFrames == TRUE) {
          Jump <- length(Element)
          NList[[A]] <- NULL
          if(is.null(names(Element)) | ForceNames == TRUE)
            names(Element) <- as.character(1:length(Element))
          Element <- as.list(Element)
          names(Element) <- paste(EName, names(Element),
                                  sep = NameSep)
          NList <- c(Before, Element, After)
        }
        Jump <- 1
      }
      else {
        NList[[A]] <- NULL
        if (is.null(names(Element)) | ForceNames == TRUE)
          names(Element) <- as.character(1:length(Element))
        Element <- LinearizeNestedList(Element, LinearizeDataFrames,
                                       NameSep, ForceNames)
        names(Element) <- AddQuotes( paste(EName,
                                           names(Element),
                                           sep = NameSep) )
        Jump <- length(Element)
        NList <- c(Before, Element, After)
      }
    }
    else {
      Jump <- 1
    }
    A <- A + Jump
    B <- length(NList)
  }
  return(NList)
}


ai <- as.integer

se <- function(x){ x <- c(na.omit(x)); return(sqrt(var(x)/length(x)))}

LocalFxnSource <- function(fxn, evaluation_environment){ 
  fxn_text <- paste(deparse(fxn), collapse="\n")
  fxn_text <- gsub(fxn_text,pattern="function \\(\\)", replace="")
  eval( parse( text = fxn_text ), envir = evaluation_environment )
}

FilterBN <- function(l_){ cienv$eq$partition(l_, function(l__){"first_time_index" %in% names(l__)}) }

cienv <- new.env( parent = emptyenv() )