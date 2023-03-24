#!/usr/bin/env Rscript
#' Visualizing matrices as heatmaps with correct north-south-east-west orientation
#'
#' A function for generating a heatmap representation of a matrix with correct spatial orientation.
#'
#' @usage
#'
#' image2( x )
#'
#' @param x (required) The numeric matrix to be visualized.
#' @param xaxt (default = `""`) The x-axis label.
#' @param yaxt (default = `""`) The y-axis label.
#' @param main (default = `""`) The main figure label.
#' @param cex.main (default = `1.`) The main figure label sizing factor.
#' @param box (default = `F`) Should a box be plotted around the image?
#'
#' @return Returns a heatmap representation of the matrix, `x`, with correct north/south/east/west orientation.
#'
#' @examples
#' #set seed
#' set.seed(1)
#'
#' #Geneate data
#' x <- matrix(rnorm(50*50), ncol = 50)
#' diag(x) <- 3
#'
#' # create plot
#' image2(x, main = "Example Text", cex.main = 2)
#'
#' @export
#' @md
#'
image2 = function(x,xaxt=NULL,yaxt = NULL,main=NULL,cex.main = NULL,box=F){
  image((t(x)[,nrow(x):1]), axes = F, main = main,xaxs = "i",cex.main = cex.main)
  if(box == T){box()}
  if(!is.null(xaxt)){ axis(1, at = 0:(nrow(x)-1)/nrow(x)*1.04, tick=F,labels = (xaxt),cex.axis = 1,las = 1)  }
  if(!is.null(yaxt)){ axis(2, at = 0:(nrow(x)-1)/nrow(x)*1.04, tick=F,labels = rev(yaxt),cex.axis = 1,las = 2)  }
}

f2n <- function(.){as.numeric(as.character(.))}

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
