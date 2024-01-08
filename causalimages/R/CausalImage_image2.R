#!/usr/bin/env Rscript
#' Visualizing matrices as heatmaps with correct north-south-east-west orientation
#'
#' A function for generating a heatmap representation of a matrix with correct spatial orientation.
#'
#' @param x The numeric matrix to be visualized.
#' @param xlab The x-axis labels.
#' @param ylab The y-axis labels.
#' @param xaxt The x-axis tick labels.
#' @param yaxt The y-axis tick labels.
#' @param main The main figure label.
#' @param cex.main The main figure label sizing factor.
#' @param col.lab Axis label color.
#' @param col.main Main label color.
#' @param cex.lab Cex for the labels.
#' @param box Draw a box around the image?

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

image2 = function(x,
                  xaxt=NULL,
                  yaxt = NULL,
                  xlab = "",
                  ylab = "",
                  main=NULL,
                  cex.main = NULL,
                  col.lab = "black",
                  col.main = "black",
                  cex.lab = 1.5,
                  box=F){
  image((t(x)[,nrow(x):1]),
        axes = F,
        main = main,
        xlab = xlab,
        ylab = ylab,
        xaxs = "i",
        cex.lab = cex.lab,
        col.lab = col.lab,
        col.main = col.main,
        cex.main = cex.main)
  if(box == T){box()}
  if(!is.null(xaxt)){ axis(1, at = 0:(nrow(x)-1)/nrow(x)*1.04, tick=F,labels = (xaxt),cex.axis = 1,las = 1)  }
  if(!is.null(yaxt)){ axis(2, at = 0:(nrow(x)-1)/nrow(x)*1.04, tick=F,labels = rev(yaxt),cex.axis = 1,las = 2)  }
}

f2n <- function(.){as.numeric(as.character(.))}
