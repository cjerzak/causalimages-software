
#!/usr/bin/env Rscript
#' Matrix visualization as heatmap with correct north-south-east-east orientation
#'
#' A function for generating a heatmap representation of a matrix with correct spatial orientation.
#'
#' @usage
#'
#' image2( x, ... )
#'
#' @param x A numeric matrix.
#' @param obsY A numeric
#'
#' @return A heatmap representation of the matrix, `x`, with correct north/south/east/west orientation.
#'
image2 = function(mat,xaxt=NULL,yaxt = NULL,main=NULL,cex.main = NULL,box=F){
  image((t(mat)[,nrow(mat):1]), axes = F, main = main,xaxs = "i",cex.main = cex.main)
  if(box == T){box()}
  if(!is.null(xaxt)){ axis(1, at = 0:(nrow(mat)-1)/nrow(mat)*1.04, tick=F,labels = (xaxt),cex.axis = 1,las = 1)  }
  if(!is.null(yaxt)){ axis(2, at = 0:(nrow(mat)-1)/nrow(mat)*1.04, tick=F,labels = rev(yaxt),cex.axis = 1,las = 2)  }
}

# zips two lists
rzip<-function(l1,l2){  fl<-list(); for(aia in 1:length(l1)){ fl[[aia]] <- list(l1[[aia]], l2[[aia]]) }; return( fl  ) }

# reshapes
reshape_fxn_DEPRECIATED <- function(input_){
    ## DEPRECIATED
    tf$reshape(input_, list(tf$shape(input_)[1],
                            tf$reduce_prod(tf$shape(input_)[2:5])))
}

