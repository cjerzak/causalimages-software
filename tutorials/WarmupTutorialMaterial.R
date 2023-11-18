#!/usr/bin/env Rscript

# example video function (this here just appends two identical images (with one rotated) for illustration only)
# in practice, image sequence / video data will be read from disk
acquireVideoRepFromMemory <- function(keys, training = F){
  tmp <- acquireImageFromMemory(keys, training = training)

  if(length(keys) == 1){
    tmp <- array(tmp,dim = c(1L,dim(tmp)[1],dim(tmp)[2],dim(tmp)[3]))
  }

  tmp <- array(tmp,dim = c(dim(tmp)[1],
                           2,
                           dim(tmp)[3],
                           dim(tmp)[4],
                           1L))
  return(  tmp  )
}
