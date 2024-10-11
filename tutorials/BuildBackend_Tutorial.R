#!/usr/bin/env Rscript
{
  # clear workspace 
  rm(list = ls()); options(error = NULL)
  
  # remote install latest version of the package
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

  # local install for development team
  # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

  # in general, you will simply use: 
  causalimages::BuildBackend()
  
  # Note: This function requires an Internet connection
  # Note: With default arguments, a conda environment called 
  # "CausalImagesEnv" will be created with required packages saved within. 
  
  # Advanced tip: 
  # if you need to points to the location of a specific version of python
  # in conda where packages will be downloaded, 
  # you can use: 
  #causalimages::BuildBackend(conda = "/Users/cjerzak/miniforge3/bin/python")
}

