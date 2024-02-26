#!/usr/bin/env Rscript
{
  # remote install latest version of the package
  # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

  # local install for development team
  # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

  # setup backend, conda points to location of python in conda where packages will be downloaded
  # note: This function requires an Internet connection
  # you can find out a list of conda Python paths via:
  # system("which python")
  causalimages::BuildBackend(conda = "/Users/cjerzak/miniforge3/bin/python")
}

