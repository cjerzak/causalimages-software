#!/usr/bin/env Rscript
{
  ##########################################
  # code for testing most functionalities of CausalImage on your hardware
  ##########################################
  tryTests <- try({
    # remote install latest version of the package
    # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

    # local install for development team
    # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

    print("Starting image confounding tutorial..."); setwd("~");
    source("~/Documents/causalimages-software/tutorials/UsingTFRecords_Tutorial.R")

    print("Starting image representations tutorial..."); setwd("~");
    source("~/Documents/causalimages-software/tutorials/ExtractImageRepresentations_Tutorial.R")

    print("Starting image heterogeneity tutorial...");  setwd("~");
    source("~/Documents/causalimages-software/tutorials/AnalyzeImageHeterogeneity_FullTutorial.R")

    print("Starting image confounding tutorial..."); setwd("~");
    source("~/Documents/causalimages-software/tutorials/AnalyzeImageConfounding_Tutorial.R")
  }, T)

  if('try-error' %in% class(tryTests)){ print("At least one test failed...") }
  if(!'try-error' %in% class(tryTests)){ print("Tests succeeded!") }
}
