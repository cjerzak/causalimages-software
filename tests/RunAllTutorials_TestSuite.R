#!/usr/bin/env Rscript
{
  ##########################################
  # Code for testing most functionalities of CausalImage on your hardware.
  # Current tests failing on optimizing video representation on METAL Other tests succeed.
  ##########################################
  tryTests <- try({
    # remote install latest version of the package
    # devtools::install_github(repo = "cjerzak/causalimages-software/causalimages")

    # local install for development team
    # install.packages("~/Documents/causalimages-software/causalimages",repos = NULL, type = "source",force = F)

    # Before running tests, it may be necessary to (re)build the backend:
    # causalimages::BuildBackend()
    # See ?causalimages::BuildBackend for help. Remember to re-start your R session after re-building.

    print("Starting image TfRecords tutorial..."); setwd("~");
    t_ <- try(source("~/Documents/causalimages-software/tutorials/UsingTFRecords_Test.R"),T)
    if("try-error" %in% class(t_)){ stop("Failed at TfRecords tutorial...") }; try(dev.off(), T)

    Sys.sleep(1L); print("Starting image representations tutorial..."); setwd("~");
    t_ <- try(source("~/Documents/causalimages-software/tests/ExtractImageRepresentations_Test.R"),T)
    if("try-error" %in% class(t_)){ stop("Failed at image representations tutorial...") }; try(dev.off(), T)

    Sys.sleep(1L); print("Starting image confounding tutorial..."); setwd("~");
    t_ <- try(source("~/Documents/causalimages-software/tests/AnalyzeImageConfounding_Test.R"),T)
    if("try-error" %in% class(t_)){ stop("Failed at confounding tutorial...") }; try(dev.off(), T)

    Sys.sleep(1L); print("Starting image heterogeneity tutorial...");  setwd("~");
    t_ <- try(source("~/Documents/causalimages-software/tests/AnalyzeImageHeterogeneity_Test.R"),T)
    if("try-error" %in% class(t_)){ stop("Failed at heterogeneity tutorial...") }; try(dev.off(), T)
  }, T)

  if('try-error' %in% class(tryTests)){ print("At least one test failed..."); print( tryTests ) }
  if(!'try-error' %in% class(tryTests)){ print("All tests succeeded!") }
}
