#!/usr/bin/env Rscript
{
  # remove all dim calls to arrays 
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
    
    options(error = NULL)
    
    print("Starting image TfRecords"); setwd("~");
    TfRecordsTest <- try(source("~/Documents/causalimages-software/causalimages/tests/Test_UsingTFRecords.R"),T)
    if("try-error" %in% class(TfRecordsTest)){ stop("Failed at TfRecordsTest (1)") }; try(dev.off(), T)

    print("Starting ImageRepTest"); setwd("~");
    ImageRepTest <- try(source("~/Documents/causalimages-software/causalimages/tests/Test_ExtractImageRepresentations.R"),T)
    if("try-error" %in% class(ImageRepTest)){ stop("Failed at ImageRepTest (2)") }; try(dev.off(), T)

    print("Starting ImConfoundTest"); setwd("~");
    ImConfoundTest <- try(source("~/Documents/causalimages-software/causalimages/tests/Test_AnalyzeImageConfounding.R"),T)
    if("try-error" %in% class(ImConfoundTest)){ stop("Failed at ImConfoundTest (3)") }; try(dev.off(), T)

    #print("Starting HetTest");  setwd("~");
    #HetTest <- try(source("~/Documents/causalimages-software/causalimages/tests/Test_AnalyzeImageHeterogeneity.R"),T)
    #if("try-error" %in% class(HetTest)){ stop("Failed at HetTest") }; try(dev.off(), T)
  }, T)

  if('try-error' %in% class(tryTests)){ print("At least one test failed"); print( tryTests ); stop(tryTests) }
  if(!'try-error' %in% class(tryTests)){ print("All tests succeeded!") }
}
