#!/usr/bin/env Rscript

test_that("Full test suite works", {
  skip_on_cran()

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

    ##########################################
    # Environment-aware path setup
    # Detects GitHub Actions vs local environment and sets appropriate paths
    ##########################################

    # Detect environment: GitHub Actions sets GITHUB_WORKSPACE
    IS_CI <- nzchar(Sys.getenv("GITHUB_WORKSPACE"))

    # Set base path for test data
    # CI: use workspace tmp directory; Local: use ~/Downloads
    if (IS_CI) {
      TEST_DATA_DIR <- file.path(Sys.getenv("GITHUB_WORKSPACE"), "tmp", "test_data")
      REPO_ROOT <- Sys.getenv("GITHUB_WORKSPACE")
    } else {
      TEST_DATA_DIR <- path.expand("~/Downloads")
      REPO_ROOT <- path.expand("~/Documents/causalimages-software")
    }

    # Ensure test data directories exist
    dir.create(TEST_DATA_DIR, recursive = TRUE, showWarnings = FALSE)
    dir.create(file.path(TEST_DATA_DIR, "ImageTutorial"), recursive = TRUE, showWarnings = FALSE)

    # Export these for use by individual test files
    assign("TEST_DATA_DIR", TEST_DATA_DIR, envir = .GlobalEnv)
    assign("REPO_ROOT", REPO_ROOT, envir = .GlobalEnv)
    assign("IS_CI", IS_CI, envir = .GlobalEnv)

    print(sprintf("Running tests in %s environment", ifelse(IS_CI, "CI", "local")))
    print(sprintf("Test data directory: %s", TEST_DATA_DIR))
    print(sprintf("Repository root: %s", REPO_ROOT))

    ##########################################
    # Run test suite
    ##########################################

    print("Starting image TfRecords"); setwd(TEST_DATA_DIR);
    TfRecordsTest <- try(source(file.path(REPO_ROOT, "causalimages/tests/testthat/test-tfrecords.R")),T)
    if("try-error" %in% class(TfRecordsTest)){ stop("Failed at TfRecordsTest (1)") }
    while(!is.null(dev.list())) try(dev.off(), TRUE)

    print("Starting ImageRepTest"); setwd(TEST_DATA_DIR);
    ImageRepTest <- try(source(file.path(REPO_ROOT, "causalimages/tests/testthat/test-representations.R")),T)
    if("try-error" %in% class(ImageRepTest)){ stop("Failed at ImageRepTest (2)") }
    while(!is.null(dev.list())) try(dev.off(), TRUE)

    print("Starting ImConfoundTest"); setwd(TEST_DATA_DIR);
    ImConfoundTest <- try(source(file.path(REPO_ROOT, "causalimages/tests/testthat/test-confounding.R")),T)
    if("try-error" %in% class(ImConfoundTest)){ stop("Failed at ImConfoundTest (3)") }
    while(!is.null(dev.list())) try(dev.off(), TRUE)

    #print("Starting HetTest");  setwd(TEST_DATA_DIR);
    #HetTest <- try(source(file.path(REPO_ROOT, "causalimages/tests/testthat/test-heterogeneity.R")),T)
    #if("try-error" %in% class(HetTest)){ stop("Failed at HetTest") }; try(dev.off(), T)
  }, T)

  if('try-error' %in% class(tryTests)){ print("At least one test failed"); print( tryTests ); stop(tryTests) }
  if(!'try-error' %in% class(tryTests)){ print("All tests succeeded!") }

  expect_true(TRUE)
})
