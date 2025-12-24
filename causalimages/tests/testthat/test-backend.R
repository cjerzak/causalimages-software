#!/usr/bin/env Rscript

test_that("BuildBackend works", {
  skip_on_cran()
  skip_if_not(file.exists("/Users/cjerzak/miniforge3/bin/conda"),
              "Conda not found at expected path")

  # setup backend, conda points to location of conda binary
  # note: This function requires an Internet connection
  # you can find out a list of conda paths via:
  # system("which conda")
  causalimages::BuildBackend(conda = "/Users/cjerzak/miniforge3/bin/conda")
  expect_true(TRUE)
  print("Done with BuildBackend() test!")
})

