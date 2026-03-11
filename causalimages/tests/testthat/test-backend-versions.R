test_that("BuildBackend is exported", {
  exports <- getNamespaceExports("causalimages")

  expect_true("BuildBackend" %in% exports)
  expect_true("ci_supported_backend_versions" %in% exports)
  expect_true(is.function(causalimages::BuildBackend))
})

test_that("supported backend versions pin a compatible transformers stack", {
  versions <- causalimages::ci_supported_backend_versions()

  expect_equal(unname(versions["python"]), "3.13")
  expect_equal(unname(versions["transformers"]), "5.3.0")
  expect_equal(unname(versions["huggingface-hub"]), "1.6.0")
  expect_equal(unname(versions["tokenizers"]), "0.22.2")

  specs <- causalimages:::ci_backend_version_specs(
    c("transformers", "huggingface-hub", "tokenizers")
  )
  expect_equal(
    unname(specs),
    c(
      "transformers==5.3.0",
      "huggingface-hub==1.6.0",
      "tokenizers==0.22.2"
    )
  )
})
