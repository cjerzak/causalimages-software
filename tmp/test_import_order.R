#!/usr/bin/env Rscript
# Minimal reproducer for torch/numpy import order segfault
# Run with: R --vanilla < test_import_order.R
#
# This script tests two import orders:
# (a) torch THEN numpy - expected to work
# (b) numpy THEN torch - may crash due to OpenMP conflict

args <- commandArgs(trailingOnly = TRUE)
test_case <- if (length(args) > 0) args[1] else "a"

library(reticulate)

# Use the CausalImagesEnv conda environment
use_condaenv("CausalImagesEnv", required = TRUE)

cat("=== Test case:", test_case, "===\n")
cat("R PID:", Sys.getpid(), "\n")
cat("Run this to check OpenMP libs:\n")
cat(sprintf("  lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp'\n\n", Sys.getpid()))

if (test_case == "a") {
  cat("=== Import order: torch THEN numpy ===\n")

  cat("Importing torch...\n")
  torch <- import("torch")
  cat("torch imported successfully\n")

  cat("\nOpenMP libs after torch:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\nImporting numpy...\n")
  np <- import("numpy")
  cat("numpy imported successfully\n")

  cat("\nOpenMP libs after numpy:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\nImporting jax...\n")
  jax <- import("jax")
  cat("jax imported successfully\n")

  cat("\nFinal OpenMP libs:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\n=== Test case 'a' PASSED ===\n")

} else if (test_case == "b") {
  cat("=== Import order: numpy THEN torch ===\n")

  cat("Importing numpy...\n")
  np <- import("numpy")
  cat("numpy imported successfully\n")

  cat("\nOpenMP libs after numpy:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\nImporting torch...\n")
  torch <- import("torch")
  cat("torch imported successfully\n")

  cat("\nOpenMP libs after torch:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\n=== Test case 'b' PASSED ===\n")

} else if (test_case == "c") {
  cat("=== Import order: jax THEN torch (simulates current bug) ===\n")

  cat("Importing jax...\n")
  jax <- import("jax")
  cat("jax imported successfully\n")

  cat("\nOpenMP libs after jax:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\nImporting torch...\n")
  torch <- import("torch")
  cat("torch imported successfully\n")

  cat("\nOpenMP libs after torch:\n")
  system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo 'No OpenMP libs found'", Sys.getpid()))

  cat("\n=== Test case 'c' PASSED ===\n")
}

cat("\nAll imports completed without segfault.\n")
