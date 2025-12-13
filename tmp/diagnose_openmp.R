#!/usr/bin/env Rscript
# Comprehensive OpenMP conflict diagnostic for causalimages
# Run with: R --vanilla < diagnose_openmp.R
#
# This script diagnoses OpenMP library conflicts that cause segfaults
# when mixing torch with numpy/jax/tensorflow.

cat(strrep("=", 62), "\n")
cat("OpenMP Conflict Diagnostic for causalimages\n")
cat(strrep("=", 62), "\n\n")

# System info
cat("System Information:\n")
cat("  Platform:", R.version$platform, "\n")
cat("  R version:", R.version.string, "\n")
cat("  PID:", Sys.getpid(), "\n\n")

library(reticulate)
use_condaenv("CausalImagesEnv", required = TRUE)

cat("Python config:\n")
py_config <- py_config()
cat("  Python:", py_config$python, "\n")
cat("  NumPy:", tryCatch(import("numpy")$`__version__`, error = function(e) "not found"), "\n\n")

# Function to list OpenMP libs
list_omp_libs <- function(label) {
  cat(sprintf("\n[%s] OpenMP libraries loaded:\n", label))
  pid <- Sys.getpid()
  if (Sys.info()["sysname"] == "Darwin") {
    cmd <- sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' | awk '{print $9}' | sort -u", pid)
  } else {
    # Linux
    cmd <- sprintf("cat /proc/%d/maps 2>/dev/null | grep -oE '/[^ ]*lib(iomp|omp|gomp)[^ ]*' | sort -u", pid)
  }
  libs <- system(cmd, intern = TRUE)
  if (length(libs) == 0) {
    cat("  (none found)\n")
  } else {
    for (lib in libs) cat("  ", lib, "\n")
  }
  return(libs)
}

# Test sequence that simulates the problematic code path
cat("\n--- Simulating problematic code path ---\n")
cat("This simulates: WriteTfRecord() -> GetImageRepresentations(pretrainedModel='clip')\n\n")

cat("Step 1: Import jax (simulates WriteTfRecord calling initialize_jax)\n")
tryCatch({
  jax <- import("jax")
  cat("  jax imported OK\n")
}, error = function(e) cat("  ERROR:", e$message, "\n"))

libs_after_jax <- list_omp_libs("After JAX")

cat("\nStep 2: Import numpy\n")
tryCatch({
  np <- import("numpy")
  cat("  numpy imported OK\n")
}, error = function(e) cat("  ERROR:", e$message, "\n"))

libs_after_numpy <- list_omp_libs("After NumPy")

cat("\nStep 3: Import tensorflow\n")
tryCatch({
  tf <- import("tensorflow")
  cat("  tensorflow imported OK\n")
}, error = function(e) cat("  ERROR:", e$message, "\n"))

libs_after_tf <- list_omp_libs("After TensorFlow")

cat("\nStep 4: Import torch (this is where segfault typically occurs)\n")
tryCatch({
  torch <- import("torch")
  cat("  torch imported OK\n")
}, error = function(e) cat("  ERROR:", e$message, "\n"))

libs_after_torch <- list_omp_libs("After PyTorch")

# Analyze results
cat("\n--- Analysis ---\n")
all_libs <- unique(c(libs_after_jax, libs_after_numpy, libs_after_tf, libs_after_torch))

has_libiomp <- any(grepl("libiomp", all_libs))
has_libomp <- any(grepl("libomp", all_libs))
has_libgomp <- any(grepl("libgomp", all_libs))

conflict_count <- sum(c(has_libiomp, has_libomp, has_libgomp))

if (conflict_count > 1) {
  cat("\n*** CONFLICT DETECTED ***\n")
  cat("Multiple OpenMP implementations are loaded:\n")
  if (has_libiomp) cat("  - libiomp5 (Intel OpenMP, typically from MKL/numpy)\n")
  if (has_libomp) cat("  - libomp (LLVM OpenMP, typically from PyTorch)\n")
  if (has_libgomp) cat("  - libgomp (GNU OpenMP, typically from GCC-compiled libs)\n")
  cat("\nThis WILL cause OMP Error #15 and potential segfaults.\n")
  cat("\nRecommended fixes (in order of preference):\n")
  cat("1. Import torch BEFORE jax/numpy/tensorflow\n")
  cat("2. Use nomkl numpy: conda install nomkl numpy\n")
  cat("3. Set MKL_THREADING_LAYER=GNU before importing\n")
  cat("4. Last resort: KMP_DUPLICATE_LIB_OK=TRUE (masks the issue)\n")
} else if (conflict_count == 1) {
  cat("\nNo OpenMP conflict detected - single OpenMP implementation in use.\n")
  cat("Segfaults unlikely to be caused by OpenMP.\n")
} else {
  cat("\nNo OpenMP libraries detected. Using single-threaded mode or platform defaults.\n")
}

cat("\n--- Test Complete ---\n")
cat("If this script completed without segfault, the issue may be\n")
cat("intermittent or triggered by specific operations.\n")
