#!/usr/bin/env Rscript
# Test the actual import path for swin model
# This mimics what RunAppl_AnalysisLoop.R does

cat("=== Testing swin import path ===\n")
cat("PID:", Sys.getpid(), "\n\n")

# Load the causalimages package
library(causalimages)

cat("1. Check if pretrained_model_requires_torch exists:\n")
cat("   exists:", exists("pretrained_model_requires_torch"), "\n")

cat("\n2. Test pretrained_model_requires_torch('swin'):\n")
result <- causalimages:::pretrained_model_requires_torch("swin")
cat("   result:", result, "\n")

cat("\n3. Check cienv state before any initialization:\n")
cat("   'jax' in cienv:", "jax" %in% ls(envir = causalimages:::cienv), "\n")
cat("   'torch' in cienv:", "torch" %in% ls(envir = causalimages:::cienv), "\n")

cat("\n4. Now simulating GetImageRepresentations call with pretrainedModel='swin':\n")
cat("   (This should call initialize_torch BEFORE initialize_jax)\n\n")

# Manually trace what GetImageRepresentations does:
pretrainedModel <- "swin"
conda_env <- "CausalImagesEnv"
conda_env_required <- TRUE
Sys.setenv_text <- NULL

cat("Step 4a: Check if torch initialization is needed:\n")
if (causalimages:::pretrained_model_requires_torch(pretrainedModel)) {
  cat("   YES - pretrained_model_requires_torch returned TRUE\n")
  if (!"torch" %in% ls(envir = causalimages:::cienv)) {
    cat("   Calling initialize_torch()...\n")
    causalimages:::initialize_torch(conda_env = conda_env,
                                    conda_env_required = conda_env_required,
                                    Sys.setenv_text = Sys.setenv_text)
    cat("   initialize_torch() completed\n")
  }
}

cat("\nStep 4b: Check OpenMP libs after torch init:\n")
system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo '   (none found)'", Sys.getpid()))

cat("\nStep 4c: Now calling initialize_jax()...\n")
if (!"jax" %in% ls(envir = causalimages:::cienv)) {
  causalimages:::initialize_jax(conda_env = conda_env,
                                conda_env_required = conda_env_required,
                                Sys.setenv_text = Sys.setenv_text)
  cat("   initialize_jax() completed\n")
}

cat("\nStep 4d: Final OpenMP libs check:\n")
system(sprintf("lsof -p %d 2>/dev/null | grep -E 'libiomp|libomp|libgomp' || echo '   (none found)'", Sys.getpid()))

cat("\n=== Test completed without segfault ===\n")
