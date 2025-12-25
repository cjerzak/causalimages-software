#' Check data quality for causalimages analyses
#'
#' Validates input data structures before WriteTfRecord or analysis functions.
#' Catches common errors early with informative messages explaining issues and fixes.
#'
#' @param obsW A numeric vector where 0's correspond to control and 1's to treated units (optional).
#' @param obsY A numeric vector containing observed outcomes (optional).
#' @param imageKeysOfUnits A vector mapping observations to image keys (optional).
#' @param uniqueImageKeys A vector of unique image identifiers for tfrecord writing (optional).
#' @param acquireImageFxn A function returning image arrays given keys (optional).
#' @param X An optional numeric matrix of tabular covariates (optional).
#' @param file Path to a tfrecord file (optional).
#' @param dataType String: "image" or "video". Default = "image".
#' @param batchSize Integer for batch size validation. Default = NULL.
#' @param learningRateMax Numeric for learning rate validation. Default = NULL.
#' @param nSGD Integer for SGD iterations validation. Default = NULL.
#' @param testFrac Numeric for test fraction validation. Default = NULL.
#' @param checkContext String: "pre_write" for WriteTfRecord checks, "pre_analysis" for analysis checks. Default = "pre_analysis".
#' @param stopOnError Logical. If TRUE, stops on first error. If FALSE, collects all errors. Default = TRUE.
#' @param quiet Logical. If TRUE, suppresses informative messages. Default = FALSE.
#'
#' @return Invisibly returns a list with:
#' \itemize{
#'   \item \code{valid} Logical indicating if all checks passed.
#'   \item \code{errors} Character vector of error messages (empty if valid).
#'   \item \code{warnings} Character vector of warning messages.
#'   \item \code{summary} Named list with validation results by category.
#' }
#'
#' @examples
#' # Check data before writing TfRecord
#' result <- CheckDataQuality(
#'   uniqueImageKeys = unique(imageKeys),
#'   acquireImageFxn = myImageLoader,
#'   file = "data.tfrecord",
#'   checkContext = "pre_write"
#' )
#'
#' # Check data before analysis
#' result <- CheckDataQuality(
#'   obsW = treatment,
#'   obsY = outcome,
#'   imageKeysOfUnits = imageKeys,
#'   file = "data.tfrecord",
#'   checkContext = "pre_analysis"
#' )
#'
#' @export
#' @md
CheckDataQuality <- function(
    obsW = NULL,
    obsY = NULL,
    imageKeysOfUnits = NULL,
    uniqueImageKeys = NULL,
    acquireImageFxn = NULL,
    X = NULL,
    file = NULL,
    dataType = "image",
    batchSize = NULL,
    learningRateMax = NULL,
    nSGD = NULL,
    testFrac = NULL,
    checkContext = "pre_analysis",
    stopOnError = TRUE,
    quiet = FALSE
) {

  # Initialize collectors
  all_errors <- character(0)
  all_warnings <- character(0)
  summary_results <- list()

  # Helper to collect results and optionally stop
  collect_results <- function(check_name, result) {
    all_errors <<- c(all_errors, result$errors)
    all_warnings <<- c(all_warnings, result$warnings)
    summary_results[[check_name]] <<- list(
      passed = length(result$errors) == 0,
      errors = result$errors,
      warnings = result$warnings
    )

    # Stop early if requested
    if (stopOnError && length(result$errors) > 0) {
      stop(paste(c(
        sprintf("[CheckDataQuality] Validation failed:"),
        "",
        result$errors
      ), collapse = "\n"), call. = FALSE)
    }
  }

  # Get obsY length for reference
  obsY_length <- if (!is.null(obsY)) length(obsY) else NULL

  # Run checks based on context
  if (!quiet) message2("Running data quality checks...")

  # 1. Vector length consistency
  if (!is.null(obsW) || !is.null(obsY) || !is.null(imageKeysOfUnits)) {
    if (!quiet) message2("  Checking vector lengths...")
    collect_results("vector_lengths", .check_vector_lengths(obsW, obsY, imageKeysOfUnits))
  }

  # 2. Binary treatment validation
  if (!is.null(obsW)) {
    if (!quiet) message2("  Checking treatment vector (obsW)...")
    collect_results("binary_treatment", .check_binary_treatment(obsW))
  }

  # 3. Numeric outcome validation
  if (!is.null(obsY)) {
    if (!quiet) message2("  Checking outcome vector (obsY)...")
    collect_results("numeric_outcome", .check_numeric_outcome(obsY))
  }

  # 4. Image key validation
  if (!is.null(imageKeysOfUnits) || !is.null(uniqueImageKeys)) {
    if (!quiet) message2("  Checking image keys...")
    collect_results("image_keys", .check_image_keys(imageKeysOfUnits, uniqueImageKeys))
  }

  # 5. acquireImageFxn testing (only for pre_write context)
  if (checkContext == "pre_write" && !is.null(acquireImageFxn)) {
    if (!quiet) message2("  Testing acquireImageFxn...")
    collect_results("acquire_function", .check_acquire_image_function(
      acquireImageFxn, uniqueImageKeys, dataType
    ))
  }

  # 6. Covariate matrix validation
  if (!is.null(X)) {
    if (!quiet) message2("  Checking covariate matrix (X)...")
    collect_results("covariate_matrix", .check_covariate_matrix(X, obsY_length))
  }

  # 7. TfRecord file validation
  if (!is.null(file) || checkContext == "pre_analysis") {
    if (!quiet) message2("  Checking tfrecord file...")
    collect_results("tfrecord_file", .check_tfrecord_file(file, checkContext))
  }

  # 8. Parameter range validation
  if (!is.null(batchSize) || !is.null(learningRateMax) ||
      !is.null(nSGD) || !is.null(testFrac)) {
    if (!quiet) message2("  Checking parameter ranges...")
    collect_results("parameter_ranges", .check_parameter_ranges(
      batchSize, learningRateMax, nSGD, testFrac, obsY_length
    ))
  }

  # Compile final result
  valid <- length(all_errors) == 0

  if (!quiet) {
    if (valid) {
      message2(sprintf("All %d checks passed.", length(summary_results)))
    } else {
      message2(sprintf("Found %d error(s) and %d warning(s).",
                       length(all_errors), length(all_warnings)))
    }
  }

  # Print warnings if any
  if (length(all_warnings) > 0 && !quiet) {
    for (w in all_warnings) {
      warning(w, call. = FALSE, immediate. = TRUE)
    }
  }

  invisible(list(
    valid = valid,
    errors = all_errors,
    warnings = all_warnings,
    summary = summary_results
  ))
}


# =============================================================================
# Internal Helper Functions
# =============================================================================

#' Check vector length consistency
#' @noRd
.check_vector_lengths <- function(obsW, obsY, imageKeysOfUnits) {
  errors <- character(0)

  # Collect lengths of provided vectors
  lengths <- list()
  if (!is.null(obsW)) lengths$obsW <- length(obsW)
  if (!is.null(obsY)) lengths$obsY <- length(obsY)
  if (!is.null(imageKeysOfUnits)) lengths$imageKeysOfUnits <- length(imageKeysOfUnits)

  # If at least 2 vectors provided, check consistency
  if (length(lengths) >= 2) {
    unique_lengths <- unique(unlist(lengths))
    if (length(unique_lengths) > 1) {
      len_str <- paste(names(lengths), "=", unlist(lengths), collapse = ", ")
      errors <- c(errors, sprintf(
        "Vector length mismatch: %s.
All vectors must have the same length representing n observations.
FIX: Ensure obsW, obsY, and imageKeysOfUnits all have exactly n elements,
where n is the number of units in your study.",
        len_str
      ))
    }
  }

  return(list(errors = errors, warnings = character(0)))
}


#' Check binary treatment vector
#' @noRd
.check_binary_treatment <- function(obsW) {
  errors <- character(0)
  warnings <- character(0)

  # Check numeric type
  if (!is.numeric(obsW)) {
    errors <- c(errors, sprintf(
      "obsW must be numeric, but got class '%s'.
FIX: Convert to numeric with as.numeric(obsW).",
      paste(class(obsW), collapse = ", ")
    ))
    return(list(errors = errors, warnings = warnings))
  }

  # Check for NAs
  na_count <- sum(is.na(obsW))
  if (na_count > 0) {
    errors <- c(errors, sprintf(
      "obsW contains %d NA value(s) (%.1f%% of data).
Treatment assignment must be complete for all observations.
FIX: Remove observations with missing treatment or check data source.",
      na_count, 100 * na_count / length(obsW)
    ))
  }

  # Check binary values (0/1 only)
  unique_vals <- unique(obsW[!is.na(obsW)])
  if (!all(unique_vals %in% c(0, 1))) {
    bad_vals <- unique_vals[!unique_vals %in% c(0, 1)]
    errors <- c(errors, sprintf(
      "obsW must contain only 0's (control) and 1's (treated).
Found invalid values: %s
FIX: Recode treatment variable so control = 0 and treated = 1.",
      paste(sort(bad_vals), collapse = ", ")
    ))
  }

  # Check treatment/control balance (warning only)
  if (length(unique_vals) >= 2 && all(unique_vals %in% c(0, 1))) {
    n_treated <- sum(obsW == 1, na.rm = TRUE)
    n_control <- sum(obsW == 0, na.rm = TRUE)
    prop_treated <- n_treated / (n_treated + n_control)

    if (prop_treated < 0.05 || prop_treated > 0.95) {
      warnings <- c(warnings, sprintf(
        "Treatment imbalance detected: %d treated (%.1f%%), %d control (%.1f%%).
Extreme imbalance may affect estimation precision.",
        n_treated, 100 * prop_treated, n_control, 100 * (1 - prop_treated)
      ))
    }
  }

  # Check for single treatment group
  if (length(unique_vals) == 1) {
    errors <- c(errors, sprintf(
      "obsW contains only %s values. Both treatment (1) and control (0) groups are required.
FIX: Ensure your data contains both treated and control observations.",
      unique_vals[1]
    ))
  }

  return(list(errors = errors, warnings = warnings))
}


#' Check numeric outcome vector
#' @noRd
.check_numeric_outcome <- function(obsY) {
  errors <- character(0)
  warnings <- character(0)

  # Check numeric type
  if (!is.numeric(obsY)) {
    errors <- c(errors, sprintf(
      "obsY must be numeric, but got class '%s'.
FIX: Convert to numeric with as.numeric() or check for factor/character values.",
      paste(class(obsY), collapse = ", ")
    ))
    return(list(errors = errors, warnings = warnings))
  }

  # Check for NAs (warning if < 20%, error if > 20%)
  na_count <- sum(is.na(obsY))
  if (na_count > 0) {
    pct <- 100 * na_count / length(obsY)
    if (pct > 20) {
      errors <- c(errors, sprintf(
        "obsY contains %d NA value(s) (%.1f%% of data).
High missingness may compromise analysis reliability.
FIX: Consider imputation or removing observations with missing outcomes.",
        na_count, pct
      ))
    } else {
      warnings <- c(warnings, sprintf(
        "obsY contains %d NA value(s) (%.1f%% of data).
Package will handle NAs during analysis, but consider investigating.",
        na_count, pct
      ))
    }
  }

  # Check for infinite values
  inf_count <- sum(is.infinite(obsY))
  if (inf_count > 0) {
    errors <- c(errors, sprintf(
      "obsY contains %d infinite value(s) (Inf or -Inf).
FIX: Replace infinite values with NA or reasonable bounds.",
      inf_count
    ))
  }

  # Check for constant outcome (warning)
  non_na_vals <- obsY[!is.na(obsY)]
  if (length(unique(non_na_vals)) == 1) {
    warnings <- c(warnings,
      "obsY has zero variance (constant outcome).
Treatment effect estimation requires outcome variability."
    )
  }

  return(list(errors = errors, warnings = warnings))
}


#' Check image key validation
#' @noRd
.check_image_keys <- function(imageKeysOfUnits, uniqueImageKeys) {
  errors <- character(0)
  warnings <- character(0)

  # Check uniqueImageKeys for WriteTfRecord (must be unique)
  if (!is.null(uniqueImageKeys)) {
    if (length(uniqueImageKeys) != length(unique(uniqueImageKeys))) {
      dup_count <- length(uniqueImageKeys) - length(unique(uniqueImageKeys))
      dup_examples <- head(uniqueImageKeys[duplicated(uniqueImageKeys)], 5)
      errors <- c(errors, sprintf(
        "uniqueImageKeys contains %d duplicate value(s).
Each key must be unique for WriteTfRecord.
Duplicate examples: %s
FIX: Use unique(imageKeysOfUnits) to create uniqueImageKeys.",
        dup_count,
        paste(dup_examples, collapse = ", ")
      ))
    }
  }

  # Check imageKeysOfUnits references valid uniqueImageKeys
  if (!is.null(imageKeysOfUnits) && !is.null(uniqueImageKeys)) {
    missing_keys <- setdiff(unique(imageKeysOfUnits), uniqueImageKeys)
    if (length(missing_keys) > 0) {
      errors <- c(errors, sprintf(
        "%d key(s) in imageKeysOfUnits are not found in uniqueImageKeys.
Missing key examples: %s
FIX: Ensure uniqueImageKeys = unique(imageKeysOfUnits) or that all referenced keys exist.",
        length(missing_keys),
        paste(head(missing_keys, 5), collapse = ", ")
      ))
    }
  }

  # Check for NULL or NA keys
  if (!is.null(imageKeysOfUnits)) {
    na_keys <- sum(is.na(imageKeysOfUnits))
    if (na_keys > 0) {
      errors <- c(errors, sprintf(
        "imageKeysOfUnits contains %d NA value(s).
Each observation must have a valid image key.
FIX: Remove observations with missing image keys or assign valid keys.",
        na_keys
      ))
    }
  }

  if (!is.null(uniqueImageKeys)) {
    na_keys <- sum(is.na(uniqueImageKeys))
    if (na_keys > 0) {
      errors <- c(errors, sprintf(
        "uniqueImageKeys contains %d NA value(s).
FIX: Remove NA values from uniqueImageKeys.",
        na_keys
      ))
    }
  }

  return(list(errors = errors, warnings = warnings))
}


#' Check acquireImageFxn with 1 and 2 keys
#' @noRd
.check_acquire_image_function <- function(acquireImageFxn, uniqueImageKeys, dataType) {
  errors <- character(0)
  warnings <- character(0)

  # Check it's a function
  if (!is.function(acquireImageFxn)) {
    errors <- c(errors, sprintf(
      "acquireImageFxn must be a function, but got class '%s'.
FIX: Define acquireImageFxn as function(keys) { ... } returning image arrays.",
      paste(class(acquireImageFxn), collapse = ", ")
    ))
    return(list(errors = errors, warnings = warnings))
  }

  # Need uniqueImageKeys to test
  if (is.null(uniqueImageKeys) || length(uniqueImageKeys) == 0) {
    warnings <- c(warnings,
      "Cannot test acquireImageFxn without uniqueImageKeys. Provide both for full validation."
    )
    return(list(errors = errors, warnings = warnings))
  }

  # Test with single key
  test_key_1 <- head(uniqueImageKeys, 1)
  result_1 <- tryCatch({
    acquireImageFxn(test_key_1)
  }, error = function(e) {
    errors <<- c(errors, sprintf(
      "acquireImageFxn failed when called with key '%s':
%s
FIX: Ensure function can load images for all keys without errors.",
      test_key_1, e$message
    ))
    return(NULL)
  })

  if (is.null(result_1)) {
    return(list(errors = errors, warnings = warnings))
  }

  # Validate single key output
  dims_1 <- dim(result_1)

  if (dataType == "image") {
    # Expected: (H, W, C) or (1, H, W, C)
    if (!length(dims_1) %in% c(3, 4)) {
      errors <- c(errors, sprintf(
        "acquireImageFxn returned array with %d dimension(s) for single key.
Expected 3 dimensions (height, width, channels) or 4 dimensions (1, height, width, channels).
Got shape: (%s)
FIX: Return array of shape (H, W, C) for single key.",
        length(dims_1), paste(dims_1, collapse = ", ")
      ))
      return(list(errors = errors, warnings = warnings))
    }
  } else if (dataType == "video") {
    # Expected: (T, H, W, C) or (1, T, H, W, C)
    if (!length(dims_1) %in% c(4, 5)) {
      errors <- c(errors, sprintf(
        "acquireImageFxn returned array with %d dimension(s) for video data with single key.
Expected 4 dimensions (time, height, width, channels) or 5 dimensions (1, time, height, width, channels).
Got shape: (%s)
FIX: Return array of shape (T, H, W, C) for single key.",
        length(dims_1), paste(dims_1, collapse = ", ")
      ))
      return(list(errors = errors, warnings = warnings))
    }
  }

  # Test with two keys if available
  if (length(uniqueImageKeys) >= 2) {
    test_keys_2 <- head(uniqueImageKeys, 2)
    result_2 <- tryCatch({
      acquireImageFxn(test_keys_2)
    }, error = function(e) {
      errors <<- c(errors, sprintf(
        "acquireImageFxn failed when called with 2 keys:
%s
FIX: Ensure function handles multiple keys correctly.",
        e$message
      ))
      return(NULL)
    })

    if (!is.null(result_2)) {
      dims_2 <- dim(result_2)

      # Check batch dimension for 2 keys
      if (dataType == "image") {
        # Expected: (2, H, W, C)
        if (length(dims_2) != 4 || dims_2[1] != 2) {
          errors <- c(errors, sprintf(
            "acquireImageFxn returned incorrect shape for 2 keys.
Expected shape: (2, height, width, channels)
Got shape: (%s)
FIX: Ensure function returns (batch, H, W, C) when given multiple keys.
Common fix: Add 'if(length(keys) == 1) { arr <- array(arr, dim = c(1, dim(arr))) }'",
            paste(dims_2, collapse = ", ")
          ))
        } else {
          # Verify spatial dimensions match
          spatial_1 <- if (length(dims_1) == 3) dims_1 else dims_1[2:4]
          spatial_2 <- dims_2[2:4]
          if (!all(spatial_1 == spatial_2)) {
            errors <- c(errors, sprintf(
              "Inconsistent image dimensions between single and batch calls.
Single key spatial dims: (%s)
Batch key spatial dims: (%s)
FIX: Ensure image dimensions are consistent across all keys.",
              paste(spatial_1, collapse = ", "),
              paste(spatial_2, collapse = ", ")
            ))
          }
        }
      } else if (dataType == "video") {
        # Expected: (2, T, H, W, C)
        if (length(dims_2) != 5 || dims_2[1] != 2) {
          errors <- c(errors, sprintf(
            "acquireImageFxn returned incorrect shape for 2 video keys.
Expected shape: (2, time, height, width, channels)
Got shape: (%s)
FIX: Ensure function returns (batch, T, H, W, C) when given multiple keys.",
            paste(dims_2, collapse = ", ")
          ))
        }
      }
    }
  } else {
    warnings <- c(warnings,
      "Only 1 unique image key provided. Cannot fully test batch dimension handling.
Provide at least 2 keys for complete validation."
    )
  }

  # Check for numeric values
  if (!is.numeric(result_1)) {
    errors <- c(errors, sprintf(
      "acquireImageFxn must return numeric array, but got class '%s'.
FIX: Ensure image data is numeric (not integer matrices, character, etc.).",
      paste(class(result_1), collapse = ", ")
    ))
  }

  # Check for NaN/Inf values (warning for NA, error for Inf)
  if (is.numeric(result_1)) {
    if (any(is.na(result_1))) {
      warnings <- c(warnings,
        "acquireImageFxn returned array with NA values. This may cause issues during training."
      )
    }
    if (any(is.infinite(result_1))) {
      errors <- c(errors,
        "acquireImageFxn returned array with infinite values. Images must have finite pixel values.
FIX: Check image loading and ensure no division by zero or log(0) operations."
      )
    }
  }

  # Check channel dimension (warning for unusual values)
  if (is.numeric(result_1) && length(dims_1) >= 3) {
    n_channels <- dims_1[length(dims_1)]
    if (n_channels > 20) {
      warnings <- c(warnings, sprintf(
        "Image has %d channels. This is unusual - verify dimension ordering is correct.
Expected order: (height, width, channels) for images or (time, height, width, channels) for video.",
        n_channels
      ))
    }
  }

  return(list(errors = errors, warnings = warnings))
}


#' Check covariate matrix
#' @noRd
.check_covariate_matrix <- function(X, obsY_length = NULL) {
  errors <- character(0)
  warnings <- character(0)

  # Attempt to coerce to matrix if needed (following existing pattern)
  if (!"matrix" %in% class(X)) {
    X <- tryCatch({
      as.matrix(X)
    }, error = function(e) {
      errors <<- c(errors, sprintf(
        "Cannot coerce X to matrix: %s
FIX: Provide X as a numeric matrix or data.frame with numeric columns only.",
        e$message
      ))
      return(NULL)
    })
  }

  if (is.null(X)) return(list(errors = errors, warnings = warnings))

  # Check row count matches obsY
  if (!is.null(obsY_length) && nrow(X) != obsY_length) {
    errors <- c(errors, sprintf(
      "X has %d rows but obsY has %d observations. Rows must match.
FIX: Ensure X has one row per observation in obsY.",
      nrow(X), obsY_length
    ))
  }

  # Check for NAs (following existing pattern from CI_Confounding.R)
  if (any(is.na(X))) {
    na_counts <- colSums(is.na(X))
    na_cols <- which(na_counts > 0)
    na_col_names <- if (!is.null(colnames(X))) colnames(X)[na_cols] else as.character(na_cols)
    errors <- c(errors, sprintf(
      "X contains NA values. is.na(sum(X)) is TRUE.
Columns with NAs: %s
FIX: Impute missing values or remove observations with NAs before analysis.",
      paste(head(na_col_names, 5), collapse = ", ")
    ))
  }

  # Check for zero-variance columns (following existing pattern)
  col_sds <- apply(X, 2, sd, na.rm = TRUE)
  zero_var_cols <- which(col_sds == 0)
  if (length(zero_var_cols) > 0) {
    col_names <- if (!is.null(colnames(X))) colnames(X)[zero_var_cols] else as.character(zero_var_cols)
    errors <- c(errors, sprintf(
      "X has %d column(s) with zero variance. any(apply(X,2,sd) == 0) is TRUE.
Constant columns: %s
FIX: Remove constant columns from X before analysis.",
      length(zero_var_cols), paste(head(col_names, 5), collapse = ", ")
    ))
  }

  # Check for non-numeric content
  if (!is.numeric(X)) {
    errors <- c(errors,
      "X must contain only numeric values. Check for factor or character columns.
FIX: Convert all columns to numeric or remove non-numeric columns."
    )
  }

  return(list(errors = errors, warnings = warnings))
}


#' Check TfRecord file
#' @noRd
.check_tfrecord_file <- function(file, checkContext) {
  errors <- character(0)
  warnings <- character(0)

  if (is.null(file)) {
    if (checkContext == "pre_analysis") {
      errors <- c(errors,
        "No file specified for tfrecord.
FIX: Provide path to a .tfrecord file created by WriteTfRecord()."
      )
    }
    return(list(errors = errors, warnings = warnings))
  }

  # For pre_write, file should NOT exist (or we warn about overwrite)
  if (checkContext == "pre_write") {
    if (file.exists(file)) {
      warnings <- c(warnings, sprintf(
        "TfRecord file already exists at '%s'. It will be overwritten.
Use a different filename if you want to preserve existing data.",
        file
      ))
    }

    # Check parent directory exists
    parent_dir <- dirname(file)
    if (parent_dir != "." && parent_dir != "" && !dir.exists(parent_dir)) {
      errors <- c(errors, sprintf(
        "Parent directory does not exist: '%s'
FIX: Create directory with dir.create('%s', recursive = TRUE) before writing.",
        parent_dir, parent_dir
      ))
    }
  }

  # For pre_analysis, file MUST exist and be readable
  if (checkContext == "pre_analysis") {
    if (!file.exists(file)) {
      errors <- c(errors, sprintf(
        "TfRecord file not found: '%s'
FIX: Check file path or create tfrecord with WriteTfRecord() first.",
        file
      ))
    } else {
      # Check file is readable
      if (file.access(file, mode = 4) != 0) {
        errors <- c(errors, sprintf(
          "TfRecord file exists but is not readable: '%s'
FIX: Check file permissions.",
          file
        ))
      }

      # Check file size is reasonable
      file_size <- file.info(file)$size
      if (file_size < 100) {
        warnings <- c(warnings, sprintf(
          "TfRecord file is very small (%d bytes). May be empty or corrupt.",
          file_size
        ))
      }
    }
  }

  return(list(errors = errors, warnings = warnings))
}


#' Check parameter ranges
#' @noRd
.check_parameter_ranges <- function(batchSize, learningRateMax, nSGD, testFrac, obsY_length = NULL) {
  errors <- character(0)
  warnings <- character(0)

  # batchSize validation
  if (!is.null(batchSize)) {
    if (!is.numeric(batchSize) || batchSize < 1) {
      errors <- c(errors, sprintf(
        "batchSize must be a positive integer, got '%s'.
FIX: Set batchSize to a positive integer (e.g., 16, 32, 64).",
        as.character(batchSize)
      ))
    } else {
      if (!is.null(obsY_length) && batchSize > obsY_length) {
        warnings <- c(warnings, sprintf(
          "batchSize (%d) exceeds number of observations (%d).
Will be automatically reduced to fit data.",
          as.integer(batchSize), obsY_length
        ))
      }
      if (batchSize > 256) {
        warnings <- c(warnings, sprintf(
          "batchSize = %d is very large. May cause memory issues.
Consider 16-64 for most use cases.",
          as.integer(batchSize)
        ))
      }
    }
  }

  # learningRateMax validation
  if (!is.null(learningRateMax)) {
    if (!is.numeric(learningRateMax) || learningRateMax <= 0) {
      errors <- c(errors, sprintf(
        "learningRateMax must be a positive number, got '%s'.
FIX: Set learningRateMax to a small positive value (e.g., 0.001, 0.0001).",
        as.character(learningRateMax)
      ))
    } else if (learningRateMax > 1) {
      warnings <- c(warnings, sprintf(
        "learningRateMax = %g is very high. Typical values are 0.0001 to 0.01.
High learning rates may cause training instability.",
        learningRateMax
      ))
    }
  }

  # nSGD validation
  if (!is.null(nSGD)) {
    if (!is.numeric(nSGD) || nSGD < 1) {
      errors <- c(errors, sprintf(
        "nSGD must be a positive integer, got '%s'.
FIX: Set nSGD to number of training iterations (e.g., 100, 500, 1000).",
        as.character(nSGD)
      ))
    } else if (nSGD < 50) {
      warnings <- c(warnings, sprintf(
        "nSGD = %d is low. Model may underfit.
Consider at least 100-500 iterations for adequate training.",
        as.integer(nSGD)
      ))
    }
  }

  # testFrac validation
  if (!is.null(testFrac)) {
    if (!is.numeric(testFrac) || testFrac <= 0 || testFrac >= 1) {
      errors <- c(errors, sprintf(
        "testFrac must be between 0 and 1 (exclusive), got '%s'.
FIX: Set testFrac to fraction of data for testing (e.g., 0.1, 0.2).",
        as.character(testFrac)
      ))
    } else if (testFrac > 0.5) {
      warnings <- c(warnings, sprintf(
        "testFrac = %g uses more than half the data for testing.
Consider 0.1-0.2 for adequate training data.",
        testFrac
      ))
    }
  }

  return(list(errors = errors, warnings = warnings))
}


# =============================================================================
# Internal Wrapper for Integration into Main Functions
# =============================================================================

#' Internal validation wrapper (not exported)
#' @noRd
.validate_inputs <- function(..., context, quiet = FALSE) {
  result <- CheckDataQuality(..., checkContext = context, stopOnError = TRUE, quiet = quiet)
  invisible(result)
}
