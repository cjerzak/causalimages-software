# Script: GetRDependencyOrder.R 
# Purpose: - Get all dependencies for set of base packages 
#          - Get safe dependency order for creation of compiled R packages 
{
# Your initial target packages
pkgs <- c("remotes", "tensorflow", "reticulate", "geosphere", "raster",
          "rrapply", "glmnet", "sf", "data.table", "pROC")

# Fetch the CRAN package database
cran_db <- available.packages(repos = "https://cloud.r-project.org")

# 1. Get the full recursive set of Depends + Imports
recursive_deps <- tools::package_dependencies(
  pkgs,
  db        = cran_db,
  which     = c("Depends", "Imports"),
  recursive = TRUE
)
all_pkgs <- unique(c(pkgs, unlist(recursive_deps)))

# Identify base/recommended packages to drop
base_pkgs <- rownames(installed.packages(priority = c("base", "recommended")))

# 2. Build a direct-dependency map for every package
direct_deps <- tools::package_dependencies(
  all_pkgs,
  db        = cran_db,
  which     = c("Depends", "Imports"),
  recursive = FALSE
)
# Remove any base packages from each dependency vector
direct_deps <- lapply(direct_deps, setdiff, y = base_pkgs)

# 3. Topological sort via DFS
ordered   <- character(0)  # will hold the final install order
visited   <- character(0)  # permanently marked nodes
visiting  <- character(0)  # temporary marks for cycle detection

visit <- function(pkg) {
  if (pkg %in% visited) return()
  if (pkg %in% visiting) {
    stop("Circular dependency detected involving: ", pkg)
  }
  visiting <<- c(visiting, pkg)
  for (dep in direct_deps[[pkg]]) {
    visit(dep)
  }
  visiting <<- setdiff(visiting, pkg)
  visited   <<- c(visited, pkg)
  ordered   <<- c(ordered, pkg)
}

# Run DFS on all packages
for (p in all_pkgs) {
  visit(p)
}

# Filter to just the ones you want (if you only want your original pkgs + all their deps)
dependency_safe_order <- intersect(ordered, all_pkgs)
dependency_safe_order <- dependency_safe_order[!dependency_safe_order %in% 
                                      rownames(installed.packages(priority = c("base")))]

# manual addins 
dependency_safe_order <- c(
            dependency_safe_order[1:which(dependency_safe_order=="Rcpp")],
            "RcppEigen",
            dependency_safe_order[(which(dependency_safe_order=="Rcpp")+1):length(dependency_safe_order)])

# Inspect
cat("Install in this order:\n")
cat(paste(dependency_safe_order, collapse = " "), "\n")
# must add in: RcppEigen
}





