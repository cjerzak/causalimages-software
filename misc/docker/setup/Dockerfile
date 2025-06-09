


docker run --platform=linux/amd64 --rm   -v "$(pwd)/binaries:/binaries"   rocker/r-ver:4.4.0 bash -exc "
    set -euo pipefail

    echo \"🔧 Installing system development libraries...\"
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential libcurl4-openssl-dev libssl-dev libxml2-dev \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    # Clean previous artifacts and prepare directories
    rm -rf /binaries/* && mkdir -p /binaries/src /binaries/bin

    # List of CRAN packages in dependency-safe order
    pkgs=( \
      remotes Rcpp rlang glue cli lifecycle withr vctrs utf8 pkgconfig pillar \
      magrittr fansi fastmatch iterators foreach RcppTOML tidyselect tibble \
      R6 generics stringi RcppEigen shape RcppArmadillo RcppParallel \
      stringr glmnet data.table R.methodsS3 R.oo R.utils plyr \
      stringdist latex2exp \
      viridisLite ggplot2 gtable isoband scales gridExtra viridis \ 
      geosphere raster animation rrapply \
      sf grf pROC devtools tensorflow keras \
    )

    # Download CRAN package sources
    echo \"📥 Downloading CRAN package sources...\"
    cd /binaries/src
    for pkg in \"\${pkgs[@]}\"; do
      echo \"📦 Downloading \$pkg source...\"
      Rscript -e \"download.packages('\$pkg', destdir='/binaries/src', type='source', repos='https://cloud.r-project.org')\"
    done

    # Build CRAN package binaries in correct dependency order
    echo \"🛠️  Building CRAN package binaries...\"
    cd /binaries/bin
    for pkg in \"\${pkgs[@]}\"; do
      src_tar=(/binaries/src/\${pkg}_*.tar.gz)
      if [[ -f \"\${src_tar[0]}\" ]]; then
        echo \"⚙️  Building \$pkg from \$(basename \"\${src_tar[0]}\")...\"
        R CMD INSTALL --build \"\${src_tar[0]}\"
      else
        echo \"⚠️  Source for \$pkg not found, skipping.\"
        exit 1    # <— stop the entire script here
      fi
    done

    # Build GitHub package binary
    cd /binaries/bin
    echo \"📦 Building causalimages from GitHub...\"
    Rscript -e \"remotes::build_github('cjerzak/causalimages-software', subdir='causalimages', dependencies=FALSE)\"
    echo \"⚙️  Building causalimages binary...\"
    R CMD INSTALL --build causalimages_*.tar.gz

    # Move built binaries to host-mounted /binaries root
    mv /binaries/bin/*.tar.gz /binaries/

    echo \"✅ Built binary tarballs:\"
  "
















!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!
!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!!~!~!~!~!








docker run --platform=linux/amd64 --rm   -v "$(pwd)/binaries:/binaries"   rocker/r-ver:4.4.0 bash -exc "
    set -euo pipefail

    echo \"🔧 Installing system development libraries...\"
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential libcurl4-openssl-dev libssl-dev libxml2-dev \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    # Clean previous artifacts and prepare directories
    rm -rf /binaries/* && mkdir -p /binaries/src /binaries/bin

    # List of CRAN packages in dependency-safe order
    pkgs=( \
      remotes Rcpp rlang glue cli lifecycle withr vctrs utf8 pkgconfig pillar \
      magrittr fansi fastmatch iterators foreach RcppTOML tidyselect tibble \
      R6 generics stringi RcppEigen shape RcppArmadillo RcppParallel \
      stringr glmnet data.table R.methodsS3 R.utils plyr \
      stringdist latex2exp viridis geosphere raster animation rrapply \docker run --platform=linux/amd64 --rm   -v "$(pwd)/binaries:/binaries"   rocker/r-ver:4.4.0 bash -exc "
    set -euo pipefail

    echo \"🔧 Installing system development libraries...\"
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential libcurl4-openssl-dev libssl-dev libxml2-dev \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    # Clean previous artifacts and prepare directories
    rm -rf /binaries/* && mkdir -p /binaries/src /binaries/bin

    # List of CRAN packages in dependency-safe order
    pkgs=( \
      remotes Rcpp rlang glue cli lifecycle withr vctrs utf8 pkgconfig pillar \
      magrittr fansi fastmatch iterators foreach RcppTOML tidyselect tibble \
      R6 generics stringi RcppEigen shape RcppArmadillo RcppParallel \
      stringr glmnet data.table R.methodsS3 R.oo R.utils plyr \
      stringdist latex2exp \ 
      viridisLite ggplot2 gridExtr viridis \
      geosphere raster animation rrapply \
      sf grf pROC devtools tensorflow keras \
    )

    # Download CRAN package sources
    echo \"📥 Downloading CRAN package sources...\"
    cd /binaries/src
    for pkg in \"\${pkgs[@]}\"; do
      echo \"📦 Downloading \$pkg source...\"
      Rscript -e \"download.packages('\$pkg', destdir='/binaries/src', type='source', repos='https://cloud.r-project.org')\"
    done

    # Build CRAN package binaries in correct dependency order
    echo \"🛠️  Building CRAN package binaries...\"
    cd /binaries/bin
    for pkg in \"\${pkgs[@]}\"; do
      src_tar=(/binaries/src/\${pkg}_*.tar.gz)
      if [[ -f \"\${src_tar[0]}\" ]]; then
        echo \"⚙️  Building \$pkg from \$(basename \"\${src_tar[0]}\")...\"
        R CMD INSTALL --build \"\${src_tar[0]}\"
      else
        echo \"⚠️  Source for \$pkg not found, skipping.\"
        exit 1    # <— stop the entire script here
      fi
    done

    # Build GitHub package binary
    cd /binaries/bin
    echo \"📦 Building causalimages from GitHub...\"
    Rscript -e \"remotes::build_github('cjerzak/causalimages-software', subdir='causalimages', dependencies=FALSE)\"
    echo \"⚙️  Building causalimages binary...\"
    R CMD INSTALL --build causalimages_*.tar.gz

    # Move built binaries to host-mounted /binaries root
    mv /binaries/bin/*.tar.gz /binaries/

    echo \"✅ Built binary tarballs:\"
    ls -1 /binaries
  "
      sf grf pROC devtools tensorflow keras \
    )

    # Download CRAN package sources
    echo \"📥 Downloading CRAN package sources...\"
    cd /binaries/src
    for pkg in \"\${pkgs[@]}\"; do
      echo \"📦 Downloading \$pkg source...\"
      Rscript -e \"download.packages('\$pkg', destdir='/binaries/src', type='source', repos='https://cloud.r-project.org')\"
    done

    # Build CRAN package binaries in correct dependency order
    echo \"🛠️  Building CRAN package binaries...\"
    cd /binaries/bin
    for pkg in \"\${pkgs[@]}\"; do
      src_tar=(/binaries/src/\${pkg}_*.tar.gz)
      if [[ -f \"\${src_tar[0]}\" ]]; then
        echo \"⚙️  Building \$pkg from \$(basename \"\${src_tar[0]}\")...\"
        R CMD INSTALL --build \"\${src_tar[0]}\"
      else
        echo \"⚠️  Source for \$pkg not found, skipping.\"
        exit 1    # <— stop the entire script here
      fi
    done

    # Build GitHub package binary
    cd /binaries/bin
    echo \"📦 Building causalimages from GitHub...\"
    Rscript -e \"remotes::build_github('cjerzak/causalimages-software', subdir='causalimages', dependencies=FALSE)\"
    echo \"⚙️  Building causalimages binary...\"
    R CMD INSTALL --build causalimages_*.tar.gz

    # Move built binaries to host-mounted /binaries root
    mv /binaries/bin/*.tar.gz /binaries/

    echo \"✅ Built binary tarballs:\"
    ls -1 /binaries
  "

















docker run --platform=linux/amd64 --rm \
  -v "$(pwd)/binaries:/binaries" \
  rocker/r-ver:4.4.0 bash -exc '
    set -euo pipefail

    echo "🔧 Installing system development libraries…"
    apt-get update -qq \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         build-essential libcurl4-openssl-dev libssl-dev libxml2-dev \
         libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

    # Clean previous artifacts and prepare directories
    rm -rf /binaries/*
    mkdir -p /binaries/src /binaries/bin

    # List of CRAN packages in dependency-safe order
    pkgs=(
      remotes lifecycle cli withr vctrs utf8 pkgconfig pillar magrittr fansi
      fastmatch iterators foreach rcpptoml tidyselect tibble r6 generics stringi
      rcppeigen shape rcpparmadillo rcppparallel fastmatch stringr glmnet data.table
      r.methodsS3 r.utils plyr stringdist latex2exp viridis geosphere raster animation
      rrapply sf grf pROC devtools tensorflow keras
    )

    # Download CRAN package sources
    echo "📥 Downloading CRAN package sources…"
    cd /binaries/src
    for pkg in "${pkgs[@]}"; do
      echo "📦 Downloading $pkg source…"
      Rscript -e "download.packages('$pkg', destdir='/binaries/src', type='source', repos='https://cloud.r-project.org')"
    done

    # Build CRAN package binaries in correct dependency order
    echo "🛠️  Building CRAN package binaries…"
    cd /binaries/bin
    for pkg in "${pkgs[@]}"; do
      src_tar=(/binaries/src/${pkg}_*.tar.gz)
      if [[ -f "${src_tar[0]}" ]]; then
        echo "⚙️  Building $pkg from $(basename "${src_tar[0]}")…"
        R CMD INSTALL --build "${src_tar[0]}"
      else
        echo "⚠️  Source for $pkg not found, skipping."
      fi
    done

    # Build GitHub package binary
    cd /binaries/bin
    echo "📦 Building causalimages from GitHub…"
    Rscript -e "remotes::build_github('cjerzak/causalimages-software', subdir='causalimages', dependencies=FALSE)"
    echo "⚙️  Building causalimages binary…"
    R CMD INSTALL --build causalimages_*.tar.gz

    # Move built binaries to host-mounted /binaries root
    mv /binaries/bin/*.tar.gz /binaries/

    echo "✅ Built binary tarballs:"
    ls -1 /binaries
  '



docker run --platform=linux/amd64 --rm \
  -v "$(pwd)/binaries:/binaries" \
  rocker/r-ver:4.4.0 bash -exc "
    set -euo pipefail

    echo '🔧 Installing system development libraries…'
    apt-get update -qq \
      && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         build-essential libcurl4-openssl-dev libssl-dev libxml2-dev \
         libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates \
      && apt-get clean \
      && rm -rf /var/lib/apt/lists/*

    # Clean previous artifacts and prepare directories
    rm -rf /binaries/*
    mkdir -p /binaries/src /binaries/bin

    # Download CRAN package sources
    cd /binaries/src
    for pkg in \
      remotes lifecycle cli withr vctrs utf8 pkgconfig pillar magrittr fansi \
      iterators foreach rcpptoml tidyselect tibble r6 generics stringi rcppeigen \
      shape rcpparmadillo rcppparallel fastmatch stringr glmnet data.table \
      r.methodsS3 r.utils plyr stringdist tensorflow keras latex2exp viridis \
      geosphere raster rrapply sf grf pROC; do
      echo \"📦 Downloading \$pkg source…\"
      Rscript -e \"download.packages('\$pkg', destdir='/binaries/src', type='source', repos='https://cloud.r-project.org')\"
    done

    # Build CRAN package binaries
    cd /binaries/bin
    for src in /binaries/src/*.tar.gz; do
      echo \"⚙️  Building \$(basename \$src) binary…\"
      R CMD INSTALL --build \"\$src\"
    done

    # Build GitHub package binary
    cd /binaries/bin
    echo '📦 Building causalimages from GitHub…'
    Rscript -e \"remotes::build_github('cjerzak/causalimages-software', subdir='causalimages', dependencies=FALSE)\"
    echo '⚙️  Building causalimages binary…'
    R CMD INSTALL --build causalimages_*.tar.gz

    # Move built binaries to host-mounted /binaries root
    mv /binaries/bin/*.tar.gz /binaries/

    echo '✅ Built binary tarballs:'
    ls -1 /binaries
  "




devtools
animation











# v0 

docker run --platform=linux/amd64 --rm \
  -v "$(pwd)/binaries:/binaries" \
  rocker/r-ver:4.4.0 bash -exc "
    set -euo pipefail

    echo '🔧 Installing system development libraries…'
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      libcurl4-openssl-dev \
      libssl-dev \
      libxml2-dev \
      libgdal-dev \
      libgeos-dev \
      libproj-dev \
      libudunits2-dev \
      ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

    cd /binaries

    ## ----------------------------------------------------------------------------
    ##  Build all CRAN packages used in your Dockerfile
    ## ----------------------------------------------------------------------------
    for pkg in \
      remotes lifecycle cli withr vctrs utf8 pkgconfig pillar magrittr fansi \
      iterators foreach rcpptoml tidyselect tibble r6 generics stringi rcppeigen \
      shape rcpparmadillo rcppparallel fastmatch stringr glmnet data.table \
      r.methodsS3 r.utils plyr stringdist tensorflow keras latex2exp viridis \
      geosphere raster animation rrapply sf grf pROC devtools; \
    do
      echo \"📦 Downloading $pkg source…\"
      Rscript -e 'download.packages(
        \"'\"\$pkg\"'\", 
        destdir      = \"/binaries\",
        type         = \"source\",
        repos        = \"https://cloud.r-project.org\" 
      )'
      echo \"⚙️  Building $pkg binary…\"
      #R CMD INSTALL --build /binaries/\${pkg}_*.tar.gz
      Rscript -e \"download.packages(\"${pkg}\", destdir='/binaries/src')\"
      R CMD INSTALL --build /binaries/src/${pkg}_*.tar.gz --library=/binaries/bin
    done

    ## ----------------------------------------------------------------------------
    ##  Build your GitHub package
    ## ----------------------------------------------------------------------------
    echo '📦 Building causalimages from GitHub…'
    Rscript -e 'remotes::build_github(
      \"cjerzak/causalimages-software\", 
      subdir       = \"causalimages\", 
      dependencies = FALSE 
    )'
    echo '⚙️  Building causalimages binary…'
    R CMD INSTALL --build /binaries/causalimages_*.tar.gz

    echo '✅ Built binary tarballs:'
    ls -1 /binaries
  "







FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install tools to add new repos & the CRAN signing key
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      software-properties-common dirmngr gnupg apt-transport-https ca-certificates \
 && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
 && add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/' \
 && apt-get update

# 2. System libs + the R packages that *are* actually on CRAN’s APT repo
RUN apt-get install -y --no-install-recommends \
      r-base r-base-dev \
      libcurl4-openssl-dev libssl-dev libxml2-dev libgdal-dev libgeos-dev libproj-dev \
      r-cran-reticulate r-cran-rcpp r-cran-rlang r-cran-glue r-cran-jsonlite r-cran-dplyr \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Install the rest via R’s install.packages()
RUN Rscript -e "install.packages( c('remotes','lifecycle','cli','withr','vctrs','utf8','pkgconfig','pillar','magrittr','fansi','iterators','foreach','rcpptoml','tidyselect','tibble','r6','generics','stringi','rcppeigen','shape','rcpparmadillo','rcppparallel','fastmatch','stringr','glmnet','data.table','r.methodsS3','r.utils','plyr','stringdist','tensorflow','keras','latex2exp','viridis','geosphere','raster','animation','rrapply','sf','grf','pROC','devtools'),repos='https://cloud.r-project.org', dependencies=TRUE)"


# 6. Install your GitHub 'causalimages' package
RUN Rscript -e "remotes::install_github('cjerzak/causalimages-software', subdir='causalimages',dependencies=FALSE, build_vignettes=FALSE)"

# 7. Pre-build the Conda-backed JAX/numpy backend -> do this in package 
# RUN Rscript -e "library(causalimages); causalimages::BuildBackend(conda_env='CausalImagesEnv', conda='auto')"

# Final message
CMD echo; echo 'Setup complete! To use:'; echo '  docker run -it causalimages-env bash'
