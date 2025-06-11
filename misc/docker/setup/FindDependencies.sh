
docker run --platform=linux/amd64 --rm \
  -v "$(pwd)/binaries:/binaries" \
  rocker/r-ver:4.4.0 bash -exc "
    set -euo pipefail

    echo \"🔧 Installing system development libraries...\"
    apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential libcurl4-openssl-dev libssl-dev libxml2-dev \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

    echo \"🛠️  Installing remotes from CRAN...\"
    Rscript -e \"options(repos = c(CRAN = 'https://cloud.r-project.org')); \
                 install.packages('remotes', dependencies = TRUE)\"

    echo \"📂 Preparing build directory...\"
    mkdir -p /binaries/bin
    cd /binaries/bin

    echo \"📦 Building causalimages from GitHub...\"
    Rscript -e \"remotes::install_github('cjerzak/causalimages-software', \
                                             subdir='causalimages', \
                                             dependencies=FALSE)\"

    echo \"⚙️  Building causalimages binary...\"
    R CMD INSTALL --build causalimages_*.tar.gz

    echo \"✅ Built binary tarballs:\"
    ls -1 *.tar.gz
  "
