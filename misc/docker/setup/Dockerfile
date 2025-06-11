
docker run --rm -it \
    --platform=linux/amd64 \
    -e CODEX_ENV_PYTHON_VERSION=3.12 \
    -e CODEX_ENV_NODE_VERSION=20 \
    -e CODEX_ENV_RUST_VERSION=1.87.0 \
    -e CODEX_ENV_GO_VERSION=1.23.8 \
    -e CODEX_ENV_SWIFT_VERSION=6.1 \
    -v "$HOME/Documents/causalimages-software/misc/docker/binaries:/binaries" \
    ghcr.io/openai/codex-universal:latest \
    -exc "
    set -euo pipefail

    echo \"🔧 Installing system development libraries...\"
    # Prepare apt for CRAN
    apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      software-properties-common dirmngr ca-certificates && \

    # Install R base + dev headers + all system libraries
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      r-base r-base-dev \
      build-essential cmake pkg-config git zip \
      libcurl4-openssl-dev libssl-dev libxml2-dev \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
      ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    # Clean previous artifacts and prepare directories
    rm -rf /binaries/* && mkdir -p /binaries/src /binaries/bin

    # List of CRAN packages in dependency-safe order
    pkgs=( \
    remotes yaml config ps R6 processx Rcpp RcppEigen RcppTOML rprojroot here jsonlite png rappdirs rlang withr reticulate base64enc magrittr whisker cli glue lifecycle vctrs tidyselect rstudioapi tfruns backports tfautograph tensorflow sp geosphere terra raster rrapply iterators foreach shape glmnet proxy e1071 classInt DBI wk s2 units sf data.table plyr pROC Matrix lattice survival codetools class KernSmooth MASS  \
    )

    echo \"🔧 Installing Miniconda...\"
    MINI=Miniconda3-latest-Linux-x86_64.sh
    wget -q https://repo.anaconda.com/miniconda/\${MINI} -O /tmp/\${MINI} && \
    bash /tmp/\${MINI} -b -p /opt/miniconda && \
    rm /tmp/\${MINI} && \
    export PATH=\"/opt/miniconda/bin:\$PATH\" && \
    conda config --set always_yes yes --set changeps1 no && \
    conda update -q conda


    # Download CRAN package sources
    echo \"📥 Downloading CRAN package sources...\"
    cd /binaries/src
    for pkg in \"\${pkgs[@]}\"; do
      echo \"📦 Downloading \$pkg source...\"
      Rscript -e \"download.packages('\$pkg', destdir='/binaries/src', type='source', repos='https://cloud.r-project.org')\" \
        || { echo \"❌ Failed to download \$pkg\"; exit 1; }
          # verify that a tar.gz actually appeared
      if ! compgen -G \"\${pkg}_*.tar.gz\" >/dev/null; then
        echo \"❌ No source tarball found for \$pkg after download, likely installed in base-R (recommended package)?.\" >&2
        # exit 1
        continue
      fi
    done

    echo \"✅ Downloaded all CRAN sources ✅\"

    # Build CRAN package binaries in correct dependency order
    echo \"🛠️  Building CRAN package binaries...\"
    cd /binaries/bin
    for pkg in \"\${pkgs[@]}\"; do
      src_tar=(/binaries/src/\${pkg}_*.tar.gz)
      if [[ -f \"\${src_tar[0]}\" ]]; then
        echo \"⚙ Building \$pkg from \$(basename \"\${src_tar[0]}\")...\"
        R CMD INSTALL --build \"\${src_tar[0]}\"
      else
        echo \"⚠ Source for \$pkg not found, skipping (likely found in base R install).\"
        #exit 1  # <— stop the entire script here
        continue
      fi
    done

    # Build GitHub package binary
    cd /binaries/src

    # clone *and* sparsify GitHub repo in one go
    git clone \
      --depth 1 \
      --filter=blob:none \
      --sparse \
      https://github.com/cjerzak/causalimages-software.git \
      /binaries/src/repo

    cd /binaries/src/repo

    # then tell Git which folder you actually want
    git sparse-checkout set causalimages

    # now build
    R CMD INSTALL --build ./causalimages
    mv causalimages_*.tar.gz /binaries/bin/causalimages.tar.gz

    echo \"✅✅ Built binary tarballs ✅✅\"

    echo \"🔧 Building conda backend...\"
    export RETICULATE_MINICONDA_PATH=/opt/miniconda
    Rscript -e \"
      library(causalimages);
      causalimages::BuildBackend(conda_env='CausalImagesEnv',
                                 conda='/opt/miniconda/bin/conda')
    \"

    echo \"📦 Zipping conda env...\"
    zip -r /binaries/CausalImagesEnv.zip /opt/miniconda/envs/CausalImagesEnv
     

    echo \"✅✅✅ Built binary tarballs ✅✅✅\"
  "


