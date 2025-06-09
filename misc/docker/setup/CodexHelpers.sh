#!/usr/bin/env bash

set -euo pipefail

#------------------------------------------------------------------------------
# 1. Install R and unzip (for unpacking the bundle)
#------------------------------------------------------------------------------
apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       r-base \
       r-base-dev \
       libcurl4-openssl-dev \
       libssl-dev \
       libxml2-dev \
       unzip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#------------------------------------------------------------------------------
# 2. Install Miniconda into /opt/conda
#------------------------------------------------------------------------------
readonly MINICONDA_SH="/tmp/Miniconda3-latest-Linux-x86_64.sh"
wget --quiet "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "$MINICONDA_SH"
bash "$MINICONDA_SH" -b -p /opt/conda
rm "$MINICONDA_SH"

# Make sure conda is on PATH
export PATH="/opt/conda/bin:${PATH}"

# Initialize conda for this shell
if [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "Error: /opt/conda/etc/profile.d/conda.sh not found" >&2
  exit 1
fi

#------------------------------------------------------------------------------
# 3. Fetch and unpack the ZIP-of-ZIPs
#------------------------------------------------------------------------------
BIN_ROOT="https://raw.githubusercontent.com/cjerzak/causalimages-software/main/misc/binaries"
BUNDLE_ZIP="binaries.zip"

# download the bundle
curl -sSL -o "/tmp/${BUNDLE_ZIP}" "${BIN_ROOT}/${BUNDLE_ZIP}"

# unpack into a temp directory
mkdir -p /tmp/binaries
unzip -q "/tmp/${BUNDLE_ZIP}" -d /tmp/binaries

#------------------------------------------------------------------------------
# 4. Install each inner ZIP into R's library path
#------------------------------------------------------------------------------
# find the first (user) library path and make sure it exists
LIB="$(Rscript -e 'cat(.libPaths()[1])')"
mkdir -p "$LIB"

# iterate through each ZIP and unzip it directly into $LIB
for z in /tmp/binaries/*.zip; do
  unzip -q "$z" -d "$LIB"
done

# clean up temporary files
rm -rf /tmp/${BUNDLE_ZIP} /tmp/binaries

#------------------------------------------------------------------------------
# 5. Install the rest of your R packages (from LOCAL_TAR) and build backend
#------------------------------------------------------------------------------
# assumes LOCAL_TAR is set in your environment
tar --strip-components=1 -xpf "${LOCAL_TAR}" -C /opt

Rscript -e "library(causalimages); causalimages::BuildBackend(conda_env = 'CausalImagesEnv', conda = 'auto')"

echo "Success: Done with download & unpacking script!"
