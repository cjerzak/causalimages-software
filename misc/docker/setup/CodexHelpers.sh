#!/usr/bin/env bash

set -euo pipefail
# set -x

#------------------------------------------------------------------------------
# 1. Install R and unzip, other utilities 
#------------------------------------------------------------------------------
apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       r-base r-base-dev \
       libssl-dev libxml2-dev unzip \
       build-essential libcurl4-openssl-dev \
       libgdal-dev libgeos-dev libproj-dev libudunits2-dev ca-certificates \
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


# Define URLs for ZIP of tars and conda ZIP
THE_ZIP_URL="https://www.dl.dropboxusercontent.com/scl/fi/ek76700s9o4p4grwfw5ko/Archive.zip?rlkey=w0vj0qh78i3eb3c5zt4vlmot9&st=f8kxeq6t&dl=1"
readonly CONDA_ENV_ZIP_URL="https://www.dl.dropboxusercontent.com/scl/fi/k5vylxygjl4icm76drtsz/CausalImagesEnv.zip?rlkey=hmmwpma9bihoze25vee44dktz&st=0vz2vmbk&dl=1"

#------------------------------------------------------------------------------
# 3. Fetch and unpack the ZIP-of-tar's
#------------------------------------------------------------------------------
BUNDLE_ZIP="binaries.zip"

# download the bundle
curl -sSL -o "/tmp/${BUNDLE_ZIP}" "${THE_ZIP_URL}"

# unpack into a temp directory
mkdir -p /tmp/binaries
unzip -q "/tmp/${BUNDLE_ZIP}" -d /tmp/binaries

echo "Files extracted to /tmp/binaries:"
find /tmp/binaries -type f -printf "  ➜ %P\n"

#------------------------------------------------------------------------------
# 4. Install each inner ZIP into R's library path
#------------------------------------------------------------------------------
# find the first (user) library path and make sure it exists
LIB="$(Rscript -e 'cat(.libPaths()[1])')"
mkdir -p "$LIB"

#echo "Binary ZIPs found under /tmp/binaries:"
#for f in /tmp/binaries/*.tar.gz; do
#  echo "  ➜ $f"
#  tar -xzf "$f" -C "$LIB"
#done

echo "Installing binary packages into $LIB:"
 for f in /tmp/binaries/*.tar.gz; do
   pkg=$(basename "$f" | sed -E 's/_.*//')           # e.g. DBI_1.2.3... → DBI
   echo "  ➜ $pkg"
   mkdir -p "$LIB/$pkg"                              # ensure lib/pkg exists
   #tar --strip-components=1 -xzf "$f" -C "$LIB/$pkg"  # drop top‐level folder
   tar -xzf "$f" -C "$LIB"
   
   echo "    Contents of $LIB/$pkg:"
   find "$LIB/$pkg" -maxdepth 1 -mindepth 1 -printf "      ➜ %f\n"
done

# clean up temporary files
 rm -rf /tmp/${BUNDLE_ZIP} /tmp/binaries

echo "Installed R packages in $LIB:"
Rscript -e '
  #lib <- Sys.getenv("R_LIBS_USER", .libPaths()[1])
  #pkgs <- rownames(installed.packages(lib.loc = lib))
  pkgs <- rownames(installed.packages(lib.loc = .libPaths()))
  cat(paste0("Installed packages found ➜ ", pkgs, "\n"), sep = "")
'

echo "Installing causalimages backend..."
readonly CONDA_ENV_ZIP="/tmp/conda_env.zip"

echo "Downloading pre-built conda environment from $CONDA_ENV_ZIP_URL"
curl -sSL -o "$CONDA_ENV_ZIP" "$CONDA_ENV_ZIP_URL"

echo "Unpacking conda environment into /opt/conda/envs/"
unzip -q "$CONDA_ENV_ZIP" -d "/opt/conda/envs/"
rm "$CONDA_ENV_ZIP"

echo "Available conda environments:"
conda env list

echo "Success: Done with download & unpacking script! Ready to experiment."
