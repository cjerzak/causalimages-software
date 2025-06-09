#!/usr/bin/env bash
# build_and_package.sh
# Run this from the same directory as your Dockerfile.

set -euo pipefail

IMAGE_NAME="causalimages-env"
TAR_NAME="${IMAGE_NAME}.tar"
ZIP_NAME="${IMAGE_NAME}.zip"

echo "1) Building Docker image..."
docker build -t ${IMAGE_NAME} .

echo "2) Saving image to tarball..."
docker save ${IMAGE_NAME} -o ${TAR_NAME}

echo "3) Zipping up Dockerfile + tarball..."
zip -r ${ZIP_NAME} Dockerfile ${TAR_NAME}

echo
echo "Done! Your package is: ${ZIP_NAME}"
echo "Push that ZIP to GitHub, then Codex can unzip and run:"
echo "  docker load -i ${TAR_NAME}"
echo "  docker run -it ${IMAGE_NAME} bash"
