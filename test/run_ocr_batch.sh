#!/usr/bin/env bash

# Run detection → OCR → SKU prediction for all images in test/test_images
# For each image, this will:
# - call test_detection_ocr_pipeline.py
# - write crops + ocr_results.txt under test/test_output/<image_stem>/

set -euo pipefail

# Resolve project root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_DIR="${PROJECT_ROOT}/test/test_images"

if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "Image directory not found: ${IMAGE_DIR}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

# Ensure virtual environment and dependencies are set up.
# This script is interactive the first time (may ask to recreate venv).
echo "Setting up / activating Python environment with: sh setup_environment.sh"
sh "${PROJECT_ROOT}/setup_environment.sh"

# Use the venv Python explicitly so we know we're running inside it
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found at ${PYTHON_BIN}. Did setup_environment.sh succeed?" >&2
  exit 1
fi

echo "Running OCR pipeline for images in: ${IMAGE_DIR}" 

shopt -s nullglob
images=("${IMAGE_DIR}"/*.jpg "${IMAGE_DIR}"/*.jpeg "${IMAGE_DIR}"/*.png)

if [[ ${#images[@]} -eq 0 ]]; then
  echo "No images found in ${IMAGE_DIR} (jpg/jpeg/png)" >&2
  exit 0
fi

for img in "${images[@]}"; do
  echo "============================================================"
  echo "Processing: ${img}"
  echo "============================================================"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/test/test_detection_ocr_pipeline.py" --image "${img}"
  echo
done

echo "All images processed. Check test/test_output/<image_stem>/ for crops and ocr_results.txt"