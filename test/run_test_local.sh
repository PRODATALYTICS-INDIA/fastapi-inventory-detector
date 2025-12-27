#!/usr/bin/env bash

# =============================================================================
# Local Test Runner
# =============================================================================
# This script:
# 1. Cleans up test_output/local_call folder
# 2. Runs test_local.py to process all images locally
# =============================================================================

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Resolve project root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/test/test_output/local_call"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Local Test Runner${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Step 1: Clean up output folder
echo -e "${YELLOW}Step 1: Cleaning up output folder...${NC}"
if [ -d "${OUTPUT_DIR}" ]; then
    echo -e "  Removing: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"/*
    echo -e "${GREEN}✓ Output folder cleaned${NC}"
else
    echo -e "  Creating: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    echo -e "${GREEN}✓ Output folder created${NC}"
fi
echo ""

# Step 2: Check if virtual environment exists
PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo -e "${YELLOW}Virtual environment not found. Setting up...${NC}"
    if [ -f "${PROJECT_ROOT}/setup_environment.sh" ]; then
        sh "${PROJECT_ROOT}/setup_environment.sh"
    else
        echo -e "${RED}❌ setup_environment.sh not found${NC}"
        exit 1
    fi
fi

# Step 3: Run local tests
echo -e "${YELLOW}Step 2: Running local tests...${NC}"
echo -e "  Python: ${PYTHON_BIN}"
echo -e "  Test script: ${PROJECT_ROOT}/test/test_local.py"
echo -e "  Output: ${OUTPUT_DIR}"
echo ""

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/test/test_local.py" --confidence 0.5

echo ""
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}Local tests completed!${NC}"
echo -e "${GREEN}Results saved to: ${OUTPUT_DIR}${NC}"
echo -e "${GREEN}=============================================================================${NC}"

