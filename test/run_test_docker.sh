#!/usr/bin/env bash

# =============================================================================
# Docker API Test Runner
# =============================================================================
# This script:
# 1. Cleans up test_output/docker_call folder
# 2. Verifies Docker API is accessible
# 3. Runs test_docker.py to process all images via Docker API
# 4. Saves api_response.json for each image
#
# Note: Assumes Docker container is already running at the specified API URL
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
OUTPUT_DIR="${PROJECT_ROOT}/test/test_output/docker_call"
API_URL="${API_URL:-http://localhost:8000}"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}Docker API Test Runner${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "  API URL: ${API_URL}"
echo -e "  Output: ${OUTPUT_DIR}"
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

# Step 2: Verify API is accessible
echo -e "${YELLOW}Step 2: Verifying Docker API is accessible...${NC}"
if curl -s "${API_URL}/health" > /dev/null 2>&1; then
    health_response=$(curl -s "${API_URL}/health")
    echo -e "${GREEN}✓ Docker API is accessible and responding${NC}"
    echo -e "  Health status: ${health_response}"
    echo ""
else
    echo -e "${RED}❌ Docker API is not accessible at ${API_URL}${NC}"
    echo -e "${RED}   Please ensure the Docker container is running:${NC}"
    echo -e "${RED}   docker run -d -p 8000:8000 prodatalytics/inventory-tracker-api:latest${NC}"
    exit 1
fi

# Step 3: Run Docker API tests
echo -e "${YELLOW}Step 3: Running Docker API tests...${NC}"
echo -e "  Test script: ${PROJECT_ROOT}/test/test_docker.py"
echo -e "  Output: ${OUTPUT_DIR}"
echo -e "  API URL: ${API_URL}"
echo ""

cd "${PROJECT_ROOT}"

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
else
    echo -e "${RED}❌ Python not found${NC}"
    exit 1
fi

# Run tests
"${PYTHON_BIN}" "${PROJECT_ROOT}/test/test_docker.py" --api-url "${API_URL}"

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}=============================================================================${NC}"
    echo -e "${GREEN}Docker API tests completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: ${OUTPUT_DIR}${NC}"
    echo -e "${GREEN}  Each test image has its own subdirectory with api_response.json${NC}"
    echo -e "${GREEN}  Example: ${OUTPUT_DIR}/20251010-1/api_response.json${NC}"
    echo -e "${GREEN}=============================================================================${NC}"
else
    echo -e "${RED}=============================================================================${NC}"
    echo -e "${RED}Docker API tests failed with exit code: ${TEST_EXIT_CODE}${NC}"
    echo -e "${RED}=============================================================================${NC}"
fi

exit $TEST_EXIT_CODE
