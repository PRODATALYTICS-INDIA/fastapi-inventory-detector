#!/usr/bin/env bash

# =============================================================================
# API Test Runner
# =============================================================================
# This script:
# 1. Cleans up test_output/api_call folder
# 2. Starts FastAPI server (app/main.py)
# 3. Runs test_api.py to process all images via API
# 4. Stops the FastAPI server after tests complete
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
OUTPUT_DIR="${PROJECT_ROOT}/test/test_output/api_call"
API_URL="${API_URL:-http://localhost:8000}"
SERVER_PORT="${SERVER_PORT:-8000}"

# Server process ID (will be set when server starts)
SERVER_PID=""

# Cleanup function to stop server on exit
cleanup() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        echo -e "${YELLOW}Stopping FastAPI server (PID: ${SERVER_PID})...${NC}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        echo -e "${GREEN}✓ Server stopped${NC}"
    fi
}

# Register cleanup function to run on script exit
trap cleanup EXIT INT TERM

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}API Test Runner${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "  API URL: ${API_URL}"
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

# Step 3: Check if server is already running
echo -e "${YELLOW}Step 2: Checking if server is already running...${NC}"
if curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Server is already running at ${API_URL}${NC}"
    echo ""
else
    # Step 4: Start FastAPI server
    echo -e "${YELLOW}Step 3: Starting FastAPI server...${NC}"
    echo -e "  Python: ${PYTHON_BIN}"
    echo -e "  App: ${PROJECT_ROOT}/app/main.py"
    echo -e "  Port: ${SERVER_PORT}"
    echo ""
    
    cd "${PROJECT_ROOT}"
    
    # Start server in background
    "${PYTHON_BIN}" -m uvicorn app.main:app --host 0.0.0.0 --port "${SERVER_PORT}" > /dev/null 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start (max 30 seconds)
    echo -e "  Waiting for server to start..."
    for i in {1..30}; do
        if curl -s "${API_URL}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server started successfully (PID: ${SERVER_PID})${NC}"
            echo ""
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Server failed to start within 30 seconds${NC}"
            exit 1
        fi
        sleep 1
    done
fi

# Step 5: Run API tests
echo -e "${YELLOW}Step 4: Running API tests...${NC}"
echo -e "  Python: ${PYTHON_BIN}"
echo -e "  Test script: ${PROJECT_ROOT}/test/test_api.py"
echo -e "  Output: ${OUTPUT_DIR}"
echo -e "  API URL: ${API_URL}"
echo ""

cd "${PROJECT_ROOT}"

# Run tests with --no-auto-start since we're managing the server ourselves
"${PYTHON_BIN}" "${PROJECT_ROOT}/test/test_api.py" --api-url "${API_URL}" --no-auto-start

echo ""
echo -e "${GREEN}=============================================================================${NC}"
echo -e "${GREEN}API tests completed!${NC}"
echo -e "${GREEN}Results saved to: ${OUTPUT_DIR}${NC}"
echo -e "${GREEN}=============================================================================${NC}"

