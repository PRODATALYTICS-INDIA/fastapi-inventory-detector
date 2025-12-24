#!/bin/bash

# =============================================================================
# FastAPI Inventory Detector - Startup Script
# =============================================================================
# This script activates the virtual environment and starts the FastAPI service
# Usage: ./start_fastapi.sh
# =============================================================================

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_PATH="venv/bin/activate"
HOST="0.0.0.0"
PORT="8000"

echo -e "${BLUE}=============================================================================${NC}"
echo -e "${BLUE}FastAPI Inventory Detector - Starting Service${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -f "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠️  Warning: Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}   Setting up environment...${NC}"
    echo ""
    
    # Check if setup_environment.sh exists and run it
    if [ -f "setup_environment.sh" ]; then
        echo -e "${BLUE}Running setup_environment.sh...${NC}"
        bash setup_environment.sh
        echo ""
        
        # Verify venv was created
        if [ ! -f "$VENV_PATH" ]; then
            echo -e "${YELLOW}⚠️  Virtual environment setup may have failed${NC}"
            echo -e "${YELLOW}   Please run setup_environment.sh manually: ./setup_environment.sh${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}   Please run setup_environment.sh first, or create venv manually:${NC}"
        echo -e "${YELLOW}   ./setup_environment.sh${NC}"
        echo -e "${YELLOW}   Or: python3 -m venv venv${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${GREEN}✓ Activating virtual environment...${NC}"
source "$VENV_PATH"

# Verify Python and required packages
echo -e "${GREEN}✓ Checking Python environment...${NC}"
python --version

# Check if required files exist
echo -e "${GREEN}✓ Checking required files...${NC}"
if [ ! -f "app/data/sku_master.json" ]; then
    echo -e "${YELLOW}⚠️  Warning: app/data/sku_master.json not found${NC}"
    echo -e "${YELLOW}   The service may not work correctly without the SKU master file${NC}"
else
    echo -e "${GREEN}✓ SKU master file found${NC}"
fi

# Print access information
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}🚀 FastAPI service is starting...${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo -e "${YELLOW}📍 Service Access Points:${NC}"
echo -e "   ${GREEN}API:${NC}         http://localhost:${PORT}"
echo -e "   ${GREEN}Docs:${NC}        http://localhost:${PORT}/docs"
echo -e "   ${GREEN}ReDoc:${NC}       http://localhost:${PORT}/redoc"
echo -e "   ${GREEN}Health:${NC}      http://localhost:${PORT}/health"
echo -e "   ${GREEN}Model Info:${NC}  http://localhost:${PORT}/model/info"
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

# Run FastAPI using uvicorn
# Using new v2.0 application structure (app.main:app)
# For legacy app, use: uvicorn main:app
uvicorn app.main:app --host "$HOST" --port "$PORT" --reload

