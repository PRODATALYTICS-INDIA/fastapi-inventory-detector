#!/bin/bash

# =============================================================================
# Environment Setup Script
# Automatically sets up Python virtual environment and installs dependencies
# Uses uv (fast Python package manager) if available, falls back to standard venv/pip
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FastAPI Inventory Detector${NC}"
echo -e "${BLUE}Environment Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: requirements.txt not found at $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if uv is available
if command_exists uv; then
    echo -e "${GREEN}✓ uv is available - using uv for faster setup${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}⚠ uv not found - falling back to standard venv/pip${NC}"
    echo -e "${YELLOW}  Install uv for faster setup: curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
    USE_UV=false
fi

# Remove existing venv if it exists and user wants to recreate
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${GREEN}Using existing virtual environment${NC}"
    fi
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    
    if [ "$USE_UV" = true ]; then
        # Use uv to create venv (faster)
        uv venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created with uv${NC}"
    else
        # Use standard Python venv
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created with Python venv${NC}"
    fi
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip if using standard venv (uv manages this automatically)
if [ "$USE_UV" = false ]; then
    echo -e "${BLUE}Upgrading pip...${NC}"
    pip install --upgrade pip setuptools wheel
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies from requirements.txt...${NC}"

if [ "$USE_UV" = true ]; then
    # Use uv to install dependencies (much faster)
    uv pip install -r "$REQUIREMENTS_FILE"
    echo -e "${GREEN}✓ Dependencies installed with uv${NC}"
else
    # Use standard pip
    pip install -r "$REQUIREMENTS_FILE"
    echo -e "${GREEN}✓ Dependencies installed with pip${NC}"
fi

# Verify installation
echo ""
echo -e "${BLUE}Verifying installation...${NC}"
python --version
pip --version

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Environment setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}To activate the environment manually, run:${NC}"
echo -e "${YELLOW}  source venv/bin/activate${NC}"
echo ""
echo -e "${BLUE}On Windows (PowerShell):${NC}"
echo -e "${YELLOW}  .\\venv\\Scripts\\Activate.ps1${NC}"
echo ""
echo -e "${BLUE}On Windows (CMD):${NC}"
echo -e "${YELLOW}  venv\\Scripts\\activate.bat${NC}"
echo ""
echo -e "${BLUE}To deactivate, run:${NC}"
echo -e "${YELLOW}  deactivate${NC}"
echo ""
