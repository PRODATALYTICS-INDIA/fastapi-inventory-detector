#!/bin/bash

# Linux Server Connection and Management Script
# Usage: ./connect.sh [action] [options...]
#
# Actions:
#   connect    - Interactive SSH connection (default)
#   view       - View server contents and system info
#   cleanup    - Cleanup server resources (project/docker/all/system)
#   setup      - Setup server prerequisites (Docker, Python requests) - only if missing

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get action name for logging
ACTION="${1:-connect}"

# Setup logging directory and log file (skip for interactive connect action)
if [ "$ACTION" != "connect" ]; then
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/connect_${ACTION}_${TIMESTAMP}.log"
    
    # Function to log all output
    log_all() {
        tee -a "$LOG_FILE"
    }
    
    # Start logging
    echo "📝 Logging all output to: $LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Connection script started at: $(date)"
    echo "Action: ${ACTION}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Redirect all output to log file while still showing on screen
    exec > >(log_all)
    exec 2>&1
fi

# Server details
SERVER_IP="13.203.145.155"
USERNAME="ubuntu"
PEM_FILE="$SCRIPT_DIR/MI_dev_Linux_test_server.pem"
REMOTE_DIR="~/fastapi-inventory-detector"

# Check if PEM file exists
if [ ! -f "$PEM_FILE" ]; then
    echo "❌ Error: PEM file not found at '$PEM_FILE'!"
    echo ""
    echo "Please ensure MI_dev_Linux_test_server.pem is in the deployment directory."
    exit 1
fi

# Set proper permissions for PEM file
chmod 400 "$PEM_FILE" 2>/dev/null

case "$ACTION" in
    connect)
        # Interactive SSH connection
        echo "🔌 Connecting to Linux Test Server..."
        echo "   IP: $SERVER_IP"
        echo "   User: $USERNAME"
        echo "   PEM File: $PEM_FILE"
        echo ""
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP"
        ;;
    
    view)
        # View server configuration, folder structure, Docker images and containers
        REMOTE_DIR="${2:-~}"
        echo "🔍 Viewing Server Configuration and Status"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   ========================================="
        echo ""
        
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" bash << 'EOF'
            REMOTE_DIR="$REMOTE_DIR"
            # Expand ~ to home directory
            REMOTE_DIR=${REMOTE_DIR/#\~/$HOME}
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "💻 SERVER CONFIGURATION"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Hostname:     $(hostname)"
            echo "IP Address:   $(hostname -I 2>/dev/null | awk '{print $1}' || echo 'N/A')"
            echo "OS:           $(lsb_release -d 2>/dev/null | cut -f2 || uname -s)"
            echo "Kernel:       $(uname -r)"
            echo "Uptime:       $(uptime -p 2>/dev/null || uptime | awk '{print $3,$4}' | sed 's/,//')"
            echo ""
            
            echo "💾 STORAGE (Disk Usage)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            if command -v df &> /dev/null; then
                DISK_INFO=$(df -h / 2>&1 | tail -1)
                if [ -n "$DISK_INFO" ] && [ "$DISK_INFO" != "df:" ]; then
                    USED=$(echo "$DISK_INFO" | awk '{print $3}')
                    TOTAL=$(echo "$DISK_INFO" | awk '{print $2}')
                    PERCENT=$(echo "$DISK_INFO" | awk '{print $5}')
                    AVAIL=$(echo "$DISK_INFO" | awk '{print $4}')
                    echo "  Filesystem:  / (root)"
                    echo "  Total:       $TOTAL"
                    echo "  Used:        $USED"
                    echo "  Available:   $AVAIL"
                    echo "  Usage:       $PERCENT"
                else
                    echo "  Unable to retrieve disk usage information"
                fi
                echo ""
                echo "All Mounted Filesystems:"
                df -h 2>&1 | grep -E "^/dev|Filesystem" | head -10 || echo "  Unable to list filesystems"
            else
                echo "  df command not found"
            fi
            echo ""
            
            echo "🧠 MEMORY (RAM Usage)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            if command -v free &> /dev/null; then
                MEM_INFO=$(free -h 2>&1 | grep -i Mem)
                if [ -n "$MEM_INFO" ]; then
                    TOTAL_MEM=$(echo "$MEM_INFO" | awk '{print $2}')
                    USED_MEM=$(echo "$MEM_INFO" | awk '{print $3}')
                    FREE_MEM=$(echo "$MEM_INFO" | awk '{print $4}')
                    AVAIL_MEM=$(echo "$MEM_INFO" | awk '{print $7}')
                    SHARED_MEM=$(echo "$MEM_INFO" | awk '{print $5}')
                    CACHE_MEM=$(echo "$MEM_INFO" | awk '{print $6}')
                    echo "  Total:       $TOTAL_MEM"
                    echo "  Used:        $USED_MEM"
                    echo "  Free:        $FREE_MEM"
                    if [ -n "$AVAIL_MEM" ] && [ "$AVAIL_MEM" != "-" ] && [ "$AVAIL_MEM" != "available" ]; then
                        echo "  Available:   $AVAIL_MEM"
                    fi
                    if [ -n "$SHARED_MEM" ] && [ "$SHARED_MEM" != "-" ] && [ "$SHARED_MEM" != "shared" ]; then
                        echo "  Shared:      $SHARED_MEM"
                    fi
                    if [ -n "$CACHE_MEM" ] && [ "$CACHE_MEM" != "-" ] && [ "$CACHE_MEM" != "buff/cache" ]; then
                        echo "  Cache:       $CACHE_MEM"
                    fi
                else
                    echo "  Unable to retrieve memory information"
                    echo "  Raw free output:"
                    free -h 2>&1 | head -5
                fi
                echo ""
                echo "Swap Usage:"
                SWAP_INFO=$(free -h 2>&1 | grep -i Swap)
                if [ -n "$SWAP_INFO" ]; then
                    TOTAL_SWAP=$(echo "$SWAP_INFO" | awk '{print $2}')
                    USED_SWAP=$(echo "$SWAP_INFO" | awk '{print $3}')
                    FREE_SWAP=$(echo "$SWAP_INFO" | awk '{print $4}')
                    if [ "$TOTAL_SWAP" != "0B" ] && [ "$TOTAL_SWAP" != "total" ]; then
                        echo "  Total:       $TOTAL_SWAP"
                        echo "  Used:        $USED_SWAP"
                        echo "  Free:        $FREE_SWAP"
                    else
                        echo "  No swap configured"
                    fi
                else
                    echo "  No swap configured"
                fi
            else
                echo "  free command not found"
            fi
            echo ""
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🐳 DOCKER IMAGES"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            if command -v docker &> /dev/null; then
                # Try to list images
                IMAGES_OUTPUT=$(docker images 2>&1)
                if [ $? -eq 0 ] && [ -n "$IMAGES_OUTPUT" ]; then
                    # Check if we have any images (more than just header)
                    IMAGE_COUNT=$(echo "$IMAGES_OUTPUT" | grep -v "REPOSITORY" | grep -v "^$" | wc -l)
                    if [ "$IMAGE_COUNT" -gt 0 ]; then
                        docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" 2>/dev/null | head -20
                    else
                        echo "  No Docker images found"
                    fi
                else
                    # Docker command failed - likely permission issue
                    echo "  ⚠️  Unable to list Docker images"
                    echo "  Error: $(echo "$IMAGES_OUTPUT" | head -1)"
                    echo "  Possible issue: User not in docker group"
                    echo "  Fix: Run 'newgrp docker' or log out and back in"
                fi
            else
                echo "  Docker is not installed"
                echo "  Install with: ./setup.sh docker"
            fi
            echo ""
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🚀 RUNNING CONTAINERS (with Ports)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            if command -v docker &> /dev/null; then
                RUNNING=$(docker ps --format "{{.Names}}" 2>/dev/null | wc -l)
                if [ "$RUNNING" -gt 0 ]; then
                    echo "Container Name | Image | Status | Ports"
                    echo "---------------|-------|--------|------"
                    docker ps --format "{{.Names}} | {{.Image}} | {{.Status}} | {{.Ports}}" 2>/dev/null
                    echo ""
                    echo "Detailed Port Mapping:"
                    docker ps --format "{{.Names}}: {{.Ports}}" 2>/dev/null | while read line; do
                        echo "  $line"
                    done
                else
                    echo "  No containers currently running"
                fi
                echo ""
                echo "All Containers (including stopped):"
                docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | head -20
            else
                echo "  Docker is not installed"
            fi
            echo ""
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📁 FOLDER STRUCTURE"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Home Directory (~):"
            if [ -d "$HOME" ]; then
                ls -lah "$HOME" | grep -E "^d|^-" | head -20
                echo ""
                echo "  Total: $(ls -1 "$HOME" 2>/dev/null | wc -l) items"
            fi
            echo ""
            
            echo "Project Directory (~/fastapi-inventory-detector):"
            PROJECT_DIR="$HOME/fastapi-inventory-detector"
            if [ -d "$PROJECT_DIR" ]; then
                echo "  Directory exists"
                echo ""
                echo "  Contents:"
                ls -lah "$PROJECT_DIR" | head -30
                echo ""
                if [ -d "$PROJECT_DIR/app" ]; then
                    echo "  app/ structure:"
                    find "$PROJECT_DIR/app" -maxdepth 2 -type d | head -20 | sed 's|^|    |'
                fi
            else
                echo "  Directory does not exist"
            fi
            echo ""
            
            if [ -n "$REMOTE_DIR" ] && [ "$REMOTE_DIR" != "$HOME" ] && [ "$REMOTE_DIR" != "~" ]; then
                echo "Requested Directory ($REMOTE_DIR):"
                if [ -d "$REMOTE_DIR" ]; then
                    ls -lah "$REMOTE_DIR" | head -30
                else
                    echo "  Directory does not exist or is not accessible"
                fi
                echo ""
            fi
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🌐 LISTENING PORTS"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "Port | Process"
            echo "-----|--------"
            sudo netstat -tuln 2>/dev/null | grep LISTEN | awk '{print $4}' | sed 's/.*://' | sort -n | uniq | while read port; do
                if [ -n "$port" ]; then
                    process=$(sudo lsof -i :$port 2>/dev/null | tail -1 | awk '{print $1}' || echo "unknown")
                    echo "$port | $process"
                fi
            done || ss -tuln 2>/dev/null | grep LISTEN | awk '{print $5}' | sed 's/.*://' | sort -n | uniq | while read port; do
                if [ -n "$port" ]; then
                    echo "$port | (check with: sudo lsof -i :$port)"
                fi
            done || echo "  Unable to list ports (may require sudo)"
            echo ""
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ Server Status Summary"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EOF
        
        echo ""
        echo "✅ View complete!"
        ;;
    
    cleanup)
        # Cleanup server resources
        CLEANUP_OPTION="${2:-project}"
        echo "🧹 Cleaning up Linux Test Server..."
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Option: $CLEANUP_OPTION"
        echo ""
        
        case "$CLEANUP_OPTION" in
            project)
                echo "📁 Removing project directory..."
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -rf $REMOTE_DIR && echo '✅ Project directory removed' || echo '⚠️  Directory may not exist'"
                ;;
            
            deployment)
                echo "📁 Removing deployment folder from server..."
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -rf $REMOTE_DIR/deployment && echo '✅ Deployment folder removed' || echo '⚠️  Deployment folder may not exist'"
                ;;
            
            docker)
                echo "🐳 Cleaning up Docker resources..."
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" << 'EOF'
                    docker ps -q | xargs -r docker stop 2>/dev/null || true
                    docker ps -aq | xargs -r docker rm 2>/dev/null || true
                    docker images -q | xargs -r docker rmi -f 2>/dev/null || true
                    docker volume ls -q | xargs -r docker volume rm 2>/dev/null || true
                    docker system prune -af --volumes 2>/dev/null || true
                    echo "✅ Docker cleanup complete"
EOF
                ;;
            
            all)
                echo "🧹 Full cleanup (project + Docker)..."
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -rf $REMOTE_DIR && echo '✅ Project directory removed' || echo '⚠️  Directory may not exist'"
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" << 'EOF'
                    docker ps -q | xargs -r docker stop 2>/dev/null || true
                    docker ps -aq | xargs -r docker rm 2>/dev/null || true
                    docker images -q | xargs -r docker rmi -f 2>/dev/null || true
                    docker volume ls -q | xargs -r docker volume rm 2>/dev/null || true
                    docker system prune -af --volumes 2>/dev/null || true
                    echo "✅ Docker cleanup complete"
EOF
                ;;
            
            system)
                echo "🔧 System cleanup (apt cache, logs)..."
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" << 'EOF'
                    sudo apt-get clean 2>/dev/null || true
                    sudo apt-get autoclean 2>/dev/null || true
                    sudo journalctl --vacuum-time=7d 2>/dev/null || true
                    sudo rm -rf /tmp/* 2>/dev/null || true
                    sudo rm -rf /var/tmp/* 2>/dev/null || true
                    echo "✅ System cleanup complete"
EOF
                ;;
            
            *)
                echo "❌ Invalid option: $CLEANUP_OPTION"
                echo ""
                echo "Usage: ./connect.sh cleanup [option]"
                echo ""
                echo "Options:"
                echo "  project     - Remove only the uploaded project directory (default)"
                echo "  deployment  - Remove deployment folder from server"
                echo "  docker      - Remove Docker containers, images, and volumes"
                echo "  all         - Remove project directory and Docker resources"
                echo "  system      - System cleanup (apt cache, logs, etc.)"
                exit 1
                ;;
        esac
        
        echo ""
        echo "✅ Cleanup complete!"
        ;;
    
    setup)
        # Setup server prerequisites - only installs if missing
        echo "🔧 Setting up Server Prerequisites"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   This will check and install only missing components"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" << 'EOF'
            echo "🔍 Step 1: Checking Docker installation..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            if command -v docker &> /dev/null; then
                DOCKER_VERSION=$(docker --version 2>/dev/null || echo "unknown")
                echo "   ✅ Docker is already installed: $DOCKER_VERSION"
                
                # Check if user is in docker group
                if groups | grep -q docker; then
                    echo "   ✅ User is in docker group"
                else
                    echo "   ⚠️  User is NOT in docker group"
                    echo "   Adding user to docker group..."
                    sudo usermod -aG docker $USER
                    echo "   ✅ User added to docker group"
                    echo "   ⚠️  Note: You may need to log out and back in, or run 'newgrp docker'"
                fi
                
                # Check if Docker service is running
                if sudo systemctl is-active --quiet docker; then
                    echo "   ✅ Docker service is running"
                else
                    echo "   ⚠️  Docker service is not running, starting it..."
                    sudo systemctl start docker
                    sudo systemctl enable docker
                    echo "   ✅ Docker service started and enabled"
                fi
            else
                echo "   ❌ Docker is not installed"
                echo "   📦 Installing Docker..."
                echo ""
                
                # Update package index
                echo "   📥 Updating package index..."
                sudo apt-get update -qq
                
                # Install prerequisites
                echo "   📦 Installing prerequisites..."
                sudo apt-get install -y -qq \
                    ca-certificates \
                    curl \
                    gnupg \
                    lsb-release > /dev/null 2>&1
                
                # Add Docker's official GPG key
                echo "   🔑 Adding Docker's official GPG key..."
                sudo mkdir -p /etc/apt/keyrings
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg > /dev/null 2>&1
                
                # Set up Docker repository
                echo "   📋 Setting up Docker repository..."
                echo \
                  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
                  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
                
                # Install Docker Engine
                echo "   📦 Installing Docker Engine..."
                sudo apt-get update -qq
                sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null 2>&1
                
                # Add user to docker group
                echo "   👤 Adding user to docker group..."
                sudo usermod -aG docker $USER
                
                # Start Docker service
                echo "   🚀 Starting Docker service..."
                sudo systemctl start docker
                sudo systemctl enable docker
                
                echo "   ✅ Docker installed successfully!"
                echo "   ⚠️  Note: You may need to log out and back in, or run 'newgrp docker' for group changes to take effect"
            fi
            echo ""
            
            echo "🔍 Step 2: Checking Python and requests module..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            # Check Python
            if command -v python3 &> /dev/null; then
                PYTHON_VERSION=$(python3 --version 2>/dev/null || echo "unknown")
                echo "   ✅ Python3 is installed: $PYTHON_VERSION"
            else
                echo "   ❌ Python3 is not installed"
                echo "   📦 Installing Python3..."
                sudo apt-get update -qq
                sudo apt-get install -y -qq python3 python3-pip > /dev/null 2>&1
                echo "   ✅ Python3 installed"
            fi
            
            # Check requests module
            if python3 -c "import requests" 2>/dev/null; then
                REQUESTS_VERSION=$(python3 -c "import requests; print(requests.__version__)" 2>/dev/null || echo "unknown")
                echo "   ✅ Python requests module is installed: version $REQUESTS_VERSION"
            else
                echo "   ❌ Python requests module is not installed"
                echo "   📦 Installing requests module..."
                
                # Try pip3 first, then python3 -m pip
                if command -v pip3 &> /dev/null; then
                    pip3 install --user requests > /dev/null 2>&1 || \
                    python3 -m pip install --user requests > /dev/null 2>&1
                else
                    python3 -m pip install --user requests > /dev/null 2>&1
                fi
                
                # Verify installation
                if python3 -c "import requests" 2>/dev/null; then
                    REQUESTS_VERSION=$(python3 -c "import requests; print(requests.__version__)" 2>/dev/null || echo "unknown")
                    echo "   ✅ Requests module installed successfully: version $REQUESTS_VERSION"
                else
                    echo "   ⚠️  Warning: Failed to install requests module"
                    echo "   You may need to install it manually: pip3 install requests --user"
                fi
            fi
            echo ""
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ Setup check complete!"
            echo ""
            echo "📋 Summary:"
            echo "   - Docker: $(command -v docker &> /dev/null && echo '✅ Installed' || echo '❌ Not installed')"
            echo "   - Python3: $(command -v python3 &> /dev/null && echo '✅ Installed' || echo '❌ Not installed')"
            echo "   - Requests: $(python3 -c 'import requests' 2>/dev/null && echo '✅ Installed' || echo '❌ Not installed')"
            echo ""
            echo "💡 Next steps:"
            echo "   - If Docker was just installed, you may need to run 'newgrp docker' or reconnect"
            echo "   - Then proceed with deployment: ./deploy.sh rebuild"
            echo ""
EOF
        
        echo "✅ Setup complete!"
        ;;
    
    *)
        echo "❌ Unknown action: $ACTION"
        echo ""
        echo "Usage: ./connect.sh [action] [options...]"
        echo ""
        echo "Actions:"
        echo "  connect [pem_file]     - Interactive SSH connection (default)"
        echo "  view [directory]       - View server contents and system info"
        echo "  cleanup [option]       - Cleanup server resources"
        echo "  setup                  - Setup server prerequisites (Docker, Python requests)"
        echo ""
        echo "Cleanup options:"
        echo "  project     - Remove project directory (default)"
        echo "  deployment  - Remove deployment folder from server"
        echo "  docker      - Remove Docker resources"
        echo "  all         - Remove project and Docker"
        echo "  system      - System cleanup"
        exit 1
        ;;
esac
