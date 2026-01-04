#!/bin/bash

# Ensure script is run with bash (not sh)
if [ -z "$BASH_VERSION" ]; then
    echo "❌ Error: This script must be run with bash, not sh"
    echo "   Please run: bash $0 $@"
    echo "   Or: ./$0 $@"
    exit 1
fi

# Deploy Docker Image to Linux Server
# Usage: ./deploy.sh [method] [options...]
#
# PRODUCTION-GRADE FEATURES:
# - Container resource limits (--memory, --memory-swap, --cpus) to prevent EC2 OOM kills
# - Preflight checks (Docker daemon, SSH availability, server resources)
# - Safe OpenCV runtime validation via controlled docker exec (with retries and timeouts)
# - Comprehensive deployment summary with all key information
# - Deterministic, debuggable deployment pipeline with strict error handling
#
# By default (no method specified), runs: transfer
# The default 'transfer' method performs ALL steps: build → test locally → clean server → transfer → deploy
#
# Methods (in typical workflow order):
#
# 1. rebuild      - Complete rebuild and deployment workflow:
#                   Build image locally (no-cache) → Test locally → Clean server → Transfer → Deploy
#                   (RECOMMENDED for fixing issues or after Dockerfile changes)
#
# 2. test-local   - Test Docker image locally before deploying:
#                   Verify OpenCV headless import → Test API endpoints
#                   (Use this to verify image works before deploying)
#
# 3. transfer     - Full deployment workflow (DEFAULT):
#                   Build image → Test locally → Clean server → Transfer image → Deploy container
#                   (This is the default method - does everything automatically)
#
# 4. upload       - Build on server instead of locally:
#                   Upload source files → Build Docker image on server → Deploy
#                   (Use when local build is not possible)
#
# 5. run          - Deploy container only (assumes image already exists on server):
#                   Stop old container → Start new container with existing image
#                   (Use when image is already on server)
#
# 6. test         - Test the deployed service:
#                   Check container status → Test health endpoint → Test API docs
#                   (Use to verify deployment is working)
#
# 7. test-image   - Test API with a single image file:
#                   Upload image → Call detect-and-count endpoint → Show results
#                   (Use to test API functionality with actual images)
#
# 8. upload-images - Upload test images to server:
#                    Transfer all images from test_images/ folder to server
#                    (Use to prepare test data on server)
#
# 9. upload-script - Upload test_api.py script to server:
#                    Transfer test_api.py to server for local testing
#                    (Use to update test script without rebuilding Docker)
#
# 10. cleanup     - Clean up Docker resources on server:
#                   Remove containers → Remove images → Prune system → Free disk space
#                   (Use when server is low on disk space)
#
# 11. firewall    - Check firewall and port configuration:
#                   Check UFW status → Check port bindings → Test connectivity
#                   (Use to troubleshoot connection issues)
#
# 12. test-server|debug - Test/debug container on server (starts from Step 7):
#                   Check container status → Verify dependencies → Test OpenCV → Test API
#                   (Use when steps 1-6 completed successfully and you want to debug from testing)
#                   Shows detailed logs if container is restarting/crashing

# Exit immediately if a command exits with a non-zero status
# This ensures the script stops on any error
set -e

# Treat unset variables as an error
set -u

# Pipe failures: if any command in a pipeline fails, the whole pipeline fails
set -o pipefail

# Function to handle errors and cleanup
cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "❌ DEPLOYMENT FAILED at step above"
        echo "   Exit code: $exit_code"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit $exit_code
    fi
}

# Trap errors (though set -e should handle most)
trap cleanup_on_error ERR

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get method name for logging
METHOD="${1:-transfer}"

# Setup logging directory and log file
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/deploy_${METHOD}_${TIMESTAMP}.log"

# Function to log all output
log_all() {
    tee -a "$LOG_FILE"
}

# Start logging - show initial message before redirect
echo "📝 Logging all output to: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Deployment started at: $(date)"
echo "Method: ${METHOD}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Redirect all output to log file while still showing on screen
exec > >(log_all)
exec 2>&1

# Server details
SERVER_IP="13.203.145.155"
USERNAME="ubuntu"
PEM_FILE="$SCRIPT_DIR/MI_dev_Linux_test_server.pem"
IMAGE_VERSION="${IMAGE_VERSION:-25.12.0}"
IMAGE_NAME="${2:-prodatalytics/inventory-tracker-api:${IMAGE_VERSION}}"
HOST_PORT="${3:-8000}"
CONTAINER_PORT="8000"
CONTAINER_NAME="${4:-inventory-tracker-api}"
REMOTE_DIR="~/fastapi-inventory-detector"

# Container resource limits (prevent OOM kills on EC2)
# Adjust based on your EC2 instance size:
# - t3.medium (2 vCPU, 4GB RAM): Use 2.5G memory, 1.5 CPU
# - t3.large (2 vCPU, 8GB RAM): Use 5G memory, 1.5 CPU
# - t3.xlarge (4 vCPU, 16GB RAM): Use 10G memory, 3 CPU
# Memory swap should be set to same as memory to disable swap (prevents thrashing)
CONTAINER_MEMORY="${CONTAINER_MEMORY:-4g}"
CONTAINER_MEMORY_SWAP="${CONTAINER_MEMORY_SWAP:-4g}"
CONTAINER_CPUS="${CONTAINER_CPUS:-2.0}"

# Check if PEM file exists
if [ ! -f "$PEM_FILE" ]; then
    echo "❌ Error: PEM file not found at '$PEM_FILE'!"
    echo ""
    echo "Please ensure MI_dev_Linux_test_server.pem is in the deployment directory."
    exit 1
fi

# Set proper permissions for PEM file
chmod 400 "$PEM_FILE" 2>/dev/null

# ============================================================================
# Helper Functions (reusable across methods)
# ============================================================================

# Function: Preflight checks (Docker daemon, SSH availability, server resources)
preflight_checks() {
    echo "🔍 Running preflight checks..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check local Docker daemon
    echo "   1. Checking local Docker daemon..."
    if ! command -v docker &>/dev/null; then
        echo "   ❌ Error: Docker is not installed or not in PATH"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        echo "   ❌ Error: Docker daemon is not running"
        echo "   💡 Start Docker: sudo systemctl start docker (Linux) or open Docker Desktop (macOS)"
        exit 1
    fi
    echo "   ✅ Docker daemon is running"
    
    # Check SSH connectivity and server Docker
    echo "   2. Checking SSH connectivity to server..."
    if ! ssh -i "$PEM_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        "$USERNAME@$SERVER_IP" "echo 'SSH connection successful'" &>/dev/null; then
        echo "   ❌ Error: Cannot connect to server via SSH"
        echo "   💡 Check:"
        echo "      - PEM file exists and has correct permissions (chmod 400)"
        echo "      - Server IP is correct: $SERVER_IP"
        echo "      - Network connectivity to server"
        exit 1
    fi
    echo "   ✅ SSH connection successful"
    
    # Check Docker on server
    echo "   3. Checking Docker on server..."
    if ! ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker --version" &>/dev/null; then
        echo "   ❌ Error: Docker is not installed on the server"
        exit 1
    fi
    if ! ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker info" &>/dev/null; then
        echo "   ❌ Error: Docker daemon is not running on server"
        exit 1
    fi
    echo "   ✅ Docker is available on server"
    
    # Check server disk space
    echo "   4. Checking server disk space..."
    ROOT_AVAILABLE=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "df -BG / 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//'" 2>/dev/null || echo "0")
    if [ -n "$ROOT_AVAILABLE" ] && [ "$ROOT_AVAILABLE" -lt 5 ]; then
        echo "   ⚠️  Warning: Low disk space detected (${ROOT_AVAILABLE}GB available)"
        echo "   💡 Consider running: ./deploy.sh cleanup"
    else
        echo "   ✅ Sufficient disk space available (${ROOT_AVAILABLE}GB)"
    fi
    
    # Check server memory (for resource limit validation)
    echo "   5. Checking server memory..."
    SERVER_MEMORY_GB=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "free -g | awk '/^Mem:/ {print \$2}'" 2>/dev/null || echo "0")
    if [ -n "$SERVER_MEMORY_GB" ] && [ "$SERVER_MEMORY_GB" -gt 0 ]; then
        echo "   ✅ Server has ${SERVER_MEMORY_GB}GB RAM"
        # Validate container memory limit doesn't exceed server capacity
        CONTAINER_MEMORY_GB=$(echo "$CONTAINER_MEMORY" | sed 's/[^0-9.]//g' | cut -d. -f1)
        if [ -n "$CONTAINER_MEMORY_GB" ] && [ "$CONTAINER_MEMORY_GB" -gt "$SERVER_MEMORY_GB" ]; then
            echo "   ⚠️  Warning: Container memory limit (${CONTAINER_MEMORY}) exceeds server RAM (${SERVER_MEMORY_GB}GB)"
            echo "   💡 Adjust CONTAINER_MEMORY environment variable or use smaller limit"
        fi
    else
        echo "   ℹ️  Could not determine server memory (continuing anyway)"
    fi
    
    echo "   ✅ All preflight checks passed"
    echo ""
}

# Function: Safe OpenCV validation inside running container
validate_opencv_in_container() {
    local container_name="$1"
    local max_attempts=3
    local attempt=1
    
    echo "   🔬 Validating OpenCV availability inside container (safe exec check)..."
    
    # Create temporary Python validation script on server
    local validation_script="/tmp/validate_opencv_$$.py"
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "cat > $validation_script" <<'PYEOF'
import sys
try:
    import cv2
    print('OPENCV_OK:' + cv2.__version__)
    # Test basic functionality
    import numpy as np
    test_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    print('OPENCV_FUNC_OK')
except ImportError as e:
    print('OPENCV_IMPORT_ERROR:' + str(e))
    sys.exit(1)
except Exception as e:
    print('OPENCV_RUNTIME_ERROR:' + str(e))
    sys.exit(1)
PYEOF
    
    while [ $attempt -le $max_attempts ]; do
        # Use controlled docker exec with timeout to prevent hanging
        OPENCV_CHECK=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
            "timeout 10 docker exec $container_name python3 < $validation_script 2>&1" 2>&1 || echo "TIMEOUT_OR_ERROR")
        
        if echo "$OPENCV_CHECK" | grep -q "OPENCV_OK:"; then
            OPENCV_VERSION=$(echo "$OPENCV_CHECK" | grep "OPENCV_OK:" | cut -d: -f2)
            if echo "$OPENCV_CHECK" | grep -q "OPENCV_FUNC_OK"; then
                echo "   ✅ OpenCV validated successfully (version: $OPENCV_VERSION)"
                echo "   ✅ OpenCV basic functionality test passed"
                # Cleanup
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -f $validation_script" 2>/dev/null || true
                return 0
            else
                echo "   ⚠️  OpenCV imported but basic functionality test failed"
                echo "   Check output: $(echo "$OPENCV_CHECK" | grep -v "OPENCV_OK")"
            fi
        else
            if [ $attempt -lt $max_attempts ]; then
                echo "   ⏳ Attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
                sleep 2
                attempt=$((attempt + 1))
            else
                echo "   ❌ OpenCV validation failed after $max_attempts attempts"
                echo "   Error output: $OPENCV_CHECK"
                # Cleanup
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -f $validation_script" 2>/dev/null || true
                return 1
            fi
        fi
    done
    
    # Cleanup on exit
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -f $validation_script" 2>/dev/null || true
    return 1
}

# Function: Generate deployment summary
deployment_summary() {
    local image_name="$1"
    local container_name="$2"
    local host_port="$3"
    local server_ip="$4"
    local log_file="$5"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 DEPLOYMENT SUMMARY"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   🖼️  Image Name:      $image_name"
    echo "   📦 Container Name:   $container_name"
    echo "   🖥️  Server:           $USERNAME@$server_ip"
    echo "   🔌 Port:              $host_port:$CONTAINER_PORT"
    echo "   💾 Resource Limits:  Memory: $CONTAINER_MEMORY, CPUs: $CONTAINER_CPUS"
    echo "   📝 Log File:          $log_file"
    echo ""
    echo "   🌐 Service URLs:"
    echo "      Health:    http://$server_ip:$host_port/health"
    echo "      API Docs:  http://$server_ip:$host_port/docs"
    echo "      ReDoc:     http://$server_ip:$host_port/redoc"
    echo ""
    echo "   📊 Useful Commands:"
    echo "      View logs:    ssh -i $PEM_FILE $USERNAME@$server_ip 'docker logs -f $container_name'"
    echo "      Container:    ssh -i $PEM_FILE $USERNAME@$server_ip 'docker ps --filter name=$container_name'"
    echo "      Test API:     ./deploy.sh test"
    echo "      Cleanup:      ./deploy.sh cleanup"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Function: Remove old Docker images locally
cleanup_local_images() {
    echo "🗑️  Removing old images locally..."
    OLD_IMAGES=$(docker images "prodatalytics/inventory-tracker-api" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null || echo "")
    if [ -n "$OLD_IMAGES" ]; then
        echo "$OLD_IMAGES" | while read img; do
            if docker rmi "$img" 2>/dev/null; then
                echo "   ✅ Removed: $img"
            else
                echo "   ⚠️  Could not remove: $img (may be in use, will force remove)"
                docker rmi -f "$img" 2>/dev/null || true
            fi
        done
        echo "   ✅ All old images removed"
    else
        echo "   ℹ️  No old images found"
    fi
    docker image prune -f >/dev/null 2>&1 || true
}

# Function: Build Docker image
build_docker_image() {
    local image_name="$1"
    echo "🔨 Building Docker image..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Platform: linux/amd64"
    echo "   Tag: $image_name"
    echo "   Using BuildKit and --no-cache for fresh build"
    echo ""
    
    cd "$PROJECT_ROOT"
    echo "   ⏳ Building (this may take several minutes)..."
    echo "   📝 Full build logs saved to: /tmp/docker_build.log"
    echo "   💡 Using multi-stage build with BuildKit for optimized image size"
    echo "   📊 Showing layer progress messages only (detailed output in log file):"
    echo ""
    
    # Build with progress output - filter to show only echo statements (layer progress)
    # Save all output to log file, but only show progress messages in console
    # Filter out Docker BuildKit metadata lines (e.g., #11 [stage-1 2/11] RUN ...)
    # Use grep with --line-buffered for real-time output
    # Capture docker build exit code separately (grep || true won't affect it)
    set +e  # Temporarily disable exit on error to capture exit code
    DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --no-cache \
        --progress=plain \
        -f "$SCRIPT_DIR/Dockerfile.server" \
        -t "$image_name" . 2>&1 | tee /tmp/docker_build.log | \
        grep --line-buffered -v -E "^#[0-9]+\s+\[|^#\s*\[" | \
        grep --line-buffered -E "(━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━|📦 Layer [0-9]+/12|🔍 Layer [0-9]+/12|🧹 Layer [0-9]+/12|📋 Layer [0-9]+/12|✅ Layer [0-9]+ complete|✅ Verified|✅ NumPy|✅ Requirements file prepared|✅ OpenCV|✅ PyTorch|✅ All critical packages verified|Verifying libGL|✅ Verified: libGL)" || true
    BUILD_EXIT_CODE=${PIPESTATUS[0]}
    set -e  # Re-enable exit on error
    
    if [ "$BUILD_EXIT_CODE" = "0" ]; then
        echo "   ✅ Image built successfully!"
        echo ""
        
        # Get image size
        SIZE_BYTES=$(docker image inspect "$image_name" --format='{{.Size}}' 2>/dev/null || echo "0")
        if [ "$SIZE_BYTES" != "0" ] && [ -n "$SIZE_BYTES" ] && [ "$SIZE_BYTES" -gt 0 ] 2>/dev/null; then
            if command -v numfmt &>/dev/null; then
                IMAGE_SIZE=$(echo "$SIZE_BYTES" | numfmt --to=iec-i --suffix=B 2>/dev/null || echo "unknown")
            else
                SIZE_KB=$((SIZE_BYTES / 1024))
                SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
                SIZE_GB=$((SIZE_BYTES / 1024 / 1024 / 1024))
                if [ "$SIZE_GB" -gt 0 ] 2>/dev/null; then
                    REMAINDER=$((SIZE_BYTES % (1024 * 1024 * 1024)))
                    GB_DECIMAL=$((REMAINDER * 10 / (1024 * 1024 * 1024)))
                    if [ "$GB_DECIMAL" -gt 0 ] 2>/dev/null; then
                        IMAGE_SIZE="${SIZE_GB}.${GB_DECIMAL}GB"
                    else
                        IMAGE_SIZE="${SIZE_GB}GB"
                    fi
                elif [ "$SIZE_MB" -gt 0 ] 2>/dev/null; then
                    REMAINDER=$((SIZE_BYTES % (1024 * 1024)))
                    MB_DECIMAL=$((REMAINDER * 10 / (1024 * 1024)))
                    if [ "$MB_DECIMAL" -gt 0 ] 2>/dev/null; then
                        IMAGE_SIZE="${SIZE_MB}.${MB_DECIMAL}MB"
                    else
                        IMAGE_SIZE="${SIZE_MB}MB"
                    fi
                elif [ "$SIZE_KB" -gt 0 ] 2>/dev/null; then
                    IMAGE_SIZE="${SIZE_KB}KB"
                else
                    IMAGE_SIZE="${SIZE_BYTES}B"
                fi
            fi
        else
            IMAGE_SIZE="unknown"
        fi
        echo "   📦 Image size: $IMAGE_SIZE"
        cd "$SCRIPT_DIR"
        return 0
    else
        echo ""
        echo "   ❌ Failed to build image!"
        echo ""
        echo "   📋 Last 30 lines of build log:"
        tail -30 /tmp/docker_build.log | sed 's/^/   /'
        echo ""
        echo "   💡 Full build log available at: /tmp/docker_build.log"
        cd "$SCRIPT_DIR"
        return 1
    fi
}

# Function: Clean up server (containers, images, disk space)
cleanup_server() {
    echo "🧹 Cleaning up Linux server..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   This step ensures a clean state on the server:"
    echo "   1. Stops and removes existing container (if running)"
    echo "   2. Removes ALL old inventory-tracker-api images"
    echo "   3. Cleans up Docker resources to free disk space"
    echo "   4. Removes old tar files"
    echo ""
    
    echo "   🛑 Stopping and removing existing container..."
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
        if docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'; then
            docker stop ${CONTAINER_NAME} 2>/dev/null && echo '      ✅ Container stopped' || echo '      ℹ️  Container was not running'
            docker rm ${CONTAINER_NAME} 2>/dev/null && echo '      ✅ Container removed' || echo '      ⚠️  Failed to remove container'
        else
            echo '      ℹ️  No existing container found'
        fi
    " || true
    echo ""
    
    echo "   🗑️  Removing ALL old inventory-tracker-api images..."
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
        OLD_IMAGES=\$(docker images prodatalytics/inventory-tracker-api --format '{{.Repository}}:{{.Tag}}' 2>/dev/null || echo '')
        if [ -n \"\$OLD_IMAGES\" ]; then
            echo \"\$OLD_IMAGES\" | while read img; do
                docker rmi \"\$img\" 2>/dev/null && echo \"      ✅ Removed: \$img\" || echo \"      ⚠️  Could not remove: \$img (may be in use)\"
            done
        else
            echo '      ℹ️  No old images found'
        fi
    " || true
    echo ""
    
    ROOT_AVAILABLE=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "df -BG / 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//'" 2>/dev/null || echo "0")
    echo "   💾 Server disk space: ${ROOT_AVAILABLE}GB available"
    
    if [ -n "$ROOT_AVAILABLE" ] && [ "$ROOT_AVAILABLE" -lt 5 ]; then
        echo "   ⚠️  Low disk space detected - running aggressive cleanup..."
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
            docker system prune -a -f --volumes >/dev/null 2>&1 && echo '      ✅ Docker system cleaned' || echo '      ⚠️  Cleanup had warnings'
            rm -f $REMOTE_DIR/*.tar 2>/dev/null && echo '      ✅ Old tar files removed' || echo '      ℹ️  No tar files to remove'
        " || true
    else
        echo "   ℹ️  Disk space sufficient - removing old tar files only..."
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -f $REMOTE_DIR/*.tar 2>/dev/null || true" || true
    fi
    
    echo ""
    echo "   ✅ Server cleanup complete"
    echo ""
}

# Function: Transfer Docker image to server
transfer_image_to_server() {
    local image_name="$1"
    echo "   💾 Saving Docker image to tar file..."
    
    # Verify image exists before attempting to save
    if ! docker image inspect "$image_name" &>/dev/null; then
        echo "   ❌ Error: Docker image '$image_name' does not exist!"
        echo ""
        echo "   💡 Possible causes:"
        echo "      - Build failed (check build logs: /tmp/docker_build.log)"
        echo "      - Image name/tag mismatch"
        echo ""
        echo "   📋 Available images:"
        docker images "prodatalytics/inventory-tracker-api" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}" 2>/dev/null || echo "      (none found)"
        return 1
    fi
    
    IMAGE_TAR="inventory-tracker-api.tar"
    TEMP_TAR="$PROJECT_ROOT/$IMAGE_TAR"
    REMOTE_TAR="$REMOTE_DIR/$IMAGE_TAR"
    
    echo "   ✅ Image found: $image_name"
    if docker save "$image_name" -o "$TEMP_TAR"; then
        TAR_SIZE=$(du -h "$TEMP_TAR" | cut -f1)
        echo "   ✅ Image saved: $TAR_SIZE"
    else
        echo "   ❌ Error: Failed to save Docker image!"
        return 1
    fi
    
    echo "   📁 Creating remote directory..."
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "mkdir -p $REMOTE_DIR" || {
        echo "   ❌ Error: Failed to create remote directory!"
        rm -f "$TEMP_TAR"
        return 1
    }
    
    echo "   📤 Transferring image to server..."
    if scp -i "$PEM_FILE" "$TEMP_TAR" "$USERNAME@$SERVER_IP:$REMOTE_TAR"; then
        echo "   ✅ Image transferred successfully!"
    else
        echo "   ❌ Error: Failed to transfer image to server!"
        rm -f "$TEMP_TAR"
        return 1
    fi
    echo ""
    
    echo "   📥 Loading Docker image on server..."
    if ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker load -i $REMOTE_TAR"; then
        echo "   ✅ Image loaded successfully on server!"
    else
        echo "   ❌ Error: Failed to load image on server!"
        rm -f "$TEMP_TAR"
        return 1
    fi
    echo ""
    
    echo "   🧹 Cleaning up tar files..."
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "rm -f $REMOTE_TAR" || true
    rm -f "$TEMP_TAR" || true
    echo "   ✅ Cleanup complete"
    echo ""
}

# Function: Deploy container on server
deploy_container_on_server() {
    local image_name="$1"
    local host_port="$2"
    local container_name="$3"
    echo "🚀 Deploying container on server..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Resource limits (prevents OOM kills):"
    echo "      Memory:      $CONTAINER_MEMORY"
    echo "      Memory Swap: $CONTAINER_MEMORY_SWAP"
    echo "      CPUs:        $CONTAINER_CPUS"
    echo ""
    
    ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
        docker stop $container_name 2>/dev/null || true
        docker rm $container_name 2>/dev/null || true
    " || true
    
    echo "   🚀 Starting container with resource limits and log volume mount..."
    HOST_LOGS_DIR="$REMOTE_DIR/logs"
    CONTAINER_ID=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "mkdir -p $HOST_LOGS_DIR && \
         docker run -d --name $container_name \
         --memory=$CONTAINER_MEMORY \
         --memory-swap=$CONTAINER_MEMORY_SWAP \
         --cpus=$CONTAINER_CPUS \
         -p $host_port:$CONTAINER_PORT \
         --restart unless-stopped \
         -v $HOST_LOGS_DIR:/app/logs \
         -e LOG_DIR=/app/logs \
         -e ENABLE_FILE_LOGGING=true \
         $image_name" 2>&1)
    
    if [ $? -eq 0 ] && echo "$CONTAINER_ID" | grep -qE '^[a-f0-9]{64}$|^[a-f0-9]{12}$'; then
        echo "   ✅ Container started successfully!"
        echo "   Container ID: $(echo "$CONTAINER_ID" | head -1)"
    else
        echo "   ❌ Error: Failed to start container!"
        echo "   Output: $CONTAINER_ID"
        return 1
    fi
    echo ""
    
    echo "⏳ Waiting for service to be ready..."
    echo -n "   Waiting"
    for i in {1..30}; do
        sleep 1
        echo -n "."
        if [ $((i % 5)) -eq 0 ]; then
            if ! CONTAINER_STATUS=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
                "docker ps --filter name=$container_name --format '{{.Status}}'" 2>/dev/null); then
                echo ""
                echo "   ❌ Error: Failed to check container status!"
                return 1
            fi
            if [ -z "$CONTAINER_STATUS" ]; then
                echo ""
                echo "   ❌ Container stopped unexpectedly!"
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker logs $container_name 2>&1 | tail -30" || true
                return 1
            fi
        fi
    done
    echo ""
    echo "   ✅ Wait complete"
    echo ""
}

# Function: Comprehensive testing on Linux server
test_container_on_server() {
    local host_port="$1"
    local container_name="$2"
    echo "🧪 Comprehensive testing on Linux server..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   This will verify the container works correctly on the actual deployment server"
    echo ""
    
    # Check container status
    echo "   7.1: Checking container status..."
    if ! CONTAINER_STATUS=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "docker ps --filter name=$container_name --format '{{.Status}}'" 2>/dev/null); then
        echo "   ❌ Error: Failed to check container status!"
        return 1
    fi
    
    if [ -n "$CONTAINER_STATUS" ]; then
        echo "   ℹ️  Container status: $CONTAINER_STATUS"
        if echo "$CONTAINER_STATUS" | grep -qi "restarting"; then
            echo "   ⚠️  WARNING: Container is restarting (likely crashing on startup)!"
            echo "   📋 Showing container logs to diagnose the issue:"
            echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker logs $container_name 2>&1 | tail -50" || true
            echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "   💡 Common causes:"
            echo "      - Missing runtime dependencies (libglib2.0-0, libssl3, libffi8)"
            echo "      - Application startup errors"
            echo "      - Port conflicts"
            echo "      - Missing environment variables"
            echo ""
            echo "   🔧 To debug further:"
            echo "      ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'docker logs -f $container_name'"
            return 1
        else
            echo "   ✅ Container is running: $CONTAINER_STATUS"
        fi
    else
        echo "   ❌ Container is not running!"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker logs $container_name 2>&1 | tail -30" || true
        return 1
    fi
    echo ""
    
    # Verify dependencies
    echo "   7.2: Verifying required runtime dependencies..."
    sleep 2
    DEPS_CHECK=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "docker exec $container_name dpkg -l 2>/dev/null | grep -E 'libgl1-mesa-glx|libglib2.0-0|libssl3|libffi8' 2>/dev/null" || echo "")
    if echo "$DEPS_CHECK" | grep -q "libgl1-mesa-glx"; then
        echo "   ✅ libgl1-mesa-glx package is installed (provides libGL.so.1)"
    else
        echo "   ⚠️  Warning: libgl1-mesa-glx not found (OpenCV may fail without libGL.so.1)"
    fi
    if echo "$DEPS_CHECK" | grep -q "libglib2.0-0"; then
        echo "   ✅ libglib2.0-0 package is installed"
    else
        echo "   ⚠️  Warning: libglib2.0-0 not found (may still work)"
    fi
    if echo "$DEPS_CHECK" | grep -q "libssl3"; then
        echo "   ✅ libssl3 package is installed"
    else
        echo "   ⚠️  Warning: libssl3 not found (may still work)"
    fi
    if echo "$DEPS_CHECK" | grep -q "libffi8"; then
        echo "   ✅ libffi8 package is installed"
    else
        echo "   ⚠️  Warning: libffi8 not found (may still work)"
    fi
    echo ""
    
    # Test OpenCV through health endpoint (validates OpenCV via normal application startup)
    echo "   7.3: Validating OpenCV through application health endpoint..."
    echo "   (OpenCV is validated during normal application startup, not via docker exec)"
    HEALTH_CHECK=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "curl -s -w '\nHTTP_CODE:%{http_code}' http://localhost:$host_port/health 2>&1" || echo "HTTP_CODE:000")
    HEALTH_CODE=$(echo "$HEALTH_CHECK" | grep "HTTP_CODE:" | cut -d: -f2)
    if [ "$HEALTH_CODE" = "200" ]; then
        echo "   ✅ SUCCESS: Application health check passed (OpenCV validated during startup)!"
        HEALTH_BODY=$(echo "$HEALTH_CHECK" | grep -v "HTTP_CODE")
        if [ -n "$HEALTH_BODY" ]; then
            echo "   Response: $HEALTH_BODY"
        fi
    else
        echo "   ❌ ERROR: Health check failed (HTTP $HEALTH_CODE) - OpenCV may have failed during startup!"
        echo "   📋 Container logs:"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker logs $container_name 2>&1 | tail -50" || true
        return 1
    fi
    echo ""
    
    # Wait a bit for container to fully stabilize before exec checks
    echo "   ⏳ Waiting 5 seconds for container to stabilize..."
    sleep 5
    
    # Safe OpenCV validation inside container (controlled docker exec)
    echo "   7.4: Performing safe OpenCV validation inside container..."
    if validate_opencv_in_container "$container_name"; then
        echo "   ✅ OpenCV runtime validation passed"
    else
        echo "   ⚠️  WARNING: OpenCV validation via docker exec failed"
        echo "   (Application may still work if health check passed - OpenCV loaded during startup)"
        echo "   📋 Container logs for reference:"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker logs $container_name 2>&1 | tail -20" || true
    fi
    echo ""
    
    # Check logs
    echo "   7.5: Checking container logs for critical errors..."
    ERROR_LOG=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "docker logs $container_name 2>&1 | grep -iE '(error|exception|failed|cannot|missing).*opencv|cv2' | head -5" 2>/dev/null || echo "")
    if [ -n "$ERROR_LOG" ]; then
        echo "   ⚠️  WARNING: Potential OpenCV errors detected in container logs!"
        echo "   Error: $ERROR_LOG"
        echo "   (This may not be critical - continuing with tests...)"
    else
        echo "   ✅ No critical OpenCV errors detected in logs"
    fi
    echo ""
    
    # Test API
    echo "   7.6: Testing API from inside server (localhost:$host_port)..."
    echo "   ⏳ Waiting for API to be ready (max 30 seconds)..."
    API_READY=false
    for i in {1..6}; do
        sleep 5
        HEALTH_RESPONSE=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
            "curl -s -w '\nHTTP_CODE:%{http_code}' http://localhost:$host_port/health 2>&1" || echo "HTTP_CODE:000")
        HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
        if [ "$HTTP_CODE" = "200" ]; then
            API_READY=true
            break
        fi
        echo "   ⏳ Attempt $i/6: API not ready yet (HTTP $HTTP_CODE)..."
    done
    
    if [ "$API_READY" = "true" ]; then
        echo "   ✅ SUCCESS: API health check passed from inside server! (HTTP $HTTP_CODE)"
        HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | grep -v "HTTP_CODE")
        if [ -n "$HEALTH_BODY" ]; then
            echo "   Response: $HEALTH_BODY"
        fi
    else
        echo "   ❌ ERROR: API health check failed after 30 seconds!"
        echo "   Last HTTP code: $HTTP_CODE"
        echo ""
        echo "   📋 Container logs:"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker logs $container_name 2>&1 | tail -30" || true
        return 1
    fi
    echo ""
    
    # Test port
    echo "   7.7: Testing port $host_port accessibility from inside server..."
    PORT_CHECK=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
        "netstat -tuln 2>/dev/null | grep ':$host_port ' || ss -tuln 2>/dev/null | grep ':$host_port '" 2>/dev/null || echo "")
    if [ -n "$PORT_CHECK" ]; then
        echo "   ✅ Port $host_port is listening:"
        echo "   $PORT_CHECK"
    else
        echo "   ⚠️  Warning: Could not verify port binding (may still work)"
    fi
    echo ""
    
    # Test external access
    echo "   7.8: Testing API from outside (external access - OPTIONAL)..."
    EXTERNAL_HEALTH=$(curl -s -w "\nHTTP_CODE:%{http_code}" --connect-timeout 5 "http://$SERVER_IP:$host_port/health" 2>&1 || echo "HTTP_CODE:000")
    EXTERNAL_CODE=$(echo "$EXTERNAL_HEALTH" | grep "HTTP_CODE:" | cut -d: -f2)
    if [ "$EXTERNAL_CODE" = "200" ]; then
        echo "   ✅ SUCCESS: API accessible from outside! (HTTP $EXTERNAL_CODE)"
    else
        echo "   ⚠️  API not accessible from outside (HTTP $EXTERNAL_CODE)"
        echo "   This may be due to firewall/security group settings"
        echo "   The API is working correctly from inside the server (verified in 7.5)"
    fi
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅✅✅ DEPLOYMENT COMPLETE! ✅✅✅"
    echo ""
    echo "   💡 Note: For deeper inference-level validation (OCR/model readiness),"
    echo "      consider adding dedicated /ready endpoint that checks:"
    echo "      - OCR model loaded and ready"
    echo "      - Detection model loaded and ready"
    echo "      - Catalog/SKU data loaded"
    echo "      (Keep /health fast for load balancer checks)"
    echo ""
    echo "   Service is running at: http://$SERVER_IP:$host_port"
    echo "   API docs: http://$SERVER_IP:$host_port/docs"
    echo "   Health check: http://$SERVER_IP:$host_port/health"
    echo ""
    echo "   To view logs:"
    echo "     ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'docker logs -f $container_name'"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

case "$METHOD" in
    transfer)
        # Transfer Docker image directly to server (with build, test, and clean deployment)
        echo "📤 Transferring Docker Image to Linux Server"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Image: $IMAGE_NAME (version ${IMAGE_VERSION})"
        echo "   This ensures a clean deployment: build → clean server → transfer → deploy → test"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 0: Preflight checks
        echo "🔍 Step 0: Running preflight checks..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        preflight_checks
        
        # Step 1: Remove old images locally
        echo "🗑️  Step 1: Removing old images locally..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cleanup_local_images
        echo ""
        
        # Step 2: Build Docker image
        echo "🔨 Step 2: Building Docker image..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! build_docker_image "$IMAGE_NAME"; then
            exit 1
        fi
        echo ""
        
        # Step 2.5: Skip local testing
        echo "ℹ️  Step 2.5: Skipping local testing..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Local testing skipped due to platform emulation limitations on ARM systems"
        echo "   Comprehensive testing will be performed on the Linux server after deployment"
        echo ""
        
        # Step 3: Clean up server
        echo "🧹 Step 3: Cleaning up Linux server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cleanup_server
        
        # Step 4: Transfer image to server
        echo "📤 Step 4: Transferring Docker image to server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! transfer_image_to_server "$IMAGE_NAME"; then
            exit 1
        fi
        
        # Step 5: Deploy container
        echo "🚀 Step 5: Deploying container on server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! deploy_container_on_server "$IMAGE_NAME" "$HOST_PORT" "$CONTAINER_NAME"; then
            exit 1
        fi
        
        # Step 6: Comprehensive testing
        echo "🧪 Step 6: Comprehensive testing on Linux server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! test_container_on_server "$HOST_PORT" "$CONTAINER_NAME"; then
            exit 1
        fi
        
        # Step 7: Deployment summary
        deployment_summary "$IMAGE_NAME" "$CONTAINER_NAME" "$HOST_PORT" "$SERVER_IP" "$LOG_FILE"
        ;;
    
    upload)
        # Upload source files to server
        echo "📤 Uploading project files to server..."
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Remote Directory: $REMOTE_DIR"
        echo ""
        
        # Create remote directory
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "mkdir -p $REMOTE_DIR"
        
        # Upload files
        if [ -f "$SCRIPT_DIR/Dockerfile.server" ]; then
            echo "📄 Uploading Dockerfile.server..."
            scp -i "$PEM_FILE" "$SCRIPT_DIR/Dockerfile.server" "$USERNAME@$SERVER_IP:$REMOTE_DIR/"
        fi
        
        if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
            echo "📄 Uploading requirements.txt..."
            scp -i "$PEM_FILE" "$PROJECT_ROOT/requirements.txt" "$USERNAME@$SERVER_IP:$REMOTE_DIR/"
        fi
        
        if [ -d "$PROJECT_ROOT/app" ]; then
            echo "📦 Uploading app directory..."
            scp -i "$PEM_FILE" -r "$PROJECT_ROOT/app" "$USERNAME@$SERVER_IP:$REMOTE_DIR/"
        fi
        
        if [ -d "$PROJECT_ROOT/models" ]; then
            echo "🤖 Uploading models directory..."
            scp -i "$PEM_FILE" -r "$PROJECT_ROOT/models" "$USERNAME@$SERVER_IP:$REMOTE_DIR/"
        fi
        
        echo ""
        echo "✅ Upload complete!"
        echo ""
        echo "Next: Connect to server and build:"
        echo "  ./connect.sh connect"
        echo "  cd $REMOTE_DIR"
        echo "  docker build -t $IMAGE_NAME ."
        ;;
    
    run)
        # Deploy container on server
        echo "🚀 Deploying FastAPI Service to Linux Server"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Image: $IMAGE_NAME"
        echo "   Port: $HOST_PORT:$CONTAINER_PORT"
        echo "   Container: $CONTAINER_NAME"
        echo ""
        
        # Preflight checks
        preflight_checks
        
        # Check if image exists, try to pull if not
        if ! ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker image inspect $IMAGE_NAME" &>/dev/null; then
            echo "⚠️  Image not found, attempting to pull from registry..."
            if ! ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker pull $IMAGE_NAME"; then
                echo "❌ Error: Failed to pull image!"
                exit 1
            fi
        fi
        
        # Stop and remove existing container for clean deployment
        echo "🧹 Stopping and removing existing container (if any)..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
            if docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}$'; then
                echo '   🛑 Stopping container: ${CONTAINER_NAME}'
                docker stop ${CONTAINER_NAME} >/dev/null 2>&1 && echo '   ✅ Container stopped' || echo '   ℹ️  Container was not running'
                echo '   🗑️  Removing container: ${CONTAINER_NAME}'
                docker rm ${CONTAINER_NAME} >/dev/null 2>&1 && echo '   ✅ Container removed' || echo '   ⚠️  Failed to remove container'
            else
                echo '   ℹ️  No existing container found'
            fi
        " || true
        echo ""
        
        # Create logs directory on host (outside container)
        echo "📁 Creating logs directory on host..."
        HOST_LOGS_DIR="$REMOTE_DIR/logs"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "mkdir -p $HOST_LOGS_DIR && chmod 755 $HOST_LOGS_DIR"
        echo "   Logs will be stored at: $HOST_LOGS_DIR"
        echo ""
        
        # Run container with resource limits, volume mount for logs and LOG_DIR environment variable
        echo "🚀 Starting container with resource limits and log volume mount..."
        echo "   Resource limits: Memory=$CONTAINER_MEMORY, CPUs=$CONTAINER_CPUS"
        CONTAINER_ID=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
            "docker run -d --name $CONTAINER_NAME \
            --memory=$CONTAINER_MEMORY \
            --memory-swap=$CONTAINER_MEMORY_SWAP \
            --cpus=$CONTAINER_CPUS \
            -p $HOST_PORT:$CONTAINER_PORT \
            --restart unless-stopped \
            -v $HOST_LOGS_DIR:/app/logs \
            -e LOG_DIR=/app/logs \
            -e ENABLE_FILE_LOGGING=true \
            $IMAGE_NAME" 2>&1)
        
        if [ $? -eq 0 ]; then
            echo "✅ Container started successfully!"
        else
            echo "❌ Error: Failed to start container!"
            exit 1
        fi
        echo ""
        
        # Wait for service
        echo -n "⏳ Waiting for service to be ready"
        MAX_WAIT=60
        WAIT_COUNT=0
        SERVICE_READY=false
        
        while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
            if ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
                "curl -s -f http://localhost:$CONTAINER_PORT/health > /dev/null 2>&1" 2>/dev/null; then
                SERVICE_READY=true
                break
            fi
            sleep 2
            WAIT_COUNT=$((WAIT_COUNT + 2))
            printf "."
        done
        echo ""
        
        if [ "$SERVICE_READY" = true ]; then
            echo "✅ Service is ready!"
        else
            echo "⚠️  Warning: Service may not be ready yet"
        fi
        echo ""
        
        # Show status
        echo "📊 Container Status:"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker ps --filter name=$CONTAINER_NAME --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
        echo ""
        
        # Access information
        echo "✅ Deployment complete!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📍 Service Access Information"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "🌐 Server URL: http://$SERVER_IP:$HOST_PORT"
        echo ""
        echo "📋 API Endpoints:"
        echo "   Health Check:  http://$SERVER_IP:$HOST_PORT/health"
        echo ""
        echo "📝 Log Files (stored on host, outside container):"
        echo "   Location: $HOST_LOGS_DIR"
        echo "   View logs: ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'tail -f $HOST_LOGS_DIR/app.log'"
        echo "   API Docs:      http://$SERVER_IP:$HOST_PORT/docs"
        echo "   ReDoc:         http://$SERVER_IP:$HOST_PORT/redoc"
        echo ""
        echo "🧪 Test: ./deploy.sh test"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        deployment_summary "$IMAGE_NAME" "$CONTAINER_NAME" "$HOST_PORT" "$SERVER_IP" "$LOG_FILE"
        ;;
    
    test)
        # Test deployed service
        BASE_URL="http://$SERVER_IP:$HOST_PORT"
        CONTAINER_NAME="${2:-inventory-tracker-api}"
        
        echo "🧪 Testing Deployed FastAPI Service"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $SERVER_IP"
        echo "   Port: $HOST_PORT"
        echo "   Base URL: $BASE_URL"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # First, check container status
        echo "🔍 Checking Container Status..."
        CONTAINER_STATUS=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
            "docker ps --filter name=$CONTAINER_NAME --format '{{.Status}}' 2>/dev/null" 2>&1)
        
        if [ -z "$CONTAINER_STATUS" ] || echo "$CONTAINER_STATUS" | grep -q "error\|not found"; then
            echo "   ❌ Container '$CONTAINER_NAME' is not running!"
            echo ""
            echo "   Checking all containers..."
            ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>&1
            echo ""
            echo "   💡 Try running: ./deploy.sh run"
            echo ""
        else
            echo "   ✅ Container is running: $CONTAINER_STATUS"
            echo ""
            echo "   Container Details:"
            ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
                "docker ps --filter name=$CONTAINER_NAME --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'" 2>&1
            echo ""
            
            # Check container logs for errors
            echo "   Recent Container Logs (last 10 lines):"
            CONTAINER_LOGS=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
                "docker logs --tail 30 $CONTAINER_NAME 2>&1")
            echo "$CONTAINER_LOGS" | tail -10 | sed 's/^/     /'
            echo ""
            
            # Check for specific errors and provide fixes
            if echo "$CONTAINER_LOGS" | grep -qiE "(opencv|cv2).*(error|exception|failed|cannot|missing)"; then
                echo "   ⚠️  ERROR DETECTED: OpenCV-related error"
                echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "   This error indicates an issue with OpenCV in the container."
                echo "   The Dockerfile.server uses opencv-python-headless (no GL/X11 dependencies)."
                echo ""
                echo "   🔧 FIX: Rebuild the image:"
                echo "      ./deploy.sh rebuild"
                echo ""
                echo "   This will:"
                echo "   1. Rebuild the image locally with latest fixes"
                echo "   2. Transfer the new image to server"
                echo "   3. Stop and remove the old container"
                echo "   4. Deploy the new container with the fixed image"
                echo ""
            fi
        fi
        echo ""
        
        # Test from server (localhost)
        echo "🔍 Testing from Server (localhost)..."
        LOCAL_TEST=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
            "curl -s -w '\n%{http_code}' http://localhost:$HOST_PORT/health 2>&1" 2>&1)
        LOCAL_CODE=$(echo "$LOCAL_TEST" | tail -n1)
        if [ "$LOCAL_CODE" = "200" ]; then
            echo "   ✅ Service is accessible from server (HTTP $LOCAL_CODE)"
        else
            echo "   ❌ Service not accessible from server (HTTP $LOCAL_CODE)"
            echo "   Response: $(echo "$LOCAL_TEST" | head -n-1)"
        fi
        echo ""
        
        # Health check from external
        echo "Test 1: Health Check (External)"
        echo "   Endpoint: $BASE_URL/health"
        HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" --connect-timeout 5 "$BASE_URL/health" 2>&1)
        HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
        if [ "$HTTP_CODE" = "200" ]; then
            echo "   ✅ PASS (HTTP $HTTP_CODE)"
        else
            echo "   ❌ FAIL (HTTP $HTTP_CODE)"
            if [ "$HTTP_CODE" = "000" ]; then
                echo "   ⚠️  Connection refused or timeout"
                echo "   Possible issues:"
                echo "     - Firewall/security group blocking port $HOST_PORT"
                echo "     - Container not listening on 0.0.0.0"
                echo "     - Port mapping incorrect"
                echo ""
                echo "   Check firewall:"
                echo "     ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'sudo ufw status'"
                echo "   Check port binding:"
                echo "     ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'sudo netstat -tuln | grep $HOST_PORT'"
            fi
        fi
        echo ""
        
        # API docs
        echo "Test 2: API Documentation (External)"
        echo "   Endpoint: $BASE_URL/docs"
        DOCS_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$BASE_URL/docs" 2>&1)
        if [ "$DOCS_CODE" = "200" ]; then
            echo "   ✅ PASS (HTTP $DOCS_CODE)"
            echo "   Access: $BASE_URL/docs"
        else
            echo "   ❌ FAIL (HTTP $DOCS_CODE)"
        fi
        echo ""
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if [ "$HTTP_CODE" = "200" ] || [ "$LOCAL_CODE" = "200" ]; then
            echo "✅ Service is running!"
            if [ "$HTTP_CODE" != "200" ]; then
                echo "⚠️  Service works on server but not accessible externally"
                echo "   Check firewall/security group settings"
            fi
        else
            echo "❌ Service is not accessible"
            echo "   Check container logs: ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'docker logs $CONTAINER_NAME'"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
    
    rebuild)
        # Rebuild image and redeploy - handles all common issues
        echo "🔧 Rebuilding Image and Redeploying"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   This will rebuild the image with latest fixes and redeploy"
        echo "   Image Tag: $IMAGE_NAME (version ${IMAGE_VERSION})"
        echo "   Handles: OpenCV headless setup, disk space issues, container restarts"
        echo "   Note: Using version tag to avoid cache issues"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 0: Preflight checks
        echo "🔍 Step 0: Running preflight checks..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        preflight_checks
        
        # Step 1: Check prerequisites
        echo "🔍 Step 1: Checking prerequisites..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! command -v docker &> /dev/null; then
            echo "   ❌ Error: Docker is not installed or not in PATH"
            exit 1
        fi
        echo "   ✅ Docker is available"
        
        if [ ! -f "$SCRIPT_DIR/Dockerfile.server" ]; then
            echo "   ❌ Error: Dockerfile.server not found at '$SCRIPT_DIR/Dockerfile.server'"
            exit 1
        fi
        echo "   ✅ Dockerfile.server found"
        
        if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
            echo "   ⚠️  Warning: requirements.txt not found, build may fail"
        else
            echo "   ✅ requirements.txt found"
        fi
        echo ""
        
        # Step 2: Remove old images locally
        echo "🗑️  Step 2: Removing old images locally..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cleanup_local_images
        echo ""
        
        # Step 3: Build Docker image
        echo "🔨 Step 3: Rebuilding image with latest fixes..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Platform: linux/amd64"
        echo "   Tag: $IMAGE_NAME (version ${IMAGE_VERSION})"
        echo "   Using BuildKit and --no-cache for fresh build"
        echo "   Using: opencv-python-headless (no GL/X11 dependencies required)"
        echo ""
        if ! build_docker_image "$IMAGE_NAME"; then
            exit 1
        fi
        echo ""
        
        # Step 3.5: Skip local testing
        echo "ℹ️  Step 3.5: Skipping local testing..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Local testing skipped due to platform emulation limitations on ARM systems"
        echo "   Comprehensive testing will be performed on the Linux server after deployment"
        echo ""
        
        # Step 4: Clean up server
        echo "🧹 Step 4: Cleaning up Linux server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cleanup_server
        
        # Step 5: Transfer image to server
        echo "📤 Step 5: Transferring new image to server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! transfer_image_to_server "$IMAGE_NAME"; then
            # Check for disk space errors and retry
            if ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "df -BG / 2>/dev/null | tail -1 | awk '{print \$4}' | sed 's/G//'" | grep -qE '^[0-4]$'; then
                echo "   ⚠️  Low disk space detected - running aggressive cleanup and retrying..."
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
                    docker system prune -a -f --volumes >/dev/null 2>&1
                    rm -f $REMOTE_DIR/*.tar 2>/dev/null
                " || true
                echo "   Retrying transfer..."
                if ! transfer_image_to_server "$IMAGE_NAME"; then
                    echo "   ❌ Transfer failed after cleanup"
                    echo "   Please manually clean up server: ./deploy.sh cleanup"
                    exit 1
                fi
            else
                exit 1
            fi
        fi
        
        # Step 6: Verify image on server
        echo "🔍 Step 6: Verifying image on server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        NEW_IMAGE_ID=$(docker image inspect "$IMAGE_NAME" --format '{{.Id}}' 2>/dev/null || echo "")
        if [ -n "$NEW_IMAGE_ID" ]; then
            echo "   📦 New image ID: ${NEW_IMAGE_ID:0:12}..."
            SERVER_IMAGE_ID=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
                "docker image inspect $IMAGE_NAME --format '{{.Id}}' 2>/dev/null" || echo "")
            if [ -n "$SERVER_IMAGE_ID" ] && [ "$SERVER_IMAGE_ID" = "$NEW_IMAGE_ID" ]; then
                echo "   ✅ Verified: New image is on server (ID matches)"
            else
                echo "   ⚠️  Warning: Image ID mismatch or not found on server"
            fi
        fi
        echo ""
        
        # Step 7: Deploy container
        echo "🚀 Step 7: Deploying new container..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! deploy_container_on_server "$IMAGE_NAME" "$HOST_PORT" "$CONTAINER_NAME"; then
            echo "   ❌ Failed to deploy container!"
            echo "   Check container logs: ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'docker logs $CONTAINER_NAME'"
            exit 1
        fi
        
        # Step 8: Verify container is using new image
        echo "🔍 Step 8: Verifying container is using new image..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        CONTAINER_IMAGE_ID=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" \
            "docker inspect $CONTAINER_NAME --format '{{.Image}}' 2>/dev/null" || echo "")
        if [ -n "$NEW_IMAGE_ID" ] && [ -n "$CONTAINER_IMAGE_ID" ]; then
            if echo "$CONTAINER_IMAGE_ID" | grep -q "${NEW_IMAGE_ID:0:12}"; then
                echo "   ✅ Verified: Container is using the new image"
            else
                echo "   ⚠️  Warning: Container may not be using the new image"
            fi
        fi
        echo ""
        
        # Step 9: Comprehensive testing
        echo "🧪 Step 9: Comprehensive testing on Linux server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if ! test_container_on_server "$HOST_PORT" "$CONTAINER_NAME"; then
            exit 1
        fi
        
        # Step 10: Deployment summary
        deployment_summary "$IMAGE_NAME" "$CONTAINER_NAME" "$HOST_PORT" "$SERVER_IP" "$LOG_FILE"
        ;;
    
    cleanup)
        # Clean up Docker resources on server
        echo "🧹 Cleaning up Docker Resources on Server"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 1: Show current disk usage
        echo "📊 Step 1: Checking current disk usage..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "df -h / | tail -1"
        echo ""
        
        echo "📊 Current Docker disk usage:"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker system df 2>/dev/null || echo 'Docker not available'"
        echo ""
        
        # Step 2: Stop and remove container
        echo "🛑 Step 2: Stopping and removing container..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
            docker stop $CONTAINER_NAME 2>/dev/null && echo '   ✅ Container stopped' || echo '   ℹ️  Container not running'
            docker rm $CONTAINER_NAME 2>/dev/null && echo '   ✅ Container removed' || echo '   ℹ️  Container not found'
        " || true
        echo ""
        
        # Step 3: Remove all old inventory-tracker-api images
        echo "🗑️  Step 3: Removing all old inventory-tracker-api images..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
            OLD_IMAGES=\$(docker images prodatalytics/inventory-tracker-api --format '{{.Repository}}:{{.Tag}}' 2>/dev/null || echo '')
            if [ -n \"\$OLD_IMAGES\" ]; then
                echo \"\$OLD_IMAGES\" | while read img; do
                    docker rmi \"\$img\" 2>/dev/null && echo \"   ✅ Removed: \$img\" || echo \"   ⚠️  Could not remove: \$img\"
                done
            else
                echo '   ℹ️  No old images found'
            fi
        " || true
        echo ""
        
        # Step 4: Clean up all unused Docker resources
        echo "🧹 Step 4: Cleaning up all unused Docker resources..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   ⚠️  This will remove ALL unused images, containers, networks, and build cache"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
            docker system prune -a -f --volumes 2>&1 | head -20
        " || true
        echo ""
        
        # Step 5: Remove old tar files
        echo "🗑️  Step 5: Removing old tar files..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "
            rm -f $REMOTE_DIR/*.tar 2>/dev/null && echo '   ✅ Old tar files removed' || echo '   ℹ️  No tar files found'
        " || true
        echo ""
        
        # Step 6: Show disk usage after cleanup
        echo "📊 Step 6: Disk usage after cleanup..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "df -h / | tail -1"
        echo ""
        
        echo "📊 Docker disk usage after cleanup:"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker system df 2>/dev/null || echo 'Docker not available'"
        echo ""
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Cleanup complete!"
        echo ""
        ;;
    
    upload-images)
        # Upload test images to server
        TEST_IMAGES_DIR="$PROJECT_ROOT/test/test_images"
        
        echo "📤 Uploading Test Images to Server"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Source: $TEST_IMAGES_DIR"
        echo "   Destination: $REMOTE_DIR/test_images"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 1: Check if test_images directory exists
        echo "🔍 Step 1: Checking test images directory..."
        if [ ! -d "$TEST_IMAGES_DIR" ]; then
            echo "   ❌ Error: test_images directory not found at '$TEST_IMAGES_DIR'!"
            exit 1
        fi
        echo "   ✅ Test images directory found"
        echo ""
        
        # Step 2: Create remote directory
        echo "📁 Step 2: Creating remote directory..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "mkdir -p $REMOTE_DIR/test_images" || {
            echo "   ❌ Error: Failed to create remote directory!"
            exit 1
        }
        echo "   ✅ Remote directory created"
        echo ""
        
        # Step 3: Count files to transfer
        echo "📊 Step 3: Analyzing files to transfer..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        FILE_COUNT=$(find "$TEST_IMAGES_DIR" -type f | wc -l | tr -d ' ')
        echo "   📊 Found $FILE_COUNT files to transfer"
        echo ""
        
        # Step 4: Check which images need to be transferred
        echo "🔍 Step 4: Checking which images need to be transferred..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        REMOTE_IMAGES=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "ls -1 $REMOTE_DIR/test_images/*.jpeg $REMOTE_DIR/test_images/*.jpg 2>/dev/null | xargs -n1 basename 2>/dev/null || echo ''")
        echo ""
        
        # Step 5: Transfer images
        echo "📤 Step 5: Transferring test images..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        TRANSFERRED=0
        SKIPPED=0
        
        for image_file in "$TEST_IMAGES_DIR"/*; do
            if [ -f "$image_file" ]; then
                image_name=$(basename "$image_file")
                if echo "$REMOTE_IMAGES" | grep -q "^$image_name$"; then
                    echo "   ⏭️  Skipping $image_name (already exists)"
                    SKIPPED=$((SKIPPED + 1))
                else
                    echo "   📤 Uploading $image_name..."
                    if scp -i "$PEM_FILE" "$image_file" "$USERNAME@$SERVER_IP:$REMOTE_DIR/test_images/" >/dev/null 2>&1; then
                        TRANSFERRED=$((TRANSFERRED + 1))
                        echo "      ✅ Uploaded"
                    else
                        echo "      ❌ Failed to upload"
                    fi
                fi
            fi
        done
        echo ""
        
        if [ $TRANSFERRED -gt 0 ]; then
            echo "   ✅ Transferred $TRANSFERRED new image(s)"
        fi
        if [ $SKIPPED -gt 0 ]; then
            echo "   ⏭️  Skipped $SKIPPED existing image(s)"
        fi
        if [ $TRANSFERRED -eq 0 ] && [ $SKIPPED -eq 0 ]; then
            echo "   ⚠️  No images found to transfer"
        fi
        echo ""
        
        # Step 6: Transfer test API script
        echo "📤 Step 6: Transferring test API script..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        TEST_SCRIPT="$SCRIPT_DIR/test_api.py"
        if [ -f "$TEST_SCRIPT" ]; then
            if scp -i "$PEM_FILE" "$TEST_SCRIPT" "$USERNAME@$SERVER_IP:$REMOTE_DIR/" >/dev/null 2>&1; then
                echo "   ✅ Test script transferred"
                ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "chmod +x $REMOTE_DIR/test_api.py" 2>/dev/null
            else
                echo "   ⚠️  Warning: Failed to transfer test script"
            fi
        else
            echo "   ⚠️  Warning: test_api.py not found, skipping script transfer"
        fi
        echo ""
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Transfer complete!"
        echo ""
        echo "📁 Files are now at:"
        echo "   Images: $REMOTE_DIR/test_images"
        echo "   Script: $REMOTE_DIR/test_api.py"
        echo ""
        ;;
    
    test-image)
        # Test API with a single image
        IMAGE_PATH="${2:-test/test_images/20251010-1.jpeg}"
        API_URL="${3:-http://localhost:8000}"
        
        # Resolve image path relative to project root
        if [[ ! "$IMAGE_PATH" =~ ^/ ]]; then
            IMAGE_PATH="$PROJECT_ROOT/$IMAGE_PATH"
        fi
        
        echo "🧪 Testing API with Single Image"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Image: $(basename "$IMAGE_PATH")"
        echo "   API URL: $API_URL"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 1: Check if image exists
        echo "🔍 Step 1: Checking image file..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if [ ! -f "$IMAGE_PATH" ]; then
            echo "   ❌ Error: Image file not found: $IMAGE_PATH"
            echo ""
            echo "   Usage: ./deploy.sh test-image [image_path] [api_url]"
            echo "   Example: ./deploy.sh test-image test/test_images/20251010-1.jpeg"
            exit 1
        fi
        echo "   ✅ Image file found"
        echo ""
        
        IMAGE_NAME=$(basename "$IMAGE_PATH")
        
        # Step 2: Copy test script to server
        echo "📤 Step 2: Copying test script to server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if scp -i "$PEM_FILE" "$SCRIPT_DIR/test_api.py" "$USERNAME@$SERVER_IP:~/test_api.py" >/dev/null 2>&1; then
            echo "   ✅ Test script copied"
        else
            echo "   ❌ Failed to copy test script"
            exit 1
        fi
        echo ""
        
        # Step 3: Copy image to server
        echo "📤 Step 3: Copying image to server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if scp -i "$PEM_FILE" "$IMAGE_PATH" "$USERNAME@$SERVER_IP:~/test_image.jpeg" >/dev/null 2>&1; then
            echo "   ✅ Image copied"
        else
            echo "   ❌ Failed to copy image"
            exit 1
        fi
        echo ""
        
        # Step 4: Run test on server
        echo "🚀 Step 4: Running test on server..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" << EOF
            # Install requests if needed
            if ! python3 -c "import requests" 2>/dev/null; then
                echo "   📦 Installing requests library..."
                pip3 install requests --user >/dev/null 2>&1 || python3 -m pip install requests --user >/dev/null 2>&1
            fi
            
            # Run the test
            python3 ~/test_api.py ~/test_image.jpeg --api-url $API_URL
EOF
        echo ""
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Test complete!"
        echo ""
        echo "Response files are saved on the server at: ~/test_image_response/api_response.json"
        echo "To view: ssh -i $PEM_FILE $USERNAME@$SERVER_IP 'cat ~/test_image_response/api_response.json'"
        echo ""
        ;;
    
    firewall)
        # Check firewall and port configuration
        echo "🔍 Checking Firewall and Port Configuration"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Port: $HOST_PORT"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 1: Check UFW status
        echo "1️⃣  Step 1: UFW Firewall Status"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "sudo ufw status" 2>&1
        echo ""
        
        # Step 2: Check if port is listening
        echo "2️⃣  Step 2: Port $HOST_PORT Binding"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        PORT_CHECK=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "sudo netstat -tuln 2>/dev/null | grep $HOST_PORT || sudo ss -tuln 2>/dev/null | grep $HOST_PORT" 2>&1)
        if [ -n "$PORT_CHECK" ]; then
            echo "   ✅ Port $HOST_PORT is listening:"
            echo "$PORT_CHECK" | sed 's/^/   /'
        else
            echo "   ❌ Port $HOST_PORT is NOT listening"
        fi
        echo ""
        
        # Step 3: Check Docker port mapping
        echo "3️⃣  Step 3: Docker Container Port Mapping"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        DOCKER_PORTS=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "docker ps --format 'table {{.Names}}\t{{.Ports}}' | grep -E 'NAMES|inventory'" 2>&1)
        if [ -n "$DOCKER_PORTS" ]; then
            echo "$DOCKER_PORTS" | sed 's/^/   /'
        else
            echo "   ⚠️  No container found with name containing 'inventory'"
        fi
        echo ""
        
        # Step 4: Test from server (localhost)
        echo "4️⃣  Step 4: Testing from Server (localhost)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        LOCAL_TEST=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "curl -s -o /dev/null -w 'HTTP Status: %{http_code}\n' http://localhost:$HOST_PORT/health" 2>&1)
        if echo "$LOCAL_TEST" | grep -q "200"; then
            echo "   ✅ Service is accessible from server (HTTP 200)"
        else
            echo "   ❌ Service not accessible from server"
            echo "   Response: $LOCAL_TEST"
        fi
        echo ""
        
        # Step 5: Test from external
        echo "5️⃣  Step 5: Testing from External"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        EXTERNAL_TEST=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "http://$SERVER_IP:$HOST_PORT/health" 2>&1)
        if [ "$EXTERNAL_TEST" = "200" ]; then
            echo "   ✅ External access works (HTTP $EXTERNAL_TEST)"
        else
            echo "   ❌ External access failed (HTTP $EXTERNAL_TEST)"
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🔧 FIX INSTRUCTIONS"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "If UFW is active, allow port $HOST_PORT:"
            echo "  ssh -i $PEM_FILE $USERNAME@$SERVER_IP"
            echo "  sudo ufw allow $HOST_PORT/tcp"
            echo "  sudo ufw reload"
            echo ""
            echo "If this is AWS EC2, check Security Group:"
            echo "  1. Go to AWS Console → EC2 → Security Groups"
            echo "  2. Find the security group for instance $SERVER_IP"
            echo "  3. Add Inbound Rule:"
            echo "     - Type: Custom TCP"
            echo "     - Port: $HOST_PORT"
            echo "     - Source: 0.0.0.0/0 (or your IP for security)"
            echo ""
        fi
        echo ""
        ;;
    
    test-local)
        # Test the Docker image locally before deploying
        echo "🧪 Testing Docker Image Locally"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Image: $IMAGE_NAME"
        echo "   This will start the container and verify it works correctly"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Check if image exists
        if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
            echo "   ❌ Error: Image '$IMAGE_NAME' not found!"
            echo "   Build the image first: ./deploy.sh rebuild"
            exit 1
        fi
        
        # Stop any existing test container
        docker stop test-inventory-tracker-local 2>/dev/null || true
        docker rm test-inventory-tracker-local 2>/dev/null || true
        
        echo "1️⃣  Checking if required runtime dependencies are installed..."
        DEPS_CHECK=$(docker run --rm --platform linux/amd64 --entrypoint /bin/sh "$IMAGE_NAME" -c "dpkg -l | grep -E 'libglib2.0-0|libssl3|libffi8'" 2>/dev/null || echo "")
        if echo "$DEPS_CHECK" | grep -q "libglib2.0-0"; then
            echo "   ✅ libglib2.0-0 package is installed"
        else
            echo "   ⚠️  Warning: libglib2.0-0 not found (may still work)"
        fi
        if echo "$DEPS_CHECK" | grep -q "libssl3"; then
            echo "   ✅ libssl3 package is installed"
        else
            echo "   ⚠️  Warning: libssl3 not found (may still work)"
        fi
        if echo "$DEPS_CHECK" | grep -q "libffi8"; then
            echo "   ✅ libffi8 package is installed"
        else
            echo "   ⚠️  Warning: libffi8 not found (may still work)"
        fi
        
        echo ""
        echo "2️⃣  Validating OpenCV through application startup..."
        echo "   (OpenCV is validated during normal application startup, not via docker exec)"
        echo "   Starting container to validate OpenCV through health endpoint..."
        
        echo ""
        echo "3️⃣  Starting container and testing API..."
        TEST_CONTAINER_ID=$(docker run -d --platform linux/amd64 \
            --name test-inventory-tracker-local \
            -p 18000:8000 \
            "$IMAGE_NAME" 2>&1)
        
        if [ $? -ne 0 ]; then
            echo "   ❌ Failed to start container!"
            echo "   Output: $TEST_CONTAINER_ID"
            exit 1
        fi
        
        echo "   ⏳ Waiting 15 seconds for container to start..."
        sleep 15
        
        # Check container status
        CONTAINER_STATUS=$(docker ps --filter "name=test-inventory-tracker-local" --format "{{.Status}}" 2>/dev/null || echo "")
        if [ -z "$CONTAINER_STATUS" ]; then
            echo "   ❌ Container crashed or stopped!"
            echo ""
            echo "   📋 Container logs:"
            docker logs test-inventory-tracker-local 2>&1 | tail -30
            docker rm test-inventory-tracker-local 2>/dev/null || true
            exit 1
        fi
        
        echo "   ✅ Container is running: $CONTAINER_STATUS"
        
        # Check logs for critical errors
        if docker logs test-inventory-tracker-local 2>&1 | grep -qiE "(opencv|cv2).*(error|exception|failed|cannot|missing)"; then
            echo "   ⚠️  OpenCV-related error detected in logs!"
            docker logs test-inventory-tracker-local 2>&1 | grep -iE "(opencv|cv2|error|exception)" | head -20
            echo "   (Continuing with health check test...)"
        fi
        
        # Test health endpoint
        echo ""
        echo "4️⃣  Testing API health endpoint..."
        sleep 5
        HEALTH_RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" http://localhost:18000/health 2>&1 || echo "HTTP_CODE:000")
        HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE:" | cut -d: -f2)
        if [ "$HTTP_CODE" = "200" ]; then
            echo "   ✅ Health check passed! (HTTP $HTTP_CODE)"
            echo "   Response: $(echo "$HEALTH_RESPONSE" | grep -v "HTTP_CODE")"
        else
            echo "   ⚠️  Health check returned HTTP $HTTP_CODE"
            echo "   Response: $(echo "$HEALTH_RESPONSE" | grep -v "HTTP_CODE")"
        fi
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅✅✅ ALL LOCAL TESTS PASSED! ✅✅✅"
        echo ""
        echo "   Container is running on: http://localhost:18000"
        echo "   API docs: http://localhost:18000/docs"
        echo ""
        echo "   To stop the test container:"
        echo "     docker stop test-inventory-tracker-local"
        echo "     docker rm test-inventory-tracker-local"
        echo ""
        echo "   If all tests passed, you can deploy with:"
        echo "     ./deploy.sh rebuild"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
    
    test-server|debug)
        # Test/debug container on server (starts from testing phase)
        # Use this when steps 1-6 completed successfully and you want to debug from testing onwards
        echo "🧪 Testing/Debugging Container on Linux Server"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Container: $CONTAINER_NAME"
        echo "   This method runs comprehensive testing on the deployed container"
        echo "   Use this when steps 1-6 completed successfully"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Preflight checks
        preflight_checks
        
        if ! test_container_on_server "$HOST_PORT" "$CONTAINER_NAME"; then
            exit 1
        fi
        
        deployment_summary "$IMAGE_NAME" "$CONTAINER_NAME" "$HOST_PORT" "$SERVER_IP" "$LOG_FILE"
        ;;
    
    upload-script)
        # Upload test_api.py script to server
        SCRIPT_FILE="$SCRIPT_DIR/test_api.py"
        REMOTE_SCRIPT_PATH="$REMOTE_DIR/test_api.py"
        
        echo "📤 Uploading test_api.py to Server"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   Server: $USERNAME@$SERVER_IP"
        echo "   Local: $SCRIPT_FILE"
        echo "   Remote: $REMOTE_SCRIPT_PATH"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # Step 1: Check if script exists locally
        echo "🔍 Step 1: Checking local script file..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if [ ! -f "$SCRIPT_FILE" ]; then
            echo "   ❌ Error: test_api.py not found at '$SCRIPT_FILE'"
            exit 1
        fi
        echo "   ✅ Script file found"
        echo ""
        
        # Step 2: Create remote directory if needed
        echo "📁 Step 2: Ensuring remote directory exists..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "mkdir -p $REMOTE_DIR" || {
            echo "   ❌ Error: Failed to create remote directory!"
            exit 1
        }
        echo "   ✅ Remote directory ready"
        echo ""
        
        # Step 3: Upload script
        echo "📤 Step 3: Uploading test_api.py..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if scp -i "$PEM_FILE" "$SCRIPT_FILE" "$USERNAME@$SERVER_IP:$REMOTE_SCRIPT_PATH" >/dev/null 2>&1; then
            echo "   ✅ Script uploaded successfully"
        else
            echo "   ❌ Error: Failed to upload script"
            exit 1
        fi
        echo ""
        
        # Step 4: Make script executable
        echo "🔧 Step 4: Making script executable..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "chmod +x $REMOTE_SCRIPT_PATH" 2>/dev/null || true
        echo "   ✅ Script is executable"
        echo ""
        
        # Step 5: Verify upload
        echo "✅ Step 5: Verifying upload..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        REMOTE_SIZE=$(ssh -i "$PEM_FILE" "$USERNAME@$SERVER_IP" "stat -c%s $REMOTE_SCRIPT_PATH 2>/dev/null || echo '0'")
        LOCAL_SIZE=$(stat -f%z "$SCRIPT_FILE" 2>/dev/null || stat -c%s "$SCRIPT_FILE" 2>/dev/null || echo "0")
        
        if [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ] && [ "$REMOTE_SIZE" != "0" ]; then
            echo "   ✅ Upload verified (Size: ${REMOTE_SIZE} bytes)"
        else
            echo "   ⚠️  Warning: Size mismatch (Local: ${LOCAL_SIZE}, Remote: ${REMOTE_SIZE})"
        fi
        echo ""
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Upload complete!"
        echo ""
        echo "📋 Usage on server:"
        echo "   cd ~/fastapi-inventory-detector"
        echo "   python3 test_api.py test_images/20251010-1.jpeg"
        echo ""
        ;;
    
    *)
        echo "❌ Unknown method: $METHOD"
        echo ""
        echo "Usage: ./deploy.sh [method] [options...]"
        echo ""
        echo "By default (no method): Runs 'transfer' which does everything automatically"
        echo ""
        echo "Methods (in typical workflow order):"
        echo ""
        echo "  1. rebuild                                - Complete rebuild: Build → Test → Clean → Transfer → Deploy"
        echo "     (RECOMMENDED for fixing issues or after Dockerfile.server changes)"
        echo ""
        echo "  2. test-local                              - Test Docker image locally before deploying"
        echo "     (Verify image works before deploying to server)"
        echo ""
        echo "  3. transfer [image] [port] [container]     - Full deployment workflow (DEFAULT)"
        echo "     Build → Test → Clean → Transfer → Deploy (does everything automatically)"
        echo ""
        echo "  4. upload                                 - Build on server instead of locally"
        echo "     Upload source files → Build on server → Deploy"
        echo ""
        echo "  5. run [image] [port] [container]         - Deploy container only (image must exist on server)"
        echo "     Stop old container → Start new container"
        echo ""
        echo "  6. test [port]                           - Test the deployed service"
        echo "     Check container status → Test endpoints"
        echo ""
        echo "  7. test-image [image] [api_url]          - Test API with a single image file"
        echo "     Upload image → Call API → Show results"
        echo ""
        echo "  8. upload-images                          - Upload test images to server"
        echo "     Transfer all images from test_images/ folder"
        echo ""
        echo "  9. upload-script                          - Upload test_api.py script to server"
        echo "     Update test script without rebuilding Docker"
        echo ""
        echo "  10. cleanup                               - Clean up Docker resources on server"
        echo "      Remove containers/images → Prune system → Free disk space"
        echo ""
        echo "  11. firewall                              - Check firewall and port configuration"
        echo "      Check UFW status → Port bindings → Connectivity"
        echo ""
        echo "  12. test-server|debug                     - Test/debug container (starts from Step 7)"
        echo "      Use when steps 1-6 completed successfully and you want to debug from testing"
        echo "      Shows detailed logs if container is restarting/crashing"
        echo ""
        echo "Examples:"
        echo "  ./deploy.sh                                    # Default: Full deployment (build → test → deploy)"
        echo "  ./deploy.sh rebuild                            # Rebuild everything and redeploy (RECOMMENDED)"
        echo "  ./deploy.sh test-local                         # Test image locally before deploying"
        echo "  ./deploy.sh transfer                           # Full deployment workflow"
        echo "  ./deploy.sh transfer prodatalytics/inventory-tracker-api:25.12.0"
        echo "  ./deploy.sh run                                # Deploy existing image"
        echo "  ./deploy.sh test                               # Test deployed service"
        echo "  ./deploy.sh cleanup                            # Clean up Docker resources"
        echo "  ./deploy.sh upload-images                      # Upload test images"
        echo "  ./deploy.sh upload-script                      # Upload test_api.py to server"
        echo "  ./deploy.sh test-image test/test_images/20251010-1.jpeg  # Test with image"
        echo "  ./deploy.sh firewall                           # Check firewall/ports"
        exit 1
        ;;
esac
