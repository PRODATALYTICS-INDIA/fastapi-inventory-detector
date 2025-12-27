# Linux Test Server Management

This directory contains scripts and tools for managing the Linux test server deployment of the FastAPI Inventory Tracker API. These scripts automate the process of connecting to the server, uploading files, transferring Docker images, and managing deployments.

## Table of Contents

- [Overview](#overview)
- [Server Details](#server-details)
- [Prerequisites](#prerequisites)
- [Folder Organization](#folder-organization)
- [Available Scripts](#available-scripts)
- [Deployment Guide](#deployment-guide)
- [Docker Deployment](#docker-deployment)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

The `deployment/` directory provides automation scripts for deploying the FastAPI Inventory Tracker API to a remote Linux server. The scripts handle:

- **SSH Connection Management**: Secure connection to the remote server using PEM key authentication
- **File Transfer**: Automated upload of project files and Docker images
- **Docker Management**: Building, transferring, and deploying Docker containers
- **Server Maintenance**: Cleanup and system management utilities

### Purpose

These scripts streamline the deployment process by:
- Eliminating manual SSH/SCP commands
- Providing consistent deployment workflows
- Automating error handling and validation
- Enabling quick server inspection and management

---

## Server Details

### Connection Information

- **IP Address**: `13.203.145.155`
- **Username**: `ubuntu`
- **SSH Port**: `22` (default)
- **Authentication**: PEM key file (`MI_dev_Linux_test_server.pem`)
- **Default Project Directory**: `~/fastapi-inventory-detector`

### Server Requirements

- **Operating System**: Linux (Ubuntu recommended)
- **Docker**: Installed and configured (for container deployment)
- **Disk Space**: Minimum 10GB free (for Docker images and project files)
- **Network**: SSH access enabled (port 22 open)

---

## Prerequisites

### Local Machine Requirements

1. **SSH Client**: Installed and configured
   ```bash
   # Verify SSH is installed
   ssh -V
   ```

2. **Docker**: Installed (for building/transferring images)
   ```bash
   # Verify Docker is installed
   docker --version
   ```

3. **PEM Key File**: `MI_dev_Linux_test_server.pem` in the `deployment/` directory
   ```bash
   # Verify PEM file exists
   ls -la deployment/MI_dev_Linux_test_server.pem
   ```

4. **Script Permissions**: All scripts must be executable
   ```bash
   # Make scripts executable
   chmod +x deployment/*.sh
   ```

### First-Time Setup

1. **Place PEM file in deployment directory**:
   ```bash
   cp /path/to/MI_dev_Linux_test_server.pem deployment/
   chmod 400 deployment/MI_dev_Linux_test_server.pem
   ```

2. **Verify connection**:
   ```bash
   cd deployment
   ./connect.sh
   ```

3. **Install Docker on server** (if not already installed):
   ```bash
   cd deployment
   ./setup.sh docker
   ```
   
   After installation, log out and log back in (or run `newgrp docker` on the server).

---

## Folder Organization

### Directory Structure

```
deployment/
├── README.md                    # This file - comprehensive documentation
├── setup.sh                     # Server setup script (Docker installation)
├── connect.sh                   # Server connection and management script
├── deploy.sh                    # Docker image deployment script
├── Dockerfile                   # Docker build configuration
└── MI_dev_Linux_test_server.pem # SSH private key (not in git)
```

### Script Dependencies

All scripts follow a consistent pattern:
- **Location Detection**: Automatically detect script and project root directories
- **PEM File Handling**: Automatically locate and set permissions for PEM file
- **Error Handling**: Validate prerequisites before execution
- **User Feedback**: Provide clear progress indicators and error messages

### Remote Server Structure

After deployment, the server structure will be:

```
~/fastapi-inventory-detector/
├── Dockerfile                   # Docker build configuration
├── requirements.txt            # Python dependencies
├── app/                        # Application code
│   ├── main.py                # FastAPI application entry point
│   ├── config.py              # Configuration settings
│   ├── models/                # ML models (detection, OCR)
│   ├── schemas/               # Pydantic schemas
│   ├── services/              # Business logic services
│   ├── utils/                 # Utility functions
│   └── data/                  # Data files (SKU master, etc.)
└── models/                     # Model weights (YOLO segmentation model)
    └── segmentation-model.pt
```

---

## Available Scripts

This directory contains **3 scripts** that handle server setup, connection, and deployment operations.

### 1. `setup.sh` - Server Setup

**Purpose**: Install required components on the server (Docker, etc.).

**Usage**:
```bash
# Install Docker on server (default)
./setup.sh docker

# Or simply
./setup.sh
```

**What it does**:
- Checks if Docker is already installed
- Updates package index
- Installs Docker prerequisites
- Adds Docker's official GPG key
- Sets up Docker repository
- Installs Docker Engine and plugins
- Adds user to docker group
- Starts and enables Docker service
- Verifies installation

**Example**:
```bash
cd deployment
./setup.sh docker
```

**After installation**:
- Log out and log back in (or run `newgrp docker`)
- Then verify: `docker run hello-world`

---

### 2. `connect.sh` - Server Connection and Management

**Purpose**: Handle all server connection and management operations.

**Usage**:
```bash
# Interactive SSH connection (default)
./connect.sh connect

# View server contents and system info
./connect.sh view [directory]

# Cleanup server resources
./connect.sh cleanup [option]
```

**Actions**:

#### `connect` - Interactive SSH Connection
```bash
./connect.sh connect
# or simply
./connect.sh connect
```
- Establishes interactive SSH session with the server
- Validates PEM file existence and permissions
- Provides shell access to the server

#### `view` - View Server Contents
```bash
./connect.sh view                    # View home directory
./connect.sh view ~/fastapi-inventory-detector  # View specific directory
```
- Lists directory contents with detailed file information
- Displays system information (hostname, uptime, disk usage, memory)

#### `cleanup` - Cleanup Server Resources
```bash
./connect.sh cleanup project   # Remove project directory (default)
./connect.sh cleanup docker   # Remove Docker containers, images, volumes
./connect.sh cleanup all      # Remove project and Docker resources
./connect.sh cleanup system   # System cleanup (apt cache, logs)
```

**Example**:
```bash
cd deployment
./connect.sh                    # Connect to server
./connect.sh view               # View server info
./connect.sh cleanup docker     # Cleanup Docker resources
```

---

### 3. `deploy.sh` - Docker Image Deployment

**Purpose**: Quickly inspect server directory contents and system information without interactive connection.

**Usage**:
```bash
# View home directory
./connect.sh view

# View specific directory
./connect.sh view ~/fastapi-inventory-detector

# View project directory
./connect.sh view ~/fastapi-inventory-detector
```

**What it does**:
- Lists directory contents with detailed file information (`ls -lah`)
- Displays system information:
  - Hostname
  - System uptime
  - Disk usage (root filesystem)
  - Memory usage

**Technical Details**:
- Uses non-interactive SSH with inline bash script
- Expands `~` to actual home directory path
- Provides formatted output for easy reading

**Example Output**:
```
🔍 Connecting to Linux Test Server and viewing contents...
   Server: ubuntu@13.203.145.155
   Directory: ~/fastapi-inventory-detector
   =========================================

📁 Contents of: /home/ubuntu/fastapi-inventory-detector
total 48K
drwxrwxr-x 5 ubuntu ubuntu 4.0K Oct 10 10:00 .
drwxr-xr-x 8 ubuntu ubuntu 4.0K Oct 10 09:00 ..
-rw-rw-r-- 1 ubuntu ubuntu 2.1K Oct 10 10:00 Dockerfile
...

💻 System Information:
   Hostname: ip-172-31-45-123
   Uptime: up 5 days, 12 hours
   Disk Usage:
   Root: 15G used / 50G total (30% used)
   Memory Usage:
   2.1G used / 8.0G total
```

---

**Purpose**: Handle all Docker image deployment operations.

**Usage**:
```bash
# Transfer image directly to server (default)
./deploy.sh transfer [image] [port] [container]

# Push to registry then deploy
./deploy.sh registry [username] [image] [port]

# Upload source files to server
./deploy.sh upload

# Deploy container on server
./deploy.sh run [image] [port] [container]

# Test deployed service
./deploy.sh test [port]
```

**Methods**:

#### `transfer` - Direct Transfer (Default)
```bash
./deploy.sh transfer                                    # Transfer default image
./deploy.sh transfer prodatalytics/inventory-tracker-api:latest
```
- Saves Docker image as tar file locally
- Transfers tar file to server via SCP
- Loads image on server using `docker load`
- Best for single server, private deployments

#### `registry` - Registry-Based Deployment
```bash
./deploy.sh registry myusername                        # Push to Docker Hub and deploy
```
- Tags image for registry
- Pushes to Docker Hub
- Pulls and deploys on server
- Best for production, multiple servers

#### `upload` - Upload Source Files
```bash
./deploy.sh upload
```
- Uploads Dockerfile, requirements.txt, app/, models/ to server
- Use when building image on server

#### `run` - Deploy Container
```bash
./deploy.sh run                                        # Deploy existing image
./deploy.sh run prodatalytics/inventory-tracker-api:latest 8080
```
- Checks if image exists (pulls from registry if not)
- Stops and removes existing container
- Runs container with proper configuration
- Verifies service is ready

#### `test` - Test Deployed Service
```bash
./deploy.sh test                                       # Test default port
./deploy.sh test 8080                                  # Test custom port
```
- Tests health check endpoint
- Tests API documentation endpoints
- Provides test summary

**Example**:
```bash
cd deployment
./deploy.sh transfer                                   # Transfer image
./deploy.sh run                                        # Deploy container
./deploy.sh test                                       # Test service
```

**Complete Workflow**:
```bash
# Build, transfer, and deploy
docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
cd deployment
./deploy.sh transfer                                   # Transfer image
./deploy.sh run                                        # Deploy container
./deploy.sh test                                       # Test service
```

---

## Deployment Guide

### Deployment Methods Explained

There are three primary deployment methods:

1. **Direct Transfer** (Simplest for Single Server) ⭐
   - Build image locally
   - Transfer image directly to server via SCP
   - No registry needed, completely private
   - **Best for**: Single server, testing, private deployments

2. **Registry-Based Deployment** (Recommended for Production)
   - Push image to Docker Hub/registry
   - Pull and deploy on server
   - **Best for**: Production, CI/CD, multiple servers

3. **Build on Server**
   - Upload source files
   - Build Docker image on server
   - **Best for**: Large images or slow networks

**Quick Decision**:
- **Single server?** → Use **Direct Transfer** (Method 1) - It's simpler!
- **Multiple servers or production?** → Use **Registry-Based** (Method 2)

### Method 1: Direct Transfer (Simplest) ⭐

**How it works**: Directly transfers the Docker image from your local machine to the server (no registry needed).

**Complete Workflow**:

1. **Build image locally** (if not already built):
   ```bash
   cd /path/to/fastapi-inventory-detector
   docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
   ```

2. **Transfer image directly to server**:
   ```bash
   cd deployment
   ./deploy.sh transfer
   ```
   This will:
   - Save the image as a tar file
   - Transfer it to the server via SCP
   - Load it on the server
   - Clean up temporary files

3. **Deploy the container** (run the service):
   ```bash
   ./deploy.sh run
   ```
   This will:
   - Check if image exists on server ✅
   - Stop any existing container
   - Run the container with proper configuration
   - Verify the service is running
   - Show you access URLs

4. **Test the deployment**:
   ```bash
   ./deploy.sh test
   ```

5. **Access the API**:
   - API: `http://13.203.145.155:8000`
   - Docs: `http://13.203.145.155:8000/docs`
   - Health: `http://13.203.145.155:8000/health`


**Advantages**:
- ✅ **No registry needed** - No Docker Hub account required
- ✅ **Private** - Image stays private, not uploaded anywhere
- ✅ **Simple** - Direct from your machine to server
- ✅ **Works immediately** - No setup required

**Disadvantages**:
- ❌ Slower for large images (must transfer 1-3GB)
- ❌ Must transfer separately to each server

**Testing from Local Machine**:

Once deployed, you can test from your local machine:

```bash
# Health check
curl http://13.203.145.155:8000/health

# Test detection endpoint
curl -X POST \
  -F "file=@/path/to/test/image.jpg" \
  http://13.203.145.155:8000/detect-and-count

# Or use the test script
cd deployment
./deploy.sh test
```

**Access API Documentation**:
Open in browser: `http://13.203.145.155:8000/docs`

---

### Method 2: Registry-Based Deployment (For Production)

**Step-by-Step**:

1. **Build image locally** (if not already built):
   ```bash
   cd /path/to/fastapi-inventory-detector
   docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
   ```

2. **Push to registry and deploy**:
   ```bash
   cd deployment
   ./deploy.sh registry yourusername
   ```
   This will:
   - Tag image for registry
   - Push to Docker Hub
   - Pull and deploy on server
   
   Or manually:
   ```bash
   docker login
   docker tag prodatalytics/inventory-tracker-api:latest yourusername/inventory-tracker-api:latest
   docker push yourusername/inventory-tracker-api:latest
   ./deploy.sh run yourusername/inventory-tracker-api:latest
   ```

3. **Test the deployment**:
   ```bash
   ./deploy.sh test
   ```

5. **Access the API**:
   - API: `http://13.203.145.155:8000`
   - Docs: `http://13.203.145.155:8000/docs`
   - Health: `http://13.203.145.155:8000/health`

**Advantages**:
- ✅ Fastest deployment method
- ✅ No large file transfers
- ✅ Works with CI/CD pipelines
- ✅ Easy to share across multiple servers
- ✅ Version control via tags
- ✅ Can pull latest version anytime

**Disadvantages**:
- Requires Docker Hub account (or registry access)
- Image is public (unless using private registry)

**Testing from Local Machine**:

Once deployed, you can test from your local machine:

```bash
# Health check
curl http://13.203.145.155:8000/health

# Test detection endpoint
curl -X POST \
  -F "file=@/path/to/test/image.jpg" \
  http://13.203.145.155:8000/detect-and-count

# Or use the test script
cd deployment
./deploy.sh test
```

**Access API Documentation**:
Open in browser: `http://13.203.145.155:8000/docs`

---

### Method 3: Build on Server

**Step-by-Step**:

1. **Upload project files**:
   ```bash
   cd deployment
   ./deploy.sh upload
   ```

2. **Connect to server**:
   ```bash
   ./connect.sh connect
   ```

3. **On server, navigate to project directory**:
   ```bash
   cd ~/fastapi-inventory-detector
   ```

4. **Build Docker image** (Dockerfile is in the root of uploaded directory):
   ```bash
   docker build -t prodatalytics/inventory-tracker-api:latest .
   ```

5. **Verify image**:
   ```bash
   docker images prodatalytics/inventory-tracker-api:latest
   ```

6. **Run container** (with log volume mount):
   ```bash
   mkdir -p ~/fastapi-inventory-detector/logs
   docker run -d \
     --name inventory-tracker-api \
     -p 8000:8000 \
     -v ~/fastapi-inventory-detector/logs:/app/logs \
     --restart unless-stopped \
     prodatalytics/inventory-tracker-api:latest
   ```

7. **Check container status**:
   ```bash
   docker ps
   docker logs inventory-tracker-api
   ```

8. **Test API**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/docs
   ```

**Advantages**:
- No large file transfer
- Build uses server resources
- Faster for initial deployment
- Can leverage server's build cache

**Disadvantages**:
- Requires build time on server
- Server must have build dependencies

---

### Method 4: Transfer Pre-built Image (Alternative to Direct Transfer)

**Step-by-Step**:

1. **Build image locally** (if not already built):
   ```bash
   cd /path/to/fastapi-inventory-detector
   docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
   ```

2. **Transfer image to server**:
   ```bash
   cd deployment
   ./deploy.sh transfer
   ```

3. **Deploy container**:
   ```bash
   ./deploy.sh run
   ```

4. **Verify image on server**:
   ```bash
   docker images prodatalytics/inventory-tracker-api:latest
   ```

5. **Run container** (with log volume mount):
   ```bash
   mkdir -p ~/fastapi-inventory-detector/logs
   docker run -d \
     --name inventory-tracker-api \
     -p 8000:8000 \
     -v ~/fastapi-inventory-detector/logs:/app/logs \
     --restart unless-stopped \
     prodatalytics/inventory-tracker-api:latest
   ```

6. **Check container status**:
   ```bash
   docker ps
   docker logs inventory-tracker-api
   ```

**Advantages**:
- No build time on server
- Faster deployment if image exists
- Consistent image (built once)

**Disadvantages**:
- Large file transfer (1-3GB)
- Requires sufficient disk space
- Network speed dependent

---

### Deployment Methods Comparison

#### Method 1: Direct Transfer (Direct Move) ✅

**What it does**: Directly transfers the Docker image from your local machine to the server.

**How it works**:
1. **Save** the Docker image as a tar file locally (`docker save`)
2. **Transfer** the tar file to server via SCP (direct file copy)
3. **Load** the image on the server (`docker load`)
4. **Clean up** temporary tar files

**Workflow**:
```
Local Machine                    Linux Server
     |                                |
     |-- docker save -->              |
     |   (creates .tar file)          |
     |                                |
     |-- scp transfer -->             |
     |   (direct file copy)            |
     |                                |
     |                    docker load  |
     |                    (loads image)
```

**Usage**:
```bash
cd deployment
./deploy.sh transfer
```

**Advantages**:
- ✅ **No registry needed** - No Docker Hub account required
- ✅ **Private** - Image stays private, not uploaded anywhere
- ✅ **Direct** - Straight from your machine to server
- ✅ **Works offline** - No internet required (except for SSH)

**Disadvantages**:
- ❌ **Slow for large images** - Must transfer entire image (1-3GB)
- ❌ **Network dependent** - Transfer speed depends on your connection
- ❌ **One server at a time** - Must transfer separately to each server

**Best for**:
- Single server deployments
- Private/internal deployments
- When you don't want to use a registry
- Testing and development

#### Method 2: Registry-Based (Docker Hub) 🌐

**What it does**: Pushes image to Docker Hub, then pulls it on the server.

**How it works**:
1. **Push** image to Docker Hub (`docker push`)
2. **Pull** image on server from Docker Hub (`docker pull`)
3. **Run** container on server

**Workflow**:
```
Local Machine          Docker Hub          Linux Server
     |                    |                    |
     |-- docker push -->   |                    |
     |   (uploads image)   |                    |
     |                    |                    |
     |                    |-- docker pull -->   |
     |                    |   (downloads)       |
     |                    |                    |
     |                    |                    docker run
```

**Usage**:
```bash
cd deployment
./deploy.sh registry yourusername
```

**Advantages**:
- ✅ **Fast** - Docker Hub's CDN is faster than direct transfer
- ✅ **Reusable** - Same image can be pulled by multiple servers
- ✅ **Version control** - Tag different versions
- ✅ **CI/CD friendly** - Works with automated pipelines
- ✅ **Easy updates** - Just pull latest version

**Disadvantages**:
- ❌ **Requires Docker Hub account** - Need to sign up (free)
- ❌ **Public by default** - Image is public unless you pay for private repos
- ❌ **Internet required** - Both push and pull need internet

**Best for**:
- Production deployments
- Multiple servers
- CI/CD pipelines
- When you want version control

#### Comparison Table

| Feature | Direct Transfer | Registry-Based |
|---------|----------------|----------------|
| **Speed** | Slow (depends on your network) | Fast (Docker Hub CDN) |
| **Privacy** | ✅ Private | ⚠️ Public (unless paid) |
| **Registry Account** | ❌ Not needed | ✅ Required |
| **Internet Required** | Only for SSH | ✅ Yes (push & pull) |
| **Multiple Servers** | ❌ Transfer each time | ✅ Pull once per server |
| **Version Control** | ❌ Manual | ✅ Tags |
| **CI/CD** | ❌ Hard to automate | ✅ Easy |
| **Best For** | Single server, private | Production, multiple servers |

#### Quick Decision Guide

**Use Direct Transfer if**:
- ✅ You only have one server
- ✅ You want to keep the image private
- ✅ You don't have/want a Docker Hub account
- ✅ You're testing or developing

**Use Registry-Based if**:
- ✅ You have multiple servers
- ✅ You want faster deployments
- ✅ You're deploying to production
- ✅ You want version control
- ✅ You're using CI/CD

---

### Complete Deployment Workflows

#### Workflow 1: Direct Transfer + Deploy

```bash
# 1. Build image locally
cd /path/to/fastapi-inventory-detector
docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .

# 2. Transfer directly to server
cd deployment
./deploy.sh transfer

# 3. Deploy container
./deploy.sh run

# 4. Test
./deploy.sh test
```

#### Workflow 2: Registry-Based Deploy

```bash
# 1. Build image locally
cd /path/to/fastapi-inventory-detector
docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .

# 2. Push to Docker Hub and deploy
cd deployment
./deploy.sh registry yourusername

# 3. Test
./deploy.sh test
```

#### Workflow 3: Build on Server

```bash
# 1. Upload source files
cd deployment
./deploy.sh upload

# 2. Connect to server
./connect.sh connect

# 3. On server, build and run
cd ~/fastapi-inventory-detector
docker build -t prodatalytics/inventory-tracker-api:latest .
mkdir -p ~/fastapi-inventory-detector/logs
docker run -d --name inventory-tracker-api -p 8000:8000 -v ~/fastapi-inventory-detector/logs:/app/logs --restart unless-stopped prodatalytics/inventory-tracker-api:latest
```

---

### Step-by-Step Deployment Guide

#### Option A: Deploy from Local Machine (Recommended)

After transferring the image, simply run:

```bash
cd deployment
./deploy.sh transfer    # Transfer image
./deploy.sh run         # Deploy container
./deploy.sh test        # Test service
```

**Access the service:**
- API: `http://13.203.145.155:8000`
- Docs: `http://13.203.145.155:8000/docs`
- Health: `http://13.203.145.155:8000/health`

#### Option B: Deploy Manually on Server

If you prefer to deploy manually by connecting to the server:

1. **Connect to Server**:
   ```bash
   cd deployment
   ./connect.sh connect
   ```

2. **On the Server, Verify Image Exists**:
   ```bash
   docker images | grep inventory-tracker-api
   ```

3. **Stop Existing Container (if any)**:
   ```bash
   docker stop inventory-tracker-api 2>/dev/null || true
   docker rm inventory-tracker-api 2>/dev/null || true
   ```

4. **Run the Container** (with log volume mount):
   ```bash
   mkdir -p ~/fastapi-inventory-detector/logs
   docker run -d \
     --name inventory-tracker-api \
     -p 8000:8000 \
     -v ~/fastapi-inventory-detector/logs:/app/logs \
     --restart unless-stopped \
     prodatalytics/inventory-tracker-api:latest
   ```

5. **Verify Container is Running**:
   ```bash
   docker ps
   docker logs inventory-tracker-api
   ```

6. **Test the Service**:
   ```bash
   curl http://localhost:8000/health
   ```

---

### Access the Service

#### From Your Local Machine

Once deployed, you can access the service from anywhere:

**Health Check:**
```bash
curl http://13.203.145.155:8000/health
```

**API Documentation:**
Open in browser: `http://13.203.145.155:8000/docs`

**Test Detection Endpoint:**
```bash
curl -X POST \
  -F "file=@/path/to/test/image.jpg" \
  http://13.203.145.155:8000/detect-and-count
```

**Or use the test command:**
```bash
cd deployment
./deploy.sh test
```

---

### Container Management

#### View Logs

**View container logs** (stdout/stderr):
```bash
# From local machine
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 \
  "docker logs -f inventory-tracker-api"

# Or connect to server
cd deployment && ./connect.sh connect
# Then on server:
docker logs -f inventory-tracker-api
```

**View application log files** (stored on host, outside container):
```bash
# From local machine
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 \
  "tail -f ~/fastapi-inventory-detector/logs/app.log"

# Or connect to server
cd deployment && ./connect.sh connect
# Then on server:
tail -f ~/fastapi-inventory-detector/logs/app.log
cat ~/fastapi-inventory-detector/logs/app.log
ls -lh ~/fastapi-inventory-detector/logs/
```

#### Stop/Start/Restart Container
```bash
# From local machine
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 \
  "docker stop inventory-tracker-api"

# Or on server
docker stop inventory-tracker-api
docker start inventory-tracker-api
docker restart inventory-tracker-api
```

#### Remove Container
```bash
docker stop inventory-tracker-api
docker rm inventory-tracker-api
```

---

## Docker Deployment

### Container Configuration

**Image Details**:
- **Repository**: `prodatalytics/inventory-tracker-api`
- **Tag**: `latest` (or version-specific like `2.0.0`)
- **Port**: `8000` (exposed internally)
- **User**: Non-root user (`app`)
- **Health Check**: Enabled (every 30 seconds)

### Running the Container

**Basic Run** (with log volume mount):
```bash
mkdir -p ~/fastapi-inventory-detector/logs
docker run -d \
  --name inventory-tracker-api \
  -p 8000:8000 \
  -v ~/fastapi-inventory-detector/logs:/app/logs \
  --restart unless-stopped \
  prodatalytics/inventory-tracker-api:latest
```

**With Environment Variables**:
```bash
docker run -d \
  --name inventory-tracker-api \
  -p 8000:8000 \
  -e USE_GPU=true \
  -e PORT=8000 \
  prodatalytics/inventory-tracker-api:latest
```

**With GPU Support** (if available):
```bash
docker run -d \
  --name inventory-tracker-api \
  -p 8000:8000 \
  --gpus all \
  -e USE_GPU=true \
  prodatalytics/inventory-tracker-api:latest
```

**With Volume Mounts** (for persistent data and logs):
```bash
mkdir -p ~/fastapi-inventory-detector/logs
docker run -d \
  --name inventory-tracker-api \
  -p 8000:8000 \
  -v ~/fastapi-inventory-detector/logs:/app/logs \
  -v ~/inventory-data:/app/app/data \
  --restart unless-stopped \
  prodatalytics/inventory-tracker-api:latest
```

### Container Management

**Start container**:
```bash
docker start inventory-tracker-api
```

**Stop container**:
```bash
docker stop inventory-tracker-api
```

**Restart container**:
```bash
docker restart inventory-tracker-api
```

**View logs**:

**Container logs** (stdout/stderr):
```bash
# Follow logs
docker logs -f inventory-tracker-api

# Last 100 lines
docker logs --tail 100 inventory-tracker-api

# Logs with timestamps
docker logs -f -t inventory-tracker-api
```

**Application log files** (stored on host filesystem):
```bash
# View current log file
tail -f ~/fastapi-inventory-detector/logs/app.log

# View all log files (including rotated backups)
ls -lh ~/fastapi-inventory-detector/logs/

# View specific rotated log
cat ~/fastapi-inventory-detector/logs/app.log.1

# Search logs
grep "ERROR" ~/fastapi-inventory-detector/logs/app.log
```

**Note**: Log files are stored on the host filesystem (outside the container) at `~/fastapi-inventory-detector/logs/` and are automatically rotated when they reach 10MB. The deployment script automatically creates this directory and mounts it as a volume.

**Execute commands in container**:
```bash
docker exec -it inventory-tracker-api bash
```

**Remove container**:
```bash
docker stop inventory-tracker-api
docker rm inventory-tracker-api
```

### Port Mapping

The container exposes port 8000 internally. Map it to a different host port if needed:

```bash
# Map to host port 8080
docker run -d -p 8080:8000 prodatalytics/inventory-tracker-api:latest

# Access via: http://server-ip:8080
```

### Health Check

The container includes a health check that runs every 30 seconds:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' inventory-tracker-api

# View health check logs
docker inspect --format='{{json .State.Health}}' inventory-tracker-api | jq
```

### Resource Limits

**Set memory limit**:
```bash
docker run -d \
  --name inventory-tracker-api \
  -p 8000:8000 \
  --memory="2g" \
  --memory-swap="2g" \
  prodatalytics/inventory-tracker-api:latest
```

**Set CPU limit**:
```bash
docker run -d \
  --name inventory-tracker-api \
  -p 8000:8000 \
  --cpus="2.0" \
  prodatalytics/inventory-tracker-api:latest
```

---

## Technical Details

### SSH Authentication

**Key-Based Authentication**:
- Uses PEM (Privacy-Enhanced Mail) format private key
- Key must have permissions `400` (read-only for owner)
- Stored in `deployment/` directory (not committed to git)

**Security Best Practices**:
- Never commit PEM files to version control
- Use `.gitignore` to exclude PEM files
- Rotate keys periodically
- Use separate keys for different environments

### Network Configuration

**SSH Connection**:
- Protocol: SSH v2
- Port: 22 (default)
- Encryption: AES-256 (or server-supported cipher)
- Compression: Optional (can be enabled for slow networks)

**SCP Transfer**:
- Protocol: SCP over SSH
- Compression: Not used by default
- Resume: Not supported
- Progress: Not shown by default

**Docker Image Transfer**:
- Format: Docker image tar archive
- Compression: Docker's internal compression
- Size: Typically 1.5-3GB
- Transfer time: 5-30 minutes (network dependent)

### File Permissions

**Script Permissions**:
```bash
chmod +x deployment/*.sh  # Executable for all scripts
```

**PEM File Permissions**:
```bash
chmod 400 deployment/MI_dev_Linux_test_server.pem  # Read-only for owner
```

**Remote Directory Permissions**:
- Created with default umask
- Typically `755` for directories
- `644` for files

### Error Handling

All scripts include error handling:
- **Validation**: Check prerequisites before execution
- **Graceful Failure**: Exit with clear error messages
- **Cleanup**: Remove temporary files on error
- **User Feedback**: Provide progress indicators

### Performance Considerations

**Upload Performance**:
- SCP is single-threaded
- Large directories take time
- Network speed is the bottleneck

**Docker Build Performance**:
- Multi-stage build reduces final image size
- Build cache speeds up subsequent builds
- Server CPU/memory affects build time

**Image Transfer Performance**:
- Network bandwidth is the limiting factor
- Compression reduces transfer size but increases CPU usage
- Consider using Docker registry for faster transfers

---

## Troubleshooting

### Connection Issues

#### Permission Denied Error

**Symptoms**:
```
Permission denied (publickey)
```

**Solutions**:
1. Check PEM file permissions:
   ```bash
   chmod 400 deployment/MI_dev_Linux_test_server.pem
   ```

2. Verify PEM file location:
   ```bash
   ls -la deployment/MI_dev_Linux_test_server.pem
   ```

3. Check username:
   ```bash
   # Should be 'ubuntu' for this server
   ```

#### Host Key Verification Failed

**Symptoms**:
```
Host key verification failed
```

**Solution**:
```bash
# Remove old host key
ssh-keygen -R 13.203.145.155

# Or remove from known_hosts manually
ssh-keygen -R 13.203.145.155 -f ~/.ssh/known_hosts
```

#### Connection Timeout

**Symptoms**:
```
Connection timed out
```

**Solutions**:
1. Check network connectivity:
   ```bash
   ping 13.203.145.155
   ```

2. Verify server IP is correct

3. Check firewall/security groups:
   - Ensure port 22 (SSH) is open
   - Check AWS security groups (if applicable)

4. Verify server is running:
   ```bash
   # Try connecting from different network
   ```

### Docker Issues

#### Docker Not Found on Server

**Symptoms**:
```
docker: command not found
```

**Solution**:
```bash
# Install Docker on server
cd deployment
./setup.sh docker

# After installation, log out and log back in (or run 'newgrp docker' on server)
# Then verify: docker run hello-world
```

#### Image Transfer Fails

**Symptoms**:
```
Error: Failed to transfer image to server
```

**Solutions**:
1. Check disk space on server:
   ```bash
   ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 "df -h"
   ```

2. Check disk space locally:
   ```bash
   df -h .
   ```

3. Verify network connection:
   ```bash
   ping 13.203.145.155
   ```

4. Try transferring in smaller chunks or use Docker registry

#### Container Won't Start

**Symptoms**:
```
Container exits immediately
```

**Solutions**:
1. Check container logs:
   ```bash
   docker logs inventory-tracker-api
   ```

2. Check container status:
   ```bash
   docker ps -a
   docker inspect inventory-tracker-api
   ```

3. Verify port is not in use:
   ```bash
   netstat -tuln | grep 8000
   ```

4. Check resource limits:
   ```bash
   docker stats inventory-tracker-api
   ```

### File Transfer Issues

#### Upload Fails Partially

**Symptoms**:
```
Some files uploaded, others failed
```

**Solutions**:
1. Re-run upload script (idempotent):
   ```bash
   ./deploy.sh upload
   ```

2. Check server disk space:
   ```bash
   ./connect.sh view
   ```

3. Verify file permissions:
   ```bash
   ./connect.sh
   ls -la ~/fastapi-inventory-detector
   ```

#### Large File Transfer Timeout

**Symptoms**:
```
Connection lost during transfer
```

**Solutions**:
1. Increase SSH timeout:
   ```bash
   # Edit script to add: -o ServerAliveInterval=60
   ```

2. Use compression for slow networks:
   ```bash
   # Add -C flag to scp commands
   ```

3. Transfer in smaller batches
4. Use Docker registry instead

### Common Errors

#### PEM File Not Found

**Error**:
```
❌ Error: PEM file not found at '...'
```

**Solution**:
- Ensure `MI_dev_Linux_test_server.pem` is in `deployment/` directory
- Or provide full path: `./connect.sh /full/path/to/pem`

#### Docker Image Not Found

**Error**:
```
❌ Error: Docker image '...' not found locally!
```

**Solution**:
- Build image first:
  ```bash
  docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
  ```
- Or check available images:
  ```bash
  docker images
  ```

#### Insufficient Disk Space

**Error**:
```
No space left on device
```

**Solution**:
1. Clean up Docker resources:
   ```bash
   ./connect.sh cleanup docker
   ```

2. Clean up project files:
   ```bash
   ./connect.sh cleanup project
   ```

3. System cleanup:
   ```bash
   ./connect.sh cleanup system
   ```

---

## Best Practices

### Security

1. **Never commit PEM files**:
   - Add to `.gitignore`
   - Use environment-specific keys
   - Rotate keys periodically

2. **Use least privilege**:
   - Run containers as non-root user
   - Limit container capabilities
   - Use read-only filesystems where possible

3. **Secure connections**:
   - Use SSH key authentication only
   - Disable password authentication
   - Use VPN for sensitive deployments

### Deployment

1. **Version control**:
   - Tag Docker images with versions
   - Keep deployment scripts in version control
   - Document deployment procedures

2. **Testing**:
   - Test deployments on staging first
   - Verify health checks after deployment
   - Monitor logs for errors

3. **Rollback plan**:
   - Keep previous image versions
   - Document rollback procedures
   - Test rollback process

### Performance

1. **Optimize transfers**:
   - Use Docker registry for large images
   - Compress files before transfer
   - Use rsync for incremental updates

2. **Resource management**:
   - Set appropriate resource limits
   - Monitor container resource usage
   - Clean up unused resources regularly

3. **Build optimization**:
   - Use multi-stage builds
   - Leverage Docker build cache
   - Minimize image layers

### Maintenance

1. **Regular cleanup**:
   ```bash
   # Weekly cleanup
   ./connect.sh cleanup docker
   ```

2. **Monitor disk space**:
   ```bash
   # Check regularly
   ./connect.sh view
   ```

3. **Update dependencies**:
   - Keep Docker updated
   - Update base images regularly
   - Review security advisories

### Documentation

1. **Keep scripts documented**:
   - Comment complex logic
   - Document parameters
   - Include examples

2. **Maintain README**:
   - Update when procedures change
   - Document known issues
   - Include troubleshooting steps

3. **Version information**:
   - Document script versions
   - Track server configurations
   - Maintain change log

---

## Quick Reference

### Common Commands

**Direct Transfer (Simplest for Single Server)**:
```bash
# Build, transfer, and deploy
cd /path/to/project
docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
cd deployment
./deploy.sh transfer    # Transfer image
./deploy.sh run         # Deploy container
./deploy.sh test        # Test service
```

**Registry-Based Deployment (For Production/Multiple Servers)**:
```bash
# Build, push, and deploy
cd /path/to/project
docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .
cd deployment
./deploy.sh registry yourusername  # Push to registry and deploy
./deploy.sh test                   # Test service
```

**Server Setup**:
```bash
# Install Docker on server
cd deployment && ./setup.sh docker
```

**Server Management**:
```bash
# Connect to server
cd deployment && ./connect.sh connect

# View server contents
cd deployment && ./connect.sh view

# Cleanup Docker resources
cd deployment && ./connect.sh cleanup docker

# Full cleanup
cd deployment && ./connect.sh cleanup all
```

**Deployment**:
```bash
# Upload project files
cd deployment && ./deploy.sh upload

# Transfer Docker image directly
cd deployment && ./deploy.sh transfer

# Deploy container
cd deployment && ./deploy.sh run

# Test service
cd deployment && ./deploy.sh test
```

### Docker Commands

**Local Development**:
```bash
# Build image
docker build -f deployment/Dockerfile -t prodatalytics/inventory-tracker-api:latest .

# Run container locally
docker run -d -p 8000:8000 prodatalytics/inventory-tracker-api:latest

# View logs
docker logs -f inventory-tracker-api

# Stop container
docker stop inventory-tracker-api

# Remove container
docker rm inventory-tracker-api
```

**Registry Operations**:
```bash
# Login to Docker Hub
docker login

# Tag image for registry
docker tag prodatalytics/inventory-tracker-api:latest username/inventory-tracker-api:latest

# Push to registry
docker push username/inventory-tracker-api:latest

# Pull from registry
docker pull username/inventory-tracker-api:latest
```

**Server Operations** (via SSH):
```bash
# Pull image on server
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 \
  "docker pull username/inventory-tracker-api:latest"

# Run container on server
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 \
  "docker run -d -p 8000:8000 --name inventory-tracker-api username/inventory-tracker-api:latest"

# View logs on server
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 \
  "docker logs -f inventory-tracker-api"
```

### Server Commands (via SSH)

```bash
# Check Docker status
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 "docker ps"

# View disk usage
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 "df -h"

# View memory usage
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 "free -h"

# Check system load
ssh -i deployment/MI_dev_Linux_test_server.pem ubuntu@13.203.145.155 "uptime"

# Test API from local machine
curl http://13.203.145.155:8000/health
curl http://13.203.145.155:8000/docs
```

---

## Support

For issues or questions:
1. Check this README first
2. Review troubleshooting section
3. Check server logs: `docker logs inventory-tracker-api`
4. Verify server status: `./connect.sh view`

---

**Last Updated**: October 2024  
**Version**: 2.0.0
