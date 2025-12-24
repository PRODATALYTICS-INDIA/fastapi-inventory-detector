# 🚀 FastAPI Inventory Detector

REST API service for automated inventory monitoring using computer vision. This service provides endpoints for processing images and videos to detect and count items using YOLOv11 detection and segmentation models.

**Version 2.0 - Now with OCR-Based SKU Identification!** 🎯

**New Features:**
- ✅ **OCR Text Extraction** - PaddleOCR for reading product labels
- ✅ **SKU Fuzzy Matching** - Intelligent SKU identification from OCR text
- ✅ **Dual-Model Pipeline** - YOLO detection + OCR extraction + SKU matching
- ✅ **Modular Architecture** - Clean, scalable project structure
- ✅ **YOLOv11 Segmentation Support** - Pixel-accurate masks

The system automatically detects whether you're using detection or segmentation models and adjusts visualization accordingly.

---

## 🚀 Tech Stack

- **Python 3.11+** - Core language
- **FastAPI** - REST API framework
- **YOLOv11 / YOLOv8** - Object detection and segmentation models
  - ✅ Detection models (bounding boxes)
  - ✅ **Segmentation models (pixel-accurate masks)**
- **PaddleOCR** - OCR text extraction from product images
- **RapidFuzz** - Fuzzy string matching for SKU identification
- **Ultralytics** - YOLO implementation
- **Supervision** - Object tracking (ByteTrack) and annotation
- **OpenCV** - Video/image processing
- **Docker** - Containerized deployment

---

## 📊 Features

### Core Functionality
- ✅ REST API endpoints for image and video processing
- ✅ Support for multiple pre-trained/fine-tuned YOLO models
- ✅ Real-time object detection and tracking with ByteTrack
- ✅ Confidence values and frame presence statistics
- ✅ Base64-encoded annotated images in responses
- ✅ Health check and model information endpoints

### YOLOv11 Segmentation Support
- ✅ **Pixel-accurate segmentation masks** (colored overlays on detected objects)
- ✅ **Automatic model type detection** (detection vs segmentation)
- ✅ **Backward compatible** with all existing detection models
- ✅ Same API, same response format - seamless integration

### OCR-Based SKU Identification (NEW in v2.0)
- ✅ **PaddleOCR Integration** - Extract text from product crops
- ✅ **Fuzzy SKU Matching** - Match OCR text to SKU master list using RapidFuzz
- ✅ **Dual-Model Pipeline** - YOLO detection → OCR extraction → SKU matching
- ✅ **Batch Processing** - Parallel OCR on multiple product crops
- ✅ **GPU Acceleration** - Optional GPU support for OCR
- ✅ **Configurable Thresholds** - Adjustable confidence and matching thresholds

---

## 📦 Setup & Installation

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd fastapi-inventory-detector

# 2. Set up environment and install dependencies
./setup_environment.sh

# 3. Start the FastAPI service
./start_fastapi.sh
```

That's it! The service will be available at http://localhost:8000

---

### Detailed Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd fastapi-inventory-detector
```

### 2. Environment Setup

**Automated Setup (Recommended):**

Use the provided setup script which automatically creates a virtual environment and installs all dependencies:

```bash
# Make script executable (if needed)
chmod +x setup_environment.sh

# Run the setup script (uses uv if available, falls back to standard venv/pip)
./setup_environment.sh
```

The script will:
- ✅ Check for `uv` (fast Python package manager) and use it if available
- ✅ Create virtual environment automatically
- ✅ Install all dependencies from `requirements.txt`
- ✅ Verify the installation

**Manual Setup (Alternative):**

If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Installing uv (Optional but Recommended):**

For faster dependency installation, install `uv`:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew (macOS)
brew install uv
```

The setup script will automatically detect and use `uv` if it's installed.

### 3. Dependencies

The `requirements.txt` includes:
- FastAPI and uvicorn for the API server
- Ultralytics and supervision for ML models
- **PaddleOCR** - OCR text extraction
- **RapidFuzz** - Fuzzy string matching
- OpenCV and Pillow for image processing
- PyTorch for model inference

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Note**: PaddleOCR installation may take a few minutes as it downloads model files on first use.

---

## 🚀 Usage

### Start the Server

**Quick Start (Recommended):**

Use the provided startup script which automatically handles environment activation and starts the service:

```bash
# Make script executable (if needed)
chmod +x start_fastapi.sh

# Start the FastAPI service (automatically activates venv if needed)
./start_fastapi.sh
```

The script will:
- ✅ Check and set up virtual environment if missing
- ✅ Activate the virtual environment
- ✅ Verify required files (e.g., `app/data/sku_master.json`)
- ✅ Start the FastAPI server with uvicorn

**Manual Start:**

**New Application (v2.0):**
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using uvicorn (recommended)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# With GPU support for OCR
USE_GPU=true uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Legacy Application (still supported):**
```bash
# Direct Python
python main.py

# Or using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access points:
- 🌐 API: http://localhost:8000
- 📖 Interactive Docs: http://localhost:8000/docs
- 📚 ReDoc: http://localhost:8000/redoc

---

## 📡 API Endpoints

### 1. Health Check

```bash
GET http://localhost:8000/
GET http://localhost:8000/health
```

**Response (v2.0):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ocr_loaded": true,
  "timestamp": "2025-11-14T10:30:45.123456"
}
```

### 1.1. Detect and Identify (NEW - v2.0) ⭐

**Main endpoint with OCR and SKU matching:**

```bash
curl -X POST "http://localhost:8000/detect-and-identify" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "enable_ocr=true" \
  -F "enable_sku_matching=true"
```

**Response:**
```json
{
  "success": true,
  "filename": "test_image.jpg",
  "products": [
    {
      "yolo_class": "Sku4_Colgate Paste",
      "sku": "Sku4_Colgate Paste",
      "sku_name": "colgate-paste",
      "confidence": 0.91,
      "yolo_confidence": 0.91,
      "sku_match_confidence": 0.85,
      "bbox": [100, 200, 300, 400],
      "mask": [[...]],
      "ocr_text": "colgate visible white 100g",
      "ocr_details": [
        {
          "text": "colgate",
          "confidence": 0.95,
          "bbox": [[...]]
        }
      ],
      "tracker_id": 1,
      "class_id": 3
    }
  ],
  "total_detections": 1,
  "processing_time_ms": 2345.6,
  "model_version": "25.11.0",
  "app_version": "2.0.0",
  "timestamp": "2025-11-14T10:30:45.123456"
}
```
```

**Parameters:**
- `file` (required): Image file
- `confidence_threshold` (optional, default: 0.5): YOLO confidence threshold
- `enable_ocr` (optional, default: true): Enable OCR text extraction
- `enable_sku_matching` (optional, default: true): Enable SKU fuzzy matching

### 2. Image Prediction (Legacy)

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "label_mode=item_name"
```

**Response:**
```json
{
  "success": true,
  "filename": "test_image.jpg",
  "confidence_threshold": 0.5,
  "label_mode": "item_name",
  "frame_count": 1,
  "detections": {
    "Product A": {
      "count": 5,
      "confidence_avg": 85.3,
      "confidence_min": 72.1,
      "confidence_max": 94.5
    },
    "Product B": {
      "count": 3,
      "confidence_avg": 91.2,
      "confidence_min": 88.5,
      "confidence_max": 95.0
    }
  },
  "statistics": [
    {
      "item_name": "Product A",
      "count": 5,
      "confidence(%)": "85",
      "frame_presence(%)": "100"
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "timestamp": "2025-11-01T10:30:45.123456"
}
```

**Note**: For segmentation models, `annotated_image` includes colored masks!

### 3. Video Prediction

```bash
curl -X POST "http://localhost:8000/predict/video" \
  -F "file=@video.mp4" \
  -F "confidence_threshold=0.5" \
  -F "label_mode=item_name" \
  -F "max_frames=100"
```

**Response:**
```json
{
  "success": true,
  "filename": "video.mp4",
  "confidence_threshold": 0.5,
  "label_mode": "item_name",
  "total_frames": 900,
  "processed_frames": 100,
  "detections": {
    "Product A": 5,
    "Product B": 3
  },
  "statistics": [
    {
      "item_name": "Product A",
      "count": 5,
      "confidence(%)": "87",
      "frame_presence(%)": "45"
    }
  ],
  "timestamp": "2025-11-01T10:35:20.123456"
}
```

### 4. Model Information

```bash
GET http://localhost:8000/model/info
```

**Response (v2.0):**
```json
{
  "yolo_model_path": "models/yolo-segmentation.pt",
  "yolo_model_name": "yolo-segmentation.pt",
  "yolo_model_version": "25.11.0",
  "yolo_is_segmentation": true,
  "ocr_available": true,
  "sku_master_count": 7,
  "app_version": "2.0.0"
}
```

**Legacy Response (v1.0):**
```json
{
  "model_path": "models/segmentation-model.pt",
  "label_mode": "item_name",
  "valid_label_modes": ["sku_code", "item_name", "brand", "sub_category", "category"],
  "confidence_threshold": 0.0,
  "model_loaded": true
}
```

### 5. Update Label Mode

```bash
curl -X POST "http://localhost:8000/model/update-label-mode" \
  -F "label_mode=brand"
```

**Response:**
```json
{
  "success": true,
  "label_mode": "brand",
  "message": "Label mode updated to: brand"
}
```

---

## 🏗️ Project Structure

### New Structure (v2.0)

```
fastapi-inventory-detector/
├── app/                          # New application package
│   ├── main.py                  # FastAPI application (NEW)
│   ├── models/                  # Model wrappers
│   │   ├── yolo_model.py        # YOLO model wrapper
│   │   └── ocr_model.py         # PaddleOCR wrapper
│   ├── services/                # Business logic services
│   │   ├── detection_service.py # YOLO detection pipeline
│   │   ├── ocr_service.py       # OCR text extraction
│   │   └── sku_matching_service.py # SKU fuzzy matching
│   ├── utils/                   # Utility functions
│   │   └── image_utils.py       # Image preprocessing & crop extraction
│   ├── schemas/                 # Pydantic models
│   │   └── response.py          # API response schemas
│   └── data/                    # Data files
│       └── sku_master.json      # SKU master list (JSON)
├── main.py                      # Legacy FastAPI app (still works)
├── py/                          # Legacy code (still works)
│   └── InventoryTracker.py     # Original tracker class
├── models/                      # Model weights
│   ├── segmentation-model.pt  # Generic model name (recommended)
│   └── ...                      # Model files
├── test/                        # Test suite
│   ├── test_api.py              # API testing
│   ├── test_api_curl.sh         # API testing (curl)
│   ├── test_ocr_detection.py    # OCR testing
│   ├── test_ocr_local.py        # Local OCR testing
│   ├── test_ocr_batch.sh        # Batch OCR testing
│   ├── test_new_structure.py    # Structure verification
│   ├── test_images/            # Test images
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── setup_environment.sh        # Environment setup script
├── start_fastapi.sh            # FastAPI startup script
├── documents/                   # Documentation
│   ├── IMPLEMENTATION_GUIDE.md  # Complete implementation guide
│   └── README.md               # Documentation index
└── README.md                    # This file
```

### Legacy Structure (v1.0 - Still Supported)

The original structure remains functional for backward compatibility.

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -f Dockerfile -t fastapi-inventory-detector:latest .

# Run the container
docker run -d -p 8000:8000 fastapi-inventory-detector:latest

# Access API at http://localhost:8000
```

### Docker Commands

```bash
# Build with tags
docker build -f Dockerfile -t fastapi-inventory-detector:25.11.01 -t fastapi-inventory-detector:latest .

# Push to registry
docker push fastapi-inventory-detector:25.11.01
docker push fastapi-inventory-detector:latest

# Stop and remove container
docker stop inventory-fastapi && docker rm inventory-fastapi
```

---

## ⚡ Performance

### Inference Speed (CPU)
- **YOLO Detection**: ~50-80ms per frame
- **YOLO Segmentation**: ~60-95ms per frame (+10-15% slower)
- **OCR per crop**: ~0.5-2 seconds (depends on image size)
- **SKU Matching**: <10ms per product

### Memory Usage
- **YOLO Detection Model**: ~500MB GPU VRAM
- **YOLO Segmentation Model**: ~600MB GPU VRAM (+20% higher)
- **PaddleOCR**: ~200-500MB RAM (CPU) or ~1GB VRAM (GPU)

### Processing Pipeline
- **Single Image**: 2-5 seconds (YOLO + OCR + Matching)
- **Batch OCR**: Parallel processing on multiple crops
- **Video Processing**: 
  - Frame skip: Every 5th frame (configurable)
  - Max frames: 100 processed frames (configurable via API)
  - Example: 30-second video @ 30fps = 900 frames → 180 processed frames

### Optimization Features
- ✅ **Lazy Loading**: OCR model loads on first use
- ✅ **Multiprocessing**: Parallel OCR on multiple crops
- ✅ **GPU Acceleration**: Optional GPU support for OCR
- ✅ **Batch Processing**: Efficient YOLO batch inference

---

## 🔧 Configuration

### Model Selection

**Model Configuration:**
The model configuration is managed in `app/config.py`:

```python
MODEL_BASE_NAME = "segmentation-model.pt"  # Generic model name
MODEL_VERSION = "25.11.0"  # Current model version
```

**Model File:**
- **Model file**: `models/segmentation-model.pt` (required)

**Using a Different Model:**

1. **Replace the model file:**
   ```bash
   mv your-model.pt models/segmentation-model.pt
   ```

2. **Use environment variable:**
   ```bash
   export MODEL_PATH=models/your-model.pt
   ```

3. **Update config.py:**
   Edit `app/config.py` and update `MODEL_BASE_NAME` and `MODEL_VERSION`

**Model Version Information:**
- Model version is tracked in `app/config.py`
- Version appears in API responses (`/model/info` and `/detect-and-identify`)
- **Current model version: 25.11.0**

### Environment Variables

**GPU Support for OCR:**
```bash
export USE_GPU=true
uvicorn app.main:app --reload
```

**API Base URL (for testing):**
```bash
export API_BASE_URL=http://localhost:8000
```

### Label Modes

Available modes: `item_name`, `category`, `sub_category`, `brand`, `sku_code`

Change via API endpoint or update in code.

### Confidence Thresholds

**YOLO Confidence:**
- Lower threshold (0.3) = more detections (more false positives)
- Higher threshold (0.7) = fewer detections (more accurate)

**SKU Matching Confidence:**
- Default: 0.65 (65%)
- Configurable in `app/services/sku_matching_service.py`
- Lower = more matches, Higher = stricter matching

---

## 🧪 Testing

### Quick Test with curl

```bash
# Health check
curl http://localhost:8000/health

# New endpoint with OCR (v2.0)
curl -X POST "http://localhost:8000/detect-and-identify" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "enable_ocr=true" \
  -F "enable_sku_matching=true"

# Legacy image prediction
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5"

# Model info
curl http://localhost:8000/model/info
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# New endpoint with OCR (v2.0)
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "confidence_threshold": 0.5,
        "enable_ocr": "true",
        "enable_sku_matching": "true"
    }
    response = requests.post(
        "http://localhost:8000/detect-and-identify",
        files=files,
        data=data,
        timeout=180
    )
    result = response.json()
    print(f"Detected {result['total_detections']} products")
    for product in result['products']:
        print(f"  - {product['sku']}: {product['ocr_text']}")

# Legacy image prediction
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    data = {"confidence_threshold": 0.5, "label_mode": "item_name"}
    response = requests.post("http://localhost:8000/predict/image", files=files, data=data)
    result = response.json()
    print(f"Detected {len(result['detections'])} item types")
```

### Comprehensive Test Suite

**Test OCR Detection:**
```bash
cd test
python test_ocr_detection.py
```

**Test OCR Locally (No API):**
```bash
cd test
python test_ocr_local.py --image test_images/20251010-1.jpeg
```

**Batch Test All Images:**
```bash
cd test
./test_ocr_batch.sh
```

See `documents/TEST_DOCUMENTATION.md` for comprehensive test documentation.

---

## 🔧 Troubleshooting

### Issue: Model Not Loading

**Solution**: Ensure model file exists in `models/` directory and path in `app/main.py` (v2.0) or `main.py` (v1.0) is correct.

### Issue: OCR Not Available

**Symptoms**: `PaddleOCR not available` error

**Solution**: 
```bash
pip install paddlepaddle paddleocr
```

**Note**: First OCR run will download model files (~100MB), which may take a few minutes.

### Issue: SKU Matching Not Working

**Symptoms**: No SKU matches in response

**Solutions**:
1. Check SKU master file exists: `app/data/sku_master.json`
2. Verify OCR text is being extracted (check `ocr_text` field)
3. Lower SKU matching confidence threshold in `sku_matching_service.py`
4. Ensure SKU keywords match OCR text patterns

### Issue: Import Errors

**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: Slow Performance

**Solutions**:
- Use GPU if available: `USE_GPU=true`
- Reduce image resolution before processing
- Increase frame skip for videos
- Disable OCR if not needed: `enable_ocr=false`
- Reduce batch size for OCR processing

### Issue: API Timeout

**Solutions**:
- Increase timeout in client requests (default: 180s)
- Reduce number of images in batch tests
- Check server logs for errors
- Verify models are loaded correctly

---

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Supervision Library](https://roboflow.github.io/supervision/)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

### Project Documentation
All documentation files are located in the `documents/` directory:
- `documents/IMPLEMENTATION_GUIDE.md` - **Complete implementation guide** covering:
  - Business logic and system architecture
  - Model information and configuration
  - Catalog and SKU management
  - Technical implementation details
- `test/README.md` - General testing guide
- `documents/TEST_DOCUMENTATION.md` - Comprehensive test documentation
- `documents/README.md` - Documentation index

### Key Features
- New `/detect-and-identify` endpoint with OCR and SKU matching
- Modular project structure in `app/` directory
- SKU master data in JSON format (`app/data/sku_master.json`)
- Model-agnostic architecture for easy migration
- Enhanced testing suite

---

## 📄 License

This project is licensed under the MIT License.

---

## 🎯 Version History

### v2.0.0 (Current)
- ✅ OCR-based SKU identification
- ✅ PaddleOCR integration
- ✅ Fuzzy SKU matching with RapidFuzz
- ✅ Modular project structure
- ✅ Enhanced testing suite
- ✅ Performance optimizations
- ✅ Model versioning system
- ✅ Generic model naming

**Model Version**: 25.11.0

### v1.0.0
- ✅ YOLO detection and segmentation
- ✅ ByteTrack object tracking
- ✅ Legacy API endpoints

---

**Ready to deploy your inventory detection API with OCR! 🚀**

