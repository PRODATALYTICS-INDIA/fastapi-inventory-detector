# FastAPI Inventory Detector - Implementation Guide

## Table of Contents
1. [Business Logic & Purpose](#business-logic--purpose)
2. [System Architecture](#system-architecture)
3. [Model Information](#model-information)
4. [Catalog & SKU Management](#catalog--sku-management)
5. [Technical Implementation](#technical-implementation)
6. [Configuration](#configuration)

---

## Business Logic & Purpose

### Overview
The FastAPI Inventory Detector is a REST API service for automated inventory monitoring using computer vision. It combines object detection/segmentation with OCR text extraction to identify and count products in images and videos.

### Core Functionality
1. **Product Detection**: Uses YOLO models to detect and localize products in images/videos
2. **Text Extraction**: Extracts text from product labels using PaddleOCR
3. **SKU Identification**: Matches OCR text to SKU master list using fuzzy matching
4. **Unified Pipeline**: Combines detection, OCR, and matching in a single endpoint

### Processing Pipeline
```
Image/Video Input
    ↓
YOLO Detection/Segmentation (bounding boxes + masks)
    ↓
Crop Extraction (individual product images)
    ↓
OCR Text Extraction (PaddleOCR)
    ↓
SKU Fuzzy Matching (RapidFuzz)
    ↓
Unified Response (detection + identification)
```

---

## System Architecture

### Project Structure
```
fastapi-inventory-detector/
├── app/                          # Main application package
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration and model paths
│   ├── models/                  # Model wrappers
│   │   └── detection_model.py   # Model-agnostic detection wrapper
│   ├── services/                # Business logic services
│   │   ├── detection_service.py # Detection pipeline
│   │   ├── ocr_service.py       # OCR text extraction
│   │   ├── sku_matching_service.py # SKU fuzzy matching
│   │   └── catalog_service.py   # SKU catalog management
│   ├── utils/                   # Utility functions
│   │   └── image_utils.py       # Image preprocessing & crop extraction
│   ├── schemas/                 # Pydantic models
│   │   └── response.py          # API response schemas
│   └── data/                    # Data files
│       └── sku_master.json      # SKU master list (JSON)
├── models/                      # Model weights
│   └── segmentation-model.pt    # Detection/segmentation model
├── requirements.txt             # Python dependencies
└── start_fastapi.sh            # Startup script
```

### Model-Agnostic Architecture

The system is designed to be **model-agnostic**, allowing easy migration to different model backends without changing the API or business logic.

**Key Abstraction:**
- `DetectionModel` class provides a unified interface for any detection/segmentation model
- Currently uses Ultralytics/YOLO, but can be swapped for other models (e.g., Facebook SAM, Detectron2)
- Business logic is separated from model implementation

**Benefits:**
- Easy migration to better models in the future
- Clean architecture with separation of concerns
- Backward compatible API responses

### Service Architecture

**DetectionService:**
- Handles YOLO detection/segmentation inference
- Extracts bounding boxes and masks
- Integrates with CatalogService for SKU metadata lookup
- Filters detections based on size and SKU validation

**OCRService:**
- Wraps PaddleOCR for text extraction
- Supports batch processing with multiprocessing
- GPU acceleration support
- Lazy loading (loads on first use)

**SKUMatchingService:**
- Fuzzy matching using RapidFuzz
- Keyword indexing for fast lookups
- Configurable confidence thresholds
- Supports both catalog service and JSON fallback

**CatalogService:**
- Manages SKU catalog data
- Scales from small JSON files to large datasets using DuckDB
- Provides efficient querying interface
- Supports both `code` and `sku_code` field names for backward compatibility

---

## Model Information

### Model Configuration

**Model File Structure:**
```
models/
└── segmentation-model.pt    # Generic model name (required)
```

**Model Resolution Logic:**
1. Environment variable: `MODEL_PATH` (if set and exists)
2. Generic model: `models/segmentation-model.pt` (default)

**Configuration in `app/config.py`:**
```python
MODEL_BASE_NAME = "segmentation-model.pt"  # Generic, model-agnostic name
MODEL_VERSION = "25.11.0"  # Current model version
```

### Model Types Supported

**Detection Models:**
- Bounding boxes only
- Faster inference (~50-80ms per frame)
- Lower memory usage (~500MB GPU VRAM)

**Segmentation Models:**
- Pixel-accurate masks + bounding boxes
- Better accuracy for overlapping objects
- Slightly slower (~60-95ms per frame)
- Higher memory usage (~600MB GPU VRAM)

The system automatically detects model type and adjusts processing accordingly.

### Model Metadata

Model information is exposed via API:
```json
{
  "yolo_model_path": "models/segmentation-model.pt",
  "yolo_model_name": "segmentation-model.pt",
  "yolo_model_version": "25.11.0",
  "yolo_is_segmentation": true,
  "detectable_classes": [
    {"class_id": 0, "class_name": "Sku1_Dettol Soap"},
    {"class_id": 1, "class_name": "Sku2_Boost"},
    ...
  ]
}
```

### Updating Models

**Method 1: Replace Generic Model (Recommended)**
```bash
# Backup old model
mv models/segmentation-model.pt models/segmentation-model.pt.backup

# Copy new model
cp new-model.pt models/segmentation-model.pt

# Update version in app/config.py
```

**Method 2: Use Environment Variable**
```bash
export MODEL_PATH=models/new-model.pt
```

---

## Catalog & SKU Management

### SKU Master Data Structure

**File Location:** `app/data/sku_master.json`

**Schema:**
```json
[
  {
    "id": 0,
    "code": "Sku1_Dettol Soap",
    "item_name": "dettol-soap",
    "brand": "Dettol",
    "sub_category": "bar-soap",
    "category": "personal-hygiene",
    "detection_keywords": ["dettol", "dettol soap", "dettol bar soap", ...],
    "price_per_unit": "-",
    "quantity_per_unit": "-",
    "upc": null,
    "additional_info": "-"
  }
]
```

**Key Fields:**
- `id`: Numeric ID used by detection model (class_id)
- `code`: SKU code (matches model class names)
- `detection_keywords`: Brand-specific keywords for OCR matching
- Other metadata: brand, category, sub_category, etc.

### Catalog Service Architecture

**Storage Strategy:**
- **Small datasets (< 10,000 SKUs)**: In-memory dictionary lookup (~1μs)
- **Medium datasets (10k-100k SKUs)**: DuckDB database (~1-5ms lookups)
- **Large datasets (200k+ SKUs)**: DuckDB with indexes (~5-20ms lookups)

**Why DuckDB?**
- Optimized for analytical queries
- Columnar storage for better compression
- Fast aggregations and joins
- Embedded database (no server required)
- Seamless pandas integration

**Database Location:**
- Default: `app/data/catalog.duckdb`
- Auto-created when dataset >= 1,000 SKUs
- Git-ignored (regenerated on startup)

### SKU Matching Logic

**Fuzzy Matching Algorithm:**
1. **Keyword Indexing**: Extracts keywords from item_name, brand, category, code
2. **Multiple Scoring Methods**:
   - Partial ratio (substring matching)
   - Token sort ratio (word order independence)
   - Token set ratio (partial matches)
3. **Weighted Combination**: Combines scores with weights
4. **Number Matching**: Matches sizes/weights from OCR text
5. **Confidence Threshold**: Default 65%, configurable

**Detection Keywords:**
- Brand-specific combinations (e.g., "dettol soap", "cadbury dairy milk")
- Constrained to avoid false matches
- Used for OCR text matching

---

## Technical Implementation

### API Endpoints

**Main Endpoint: `/detect-and-identify`**
```bash
POST /detect-and-identify
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- confidence_threshold: Detection confidence (default: 0.5)
- enable_ocr: Enable OCR processing (default: true)
- enable_sku_matching: Enable SKU matching (default: true)
```

**Response Format:**
```json
{
  "success": true,
  "filename": "image.jpg",
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
      "ocr_details": [...],
      "tracker_id": 1,
      "class_id": 3
    }
  ],
  "total_detections": 1,
  "processing_time_ms": 1234.5,
  "model_version": "25.11.0",
  "app_version": "2.0.0",
  "timestamp": "2025-11-14T10:30:00"
}
```

**Other Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model information and detectable classes
- `POST /predict/image` - Legacy image prediction (backward compatible)
- `POST /predict/video` - Legacy video prediction (backward compatible)

### Performance Optimizations

**1. Multiprocessing for OCR:**
- Parallel processing of multiple product crops
- Uses ThreadPoolExecutor for I/O-bound operations
- Configurable via `OCRService(use_multiprocessing=True)`

**2. Lazy Model Loading:**
- OCR model loads on first use (not at startup)
- Reduces startup time
- Models are cached after first load

**3. GPU Acceleration:**
- Enable GPU for OCR: `USE_GPU=true`
- Automatic CPU fallback if GPU unavailable
- YOLO already supports GPU via PyTorch

**4. Batch Processing:**
- YOLO processes full image in single batch
- OCR processes multiple crops in parallel
- Optimized crop extraction with padding

**Performance Metrics:**
- **YOLO Detection**: ~50-80ms per frame (CPU)
- **YOLO Segmentation**: ~60-95ms per frame (CPU)
- **OCR per crop**: ~0.5-2 seconds (depends on image size)
- **SKU Matching**: <10ms per product

### Error Handling

**Detection Validation:**
- Filters detections below minimum area threshold
- Validates detected SKUs against known classes
- Removes false positives from unknown objects

**OCR Fallback:**
- Continues processing if OCR fails on individual crops
- Returns empty OCR text if extraction fails
- Logs errors without breaking pipeline

**SKU Matching Fallback:**
- Falls back to YOLO class name if no SKU match found
- Uses detection confidence if matching confidence unavailable
- Handles missing catalog gracefully

---

## Configuration

### Environment Variables

**Model Configuration:**
```bash
MODEL_PATH=models/segmentation-model.pt  # Override model path
```

**OCR Configuration:**
```bash
OCR_ENABLED=true          # Enable/disable OCR (default: true)
USE_GPU=false            # Enable GPU for OCR (default: false)
```

**Catalog Configuration:**
```bash
CATALOG_PATH=app/data/sku_master.json  # SKU master file path
USE_CATALOG_DATABASE=true              # Use DuckDB for large datasets
```

**SKU Matching:**
```bash
SKU_MATCHING_MIN_CONFIDENCE=0.65  # Minimum match confidence (0.0-1.0)
```

### Configuration File

**`app/config.py`** centralizes all configuration:
- Model paths and versioning
- Feature flags (OCR, GPU)
- Catalog paths
- Default thresholds

### Startup Configuration

**Application Initialization:**
1. Loads model from configured path
2. Initializes catalog service (JSON → DuckDB if needed)
3. Sets up detection, OCR, and SKU matching services
4. Validates required files (model, SKU master)
5. Exposes model information via `/model/info` endpoint

---

## Key Design Decisions

### 1. Model-Agnostic Architecture
- Abstraction layer allows switching model backends
- Business logic independent of model implementation
- Future-proof for better models

### 2. Modular Service Design
- Each service has single responsibility
- Services can be tested independently
- Easy to extend with new features

### 3. Scalable Catalog Management
- Automatic scaling from in-memory to database
- No manual database management required
- Efficient for both small and large datasets

### 4. Backward Compatibility
- Legacy endpoints remain functional
- Legacy field names maintained in API responses
- Gradual migration path for clients

### 5. Performance First
- Lazy loading reduces startup time
- Multiprocessing for parallel operations
- GPU acceleration where available
- Optimized batch processing

---

**Version**: 2.0.0  
**Last Updated**: 2025-11-14  
**Status**: Production Ready
