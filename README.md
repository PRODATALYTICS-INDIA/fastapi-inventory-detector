# 🚀 FastAPI Inventory Detector

REST API service for automated inventory monitoring using computer vision. This service provides endpoints for processing images and videos to detect and count items using YOLOv11 detection and segmentation models.

**Features:**
- ✅ YOLOv11 detection and segmentation models
- ✅ OCR text extraction with PaddleOCR
- ✅ SKU matching via fuzzy and keyword similarity
- ✅ Decision Engine with weighted evidence fusion
- ✅ Explainable decision reasons for auditability
- ✅ Real-time object detection and tracking with ByteTrack
- ✅ Support for both images and videos
- ✅ Pixel-accurate segmentation masks
- ✅ Base64-encoded annotated images in responses
- ✅ Configurable confidence thresholds and label modes
- ✅ Hierarchical 3-level JSON response structure

---

## 🚀 Quick Start

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd fastapi-inventory-detector

# 2. Set up environment and install dependencies
./setup_environment.sh

# 3. Start the FastAPI service
./start_fastapi.sh
```

The service will be available at **http://localhost:8000**

### Manual Start

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server using uvicorn (recommended)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python -m app.main
# Or: python app/main.py
```

**Access Points:**
- 🌐 API: http://localhost:8000
- 📖 Interactive Docs: http://localhost:8000/docs
- 📚 ReDoc: http://localhost:8000/redoc

---

## 📡 API Documentation

### FastAPI Service Endpoints (`app/main.py` - v2.0)

The service provides the following REST API endpoints:

#### 1. **GET `/`** - Root Health Check

**Description:** Simple health check to verify the API is running.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Inventory Tracking API is running",
  "status": "healthy",
  "timestamp": "2025-12-26T10:30:00.123456",
  "model_loaded": true
}
```

**Response Fields:**
- `message` (string): Status message
- `status` (string): Service status ("healthy")
- `timestamp` (string): ISO 8601 timestamp
- `model_loaded` (boolean): Whether the YOLO model is loaded

---

#### 2. **GET `/health`** - Detailed Health Check

**Description:** Detailed health check with model information.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/segmentation-model.pt",
  "timestamp": "2025-12-26T10:30:00.123456"
}
```

**Response Fields:**
- `status` (string): Service status ("healthy")
- `model_loaded` (boolean): Whether the model is loaded
- `model_path` (string): Path to the loaded model file
- `timestamp` (string): ISO 8601 timestamp

---

#### 3. **POST `/predict/image`** - Image Detection

**Description:** Process a single image and return inventory detection results with bounding boxes/masks and statistics.

**Input:**

**Request Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required, file): Image file (jpg, jpeg, png)
- `confidence_threshold` (optional, float, default: 0.5): YOLO confidence threshold (0.0-1.0)
  - Lower values detect more items but may include false positives
  - Higher values are more accurate but may miss some items
- `label_mode` (optional, string, default: "item_name"): How to aggregate detected items
  - Options: `"item_name"`, `"category"`, `"sub_category"`, `"brand"`, `"sku_code"`

**Request Example:**
```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "label_mode=item_name"
```

**Output:**

**Response Format:** `application/json`

**Success Response (200):**
```json
{
  "success": true,
  "filename": "test_image.jpg",
  "confidence_threshold": 0.5,
  "label_mode": "item_name",
  "frame_count": 1,
  "detections": {
    "boost": {
      "count": 2,
      "confidence_avg": 92.5,
      "confidence_min": 91.2,
      "confidence_max": 93.8
    },
    "colgate-paste": {
      "count": 1,
      "confidence_avg": 88.3,
      "confidence_min": 88.3,
      "confidence_max": 88.3
    }
  },
  "statistics": [
    {
      "item_name": "boost",
      "count": 2,
      "confidence(%)": "92",
      "frame_presence(%)": "100"
    },
    {
      "item_name": "colgate-paste",
      "count": 1,
      "confidence(%)": "88",
      "frame_presence(%)": "100"
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "timestamp": "2025-12-26T10:30:00.123456"
}
```

**Response Fields:**
- `success` (boolean): Whether processing was successful
- `filename` (string): Original filename
- `confidence_threshold` (float): Applied confidence threshold
- `label_mode` (string): Applied label aggregation mode
- `frame_count` (integer): Number of frames processed (always 1 for images)
- `detections` (object): Detection summary by item name
  - Each key is an item name
  - Value contains:
    - `count` (integer): Number of detections
    - `confidence_avg` (float): Average confidence percentage
    - `confidence_min` (float): Minimum confidence percentage
    - `confidence_max` (float): Maximum confidence percentage
- `statistics` (array): Detailed statistics per item
  - `item_name` (string): Item name
  - `count` (integer): Detection count
  - `confidence(%)` (string): Average confidence as percentage
  - `frame_presence(%)` (string): Frame presence as percentage
- `annotated_image` (string): Base64-encoded annotated image (data URI format)
  - For segmentation models: Includes colored masks overlaid on detected objects
  - For detection models: Includes bounding boxes around detected objects
- `timestamp` (string): ISO 8601 timestamp when processing completed

**Error Responses:**
- `400 Bad Request`: File must be an image
- `500 Internal Server Error`: Model not loaded or processing error

---

#### 4. **POST `/predict/video`** - Video Detection

**Description:** Process a video file and return aggregated inventory detection results across all frames.

**Input:**

**Request Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required, file): Video file (mp4, mov, avi, mkv)
- `confidence_threshold` (optional, float, default: 0.5): YOLO confidence threshold (0.0-1.0)
- `label_mode` (optional, string, default: "item_name"): How to aggregate detected items
  - Options: `"item_name"`, `"category"`, `"sub_category"`, `"brand"`, `"sku_code"`
- `max_frames` (optional, integer, default: 100): Maximum number of frames to process
  - The system processes every 5th frame for performance
  - Example: `max_frames=100` means ~500 actual frames will be read from the video

**Request Example:**
```bash
curl -X POST "http://localhost:8000/predict/video" \
  -F "file=@test_video.mp4" \
  -F "confidence_threshold=0.5" \
  -F "label_mode=item_name" \
  -F "max_frames=100"
```

**Output:**

**Response Format:** `application/json`

**Success Response (200):**
```json
{
  "success": true,
  "filename": "test_video.mp4",
  "confidence_threshold": 0.5,
  "label_mode": "item_name",
  "total_frames": 450,
  "processed_frames": 90,
  "detections": {
    "boost": 3,
    "colgate-paste": 2,
    "dettol-soap": 1
  },
  "statistics": [
    {
      "item_name": "boost",
      "count": 3,
      "confidence(%)": "91",
      "frame_presence(%)": "45"
    },
    {
      "item_name": "colgate-paste",
      "count": 2,
      "confidence(%)": "87",
      "frame_presence(%)": "30"
    }
  ],
  "timestamp": "2025-12-26T10:30:00.123456"
}
```

**Response Fields:**
- `success` (boolean): Whether processing was successful
- `filename` (string): Original filename
- `confidence_threshold` (float): Applied confidence threshold
- `label_mode` (string): Applied label aggregation mode
- `total_frames` (integer): Total frames in the video
- `processed_frames` (integer): Number of frames actually processed by YOLO
- `detections` (object): Dictionary of unique item IDs tracked
  - Keys are item names
  - Values are counts of unique tracked objects across all frames
- `statistics` (array): Detailed statistics per item (same format as image endpoint)
- `timestamp` (string): ISO 8601 timestamp when processing completed

**Error Responses:**
- `400 Bad Request`: File must be a video
- `500 Internal Server Error`: Model not loaded or processing error

**Note:** Objects are tracked across frames using ByteTrack algorithm. The `detections` field shows unique object counts, not per-frame counts.

---

#### 5. **GET `/model/info`** - Model Information

**Description:** Get information about the currently loaded YOLO model and its configuration.

**Request:**
```bash
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "model_path": "models/segmentation-model.pt",
  "label_mode": "item_name",
  "valid_label_modes": ["item_name", "category", "sub_category", "brand", "sku_code"],
  "confidence_threshold": 0.5,
  "model_loaded": true
}
```

**Response Fields:**
- `model_path` (string): Path to the loaded model file
- `label_mode` (string): Current label aggregation mode
- `valid_label_modes` (array): List of all available label modes
- `confidence_threshold` (float): Current confidence threshold
- `model_loaded` (boolean): Whether the model is successfully loaded

**Error Responses:**
- `500 Internal Server Error`: Model not loaded

---

#### 6. **POST `/detect`** - Unified Detection with OCR and SKU Matching (v2.0)

**Description:** Main unified endpoint for inventory detection with OCR validation and SKU matching. Returns evidence-based, frame-first response structure.

**Input:**

**Request Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required, file): Image file (jpg, jpeg, png)
- `confidence_threshold` (optional, float, default: 0.5): Detection confidence threshold (0.0-1.0)
- `enable_ocr` (optional, bool, default: true): Enable OCR text extraction
- `enable_sku_matching` (optional, bool, default: true): Enable SKU matching using OCR text
- `min_validation_confidence` (optional, float, default: 0.6): Minimum combined confidence for validation

**Request Example:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5" \
  -F "enable_ocr=true" \
  -F "enable_sku_matching=true" \
  -F "min_validation_confidence=0.6"
```

**Output:**

**Response Format:** `application/json`

**Success Response (200):**
```json
{
  "success": true,
  "status": "success",
  "timestamp": "2025-12-26T10:30:00.123456",
  "model_version": "25.11.0",
  "app_version": "2.0.0",
  "processing_time_ms": 1234.5,
  "input": {
    "type": "image",
    "filename": "test_image.jpg",
    "frame_count": 1
  },
  "summary": {
    "sku_counts": {
      "sku1_dettol_soap": {
        "sku_id": "Sku1_Dettol Soap",
        "count": 2,
        "avg_confidence": 0.85,
        "min_confidence": 0.80,
        "max_confidence": 0.90
      },
      "sku2_boost": {
        "sku_id": "Sku2_Boost",
        "count": 1,
        "avg_confidence": 0.92,
        "min_confidence": 0.92,
        "max_confidence": 0.92
      }
    },
    "total_detections": 3,
    "validated_detections": 3
  },
  "details": [
    {
      "detection_id": "f0_d0",
      "final_prediction": {
        "sku_id": "Sku1_Dettol Soap",
        "sku_name": "dettol-soap",
        "confidence": 0.89,
        "bbox": [100.0, 200.0, 300.0, 400.0],
        "validation_status": "fully_validated",
        "decision_reason": [
          "detection_ocr_agreement",
          "both_predicted_Sku1_Dettol Soap"
        ]
      },
      "detection_model": {
        "model_class": "Sku1_Dettol Soap",
        "confidence": 0.95,
        "mask": null
      },
      "ocr_model": {
        "text_raw": "dettol soap",
        "confidence_avg": 0.88,
        "sku_matching": {
          "method": "fuzzy",
          "matched_sku_id": "Sku1_Dettol Soap",
          "matched_sku_name": "dettol-soap",
          "match_confidence": 0.92
        }
      }
    }
  ]
}
```

**Response Structure (Hierarchical - 3 Levels):**

The response follows a hierarchical, scalable architecture:

**Level 1 (Top Level):**
- `success` (boolean): Whether processing was successful
- `status` (string): Processing status ("success")
- `timestamp` (string): ISO 8601 timestamp when processing completed
- `model_version` (string): Detection model version
- `app_version` (string): Application version
- `processing_time_ms` (float): Processing time in milliseconds
- `input` (object): Input information (type, filename, frame_count)
- `summary` (object): Derived summary statistics
  - `sku_counts`: SKU counts with confidence statistics (min/max/avg)
  - `total_detections`: Total number of detections
  - `validated_detections`: Number of validated detections
- `details` (array): List of detection details (Level 2)

**Level 2 (Inside `details[]` - one per detected object):**

Each detection detail contains:

- **`detection_id`** (string): Unique detection identifier (e.g., "f0_d0")
- **`final_prediction`** (object): Final output for this SKU
  - `sku_id` (string): Final SKU ID (stable identifier)
  - `sku_name` (string): Final SKU display name
  - `confidence` (float): Final aggregated confidence [0,1]
  - `bbox` (array): Bounding box [x1, y1, x2, y2] - **moved here from detection_model**
  - `validation_status` (string): Validation status enum
    - `fully_validated`: Visual detection and OCR agree on the same SKU (highest confidence)
    - `ocr_validated`: OCR matched a SKU in the catalog
    - `ocr_not_validated`: OCR found text but didn't match a SKU
    - `only_ocr_validated`: Only OCR model matched but not DETECTION MODEL
    - `detection_validated`: Visual detection matched a SKU in the catalog
    - `detection_not_validated`: Visual detection not matched any SKU in the catalog
    - `only_detection_validated`: Only detection model found something, OCR model not
    - `not_validated`: Visual detection and OCR both failed or with low confidence or providing conflicting signals
  - `decision_reason[]` (array): List of reasons for this prediction (explainable)
- **`detection_model`** (object): Detection model output
- **`ocr_model`** (object): OCR model output (combines text extraction and SKU matching)

**Level 3 (Details within `detection_model` and `ocr_model`):**

- **`detection_model`** contains:
  - `model_class` (string): Raw detection class from model
  - `confidence` (float): Detection confidence [0,1]
  - `mask` (array|null): Segmentation mask coordinates (null for detection-only models)

- **`ocr_model`** contains:
  - `text_raw` (string): Concatenated text extracted by OCR
  - `confidence_avg` (float): Average OCR confidence [0,1]
  - `sku_matching` (object): SKU matching results (Level 3)
    - `method` (string): Matching method used (e.g., "fuzzy")
    - `matched_sku_id` (string|null): Matched SKU ID (stable identifier)
    - `matched_sku_name` (string|null): Matched SKU display name
    - `match_confidence` (float|null): SKU match confidence [0,1]

**Response Fields Summary:**
- `success` (boolean): Whether processing was successful
- `status` (string): Processing status
- `timestamp` (string): Processing timestamp (ISO 8601)
- `model_version` (string): Detection model version
- `app_version` (string): Application version
- `processing_time_ms` (float): Processing time in milliseconds
- `input` (object): Input information
- `summary` (object): Summary statistics derived from base output
- `details` (array): List of detection details (Level 2)

**Error Responses:**
- `400 Bad Request`: File must be an image
- `500 Internal Server Error`: Detection model not loaded or processing error

**Design Principles:**
- **Hierarchical Structure**: 3-level hierarchy (Level 1: response, Level 2: detection details, Level 3: model outputs)
- **Final Output First**: `final_prediction` at Level 2 contains bbox, confidence, and SKU info
- **Model Outputs Separated**: `detection_model` and `ocr_model` are separate at Level 2
- **OCR Combined**: OCR text extraction and SKU matching are combined under `ocr_model` (Level 2/3)
- **Scalable**: Easy to extend with more SKUs or additional model outputs
- **Explainable**: Decision reasons tracked for auditability
- **Mask Handling**: `mask: null` indicates detection-only model (not segmentation)

---

#### 7. **POST `/model/update-label-mode`** - Update Label Mode

**Description:** Update the label mode (aggregation method) for the tracker without restarting the server.

**Input:**

**Request Method:** `POST`  
**Content-Type:** `multipart/form-data`

**Parameters:**
- `label_mode` (required, string): New label mode to apply
  - Must be one of: `"item_name"`, `"category"`, `"sub_category"`, `"brand"`, `"sku_code"`

**Request Example:**
```bash
curl -X POST "http://localhost:8000/model/update-label-mode" \
  -F "label_mode=category"
```

**Output:**

**Response Format:** `application/json`

**Success Response (200):**
```json
{
  "success": true,
  "label_mode": "category",
  "message": "Label mode updated to: category"
}
```

**Response Fields:**
- `success` (boolean): Whether the update was successful
- `label_mode` (string): The new label mode that was set
- `message` (string): Confirmation message

**Error Responses:**
- `400 Bad Request`: Invalid label_mode
- `500 Internal Server Error`: Model not loaded

---

## 🧠 Detection Logic and Decision Engine

The `/detect` endpoint uses a **Decision Engine** that implements deterministic, explainable final SKU decision logic by fusing detection and OCR signals. This section explains how the decision engine works.

### Core Principle

**Detection provides a visual prior. OCR provides confirmation. This is evidence fusion, not voting.**

### Evidence Flow

The system processes evidence in three stages:

1. **Object Detection** → Raw detection from YOLO model
2. **OCR Extraction** → Text extraction from detected object crops
3. **SKU Matching** → Fuzzy matching of OCR text to SKU catalog
4. **Decision Engine** → Fuses both signals into final prediction

### Decision Engine - Weighted Evidence Fusion

The decision engine uses **weighted evidence fusion** with dynamic weight adjustment:

#### Base Formula

```
final_confidence = W_det × detection_confidence + W_ocr × ocr_match_confidence
```

#### Default Weights

- `W_det = 0.6` (Detection weight)
- `W_ocr = 0.4` (OCR match weight)

#### Dynamic Weight Adjustment

**Case A - OCR weak or missing:**
- If `ocr_token_confidence_avg < 0.5` OR `ocr_text_raw` is empty:
  - `W_det = 0.85`
  - `W_ocr = 0.15`

**Case B - OCR strong (brand keyword hit):**
- If `ocr_match_confidence >= 0.8`:
  - `W_det = 0.5`
  - `W_ocr = 0.5`

### Decision Logic (5 Cases)

The decision engine implements deterministic cases with new validation statuses:

1. **Both weak → reject**
   - If both detection and OCR confidence below thresholds → `not_validated`

2. **Detection & OCR agree → best case**
   - If `matched_sku_id == detection_class` → `fully_validated`

3. **Detection strong, OCR missing or weak**
   - If detection strong but OCR weak/missing:
     - If detection_class is valid SKU → `only_detection_validated`
     - If detection_class is unknown/-9999 → `detection_not_validated`
   - Confidence penalty applied: `final_confidence × DETECTION_MIN`

4. **OCR strong, detection weak**
   - If OCR strong but detection weak:
     - If detection_class is valid SKU → `ocr_validated`
     - If detection_class is unknown/-9999 → `only_ocr_validated`
   - Confidence penalty applied: `final_confidence × OCR_MATCH_MIN`

5. **Detection & OCR conflict**
   - If `matched_sku_id != detection_class`:
     - If OCR significantly stronger (delta > 0.2):
       - If detection_class is valid SKU → `ocr_validated`
       - If detection_class is unknown/-9999 → `only_ocr_validated`
     - Otherwise (detection wins):
       - If detection_class is valid SKU → `detection_validated`
       - If detection_class is unknown/-9999 → `detection_not_validated`
   - Conflict penalty applied: `final_confidence - 0.15`

6. **Detection matched SKU, but OCR found text that didn't match**
   - If detection_class is valid SKU and OCR found text but no match → `ocr_not_validated`

### Configurable Thresholds

All thresholds are configurable via environment variables:

- `DETECTION_MIN = 0.6` (default)
- `OCR_MATCH_MIN = 0.7` (default)
- `OCR_TOKEN_MIN = 0.5` (default)
- `CONFLICT_DELTA = 0.2` (default)

#### Examples

**Example 1: Detection Only (No OCR)**
```
detection_confidence = 0.95
ocr_confidence_avg = 0.0
match_confidence = None

final_confidence = 0.95 (100% from detection)
```

**Example 2: Detection + OCR (No SKU Match)**
```
detection_confidence = 0.90
ocr_confidence_avg = 0.85
match_confidence = None (no SKU match)

Weights: W_det = 0.6, W_ocr = 0.4 (default)
But since match_confidence is None, OCR weight is effectively 0
final_confidence = (0.90 × 0.6) + (0.0 × 0.4)
                 = 0.54

Status: only_detection_validated (OCR text extracted but no match)
Penalty: final_confidence × DETECTION_MIN = 0.54 × 0.6 = 0.324
```

**Example 3: Full Pipeline (Detection + OCR + SKU Match)**
```
detection_confidence = 0.92
ocr_confidence_avg = 0.88
match_confidence = 0.90 (SKU matched)

Weights: W_det = 0.5, W_ocr = 0.5 (OCR strong, match_confidence >= 0.8)
final_confidence = (0.92 × 0.5) + (0.90 × 0.5)
                 = 0.46 + 0.45
                 = 0.91

Status: fully_validated (detection and OCR agree on same SKU)
No penalty applied
```

### Validation Status - Detailed Explanation

The system assigns one of eight validation statuses based on available evidence and confidence thresholds. Each status indicates the quality and reliability of the detection:

#### 1. **`fully_validated`** - Best Case: Both Models Agree

**When it occurs:**
- Detection model predicts a SKU
- OCR matches the same SKU (`matched_sku_id == detection_class`)
- Both signals agree on the prediction

**What it means:**
- Highest confidence scenario - both visual and text evidence agree
- No confidence penalties applied
- Most reliable detection for production use

**Use cases:**
- Production-ready validated detections
- High-confidence inventory counting
- Automated systems requiring reliable SKU identification

**Example scenario:**
```
Detection: "Sku1_Dettol Soap" (confidence: 0.92)
OCR: Matched "Sku1_Dettol Soap" (match_confidence: 0.88)
Result: fully_validated, final_confidence: weighted_fusion (no penalty)
```

---

#### 2. **`ocr_validated`** - OCR Matched a SKU

**When it occurs:**
- OCR match confidence ≥ `OCR_MATCH_MIN` (default: 0.7)
- Detection also matched a SKU (but different or weaker)
- Both models detected SKUs, OCR is stronger or wins in conflict

**What it means:**
- OCR successfully matched a SKU in the catalog
- Detection also matched a SKU (may be different)
- OCR evidence is stronger or wins in conflict resolution

**Use cases:**
- Text-based identification when visual features are ambiguous
- Products with clear text labels but similar visual appearance
- OCR stronger in conflict scenarios

**Example scenario:**
```
Detection: "Sku1_Dettol Soap" (confidence: 0.65)
OCR: Matched "Sku3_Dettol Handwash" (match_confidence: 0.90)
Result: ocr_validated, final_confidence: (weighted_fusion) - 0.15 (conflict penalty)
```

---

#### 3. **`only_ocr_validated`** - Only OCR Model Matched

**When it occurs:**
- OCR match confidence ≥ `OCR_MATCH_MIN` (default: 0.7)
- Detection confidence < `DETECTION_MIN` OR detection_class is unknown/-9999
- OCR successfully matched a SKU, but detection didn't match any SKU

**What it means:**
- Text was successfully extracted and matched to a SKU
- Visual detection was below threshold or didn't match any SKU
- **Confidence penalty applied:** `final_confidence × OCR_MATCH_MIN`

**Use cases:**
- Text-based identification when visual detection failed
- Products with clear text labels but unrecognized visual appearance
- Low-quality images where text is clearer than visual features

**Example scenario:**
```
Detection: "unknown" or confidence: 0.55 (below threshold)
OCR: Matched "Sku2_Boost" (match_confidence: 0.85)
Result: only_ocr_validated, final_confidence: (weighted_fusion) × 0.7
```

---

#### 4. **`ocr_not_validated`** - OCR Found Text But No Match

**When it occurs:**
- Detection model matched a valid SKU
- OCR found text but didn't match any SKU in catalog
- `ocr_text_raw` is not empty but `matched_sku_id` is None

**What it means:**
- Visual detection matched a SKU
- OCR extracted text but couldn't match it to any SKU
- Text may be unclear, partial, or not in catalog

**Use cases:**
- Products with text that doesn't match catalog entries
- Partial text extraction
- Unclear or damaged text labels

**Example scenario:**
```
Detection: "Sku2_Boost" (confidence: 0.85)
OCR: Found text "boos" but no SKU match
Result: ocr_not_validated
```

---

#### 5. **`detection_validated`** - Detection Matched a SKU

**When it occurs:**
- Detection model matched a valid SKU
- OCR also found text/matched (but different or weaker)
- Detection wins in conflict or both matched different SKUs

**What it means:**
- Visual detection matched a SKU in the catalog
- OCR also provided evidence (may be different SKU)
- Detection evidence is stronger or wins in conflict

**Use cases:**
- Visual features are more reliable than text
- Detection model correctly identifies visually distinct items
- OCR matched different SKU but detection is more confident

**Example scenario:**
```
Detection: "Sku1_Dettol Soap" (confidence: 0.85)
OCR: Matched "Sku3_Dettol Handwash" (match_confidence: 0.70)
Result: detection_validated, final_confidence: (weighted_fusion) - 0.15 (conflict penalty)
```

---

#### 6. **`only_detection_validated`** - Only Detection Model Matched

**When it occurs:**
- Detection model confidence ≥ `DETECTION_MIN` (default: 0.6)
- Detection matched a valid SKU
- OCR text is empty OR OCR match confidence < `OCR_MATCH_MIN` (default: 0.7)

**What it means:**
- Object was visually detected with sufficient confidence
- OCR either failed, was disabled, or didn't match any SKU in catalog
- **Confidence penalty applied:** `final_confidence × DETECTION_MIN`

**Use cases:**
- Fast detection when OCR is disabled (`enable_ocr=false`)
- Objects with no visible text (e.g., generic packaging)
- OCR failure due to poor image quality
- Text present but doesn't match any SKU in catalog

**Example scenario:**
```
Detection: "Sku2_Boost" (confidence: 0.85)
OCR: Empty text or match confidence: 0.5
Result: only_detection_validated, final_confidence: 0.85 × 0.6 = 0.51
```

---

#### 7. **`detection_not_validated`** - Detection Didn't Match Any SKU

**When it occurs:**
- Detection model didn't match any SKU (detection_class is unknown/-9999)
- Detection confidence may be above threshold but SKU is unknown

**What it means:**
- Visual detection found an object but couldn't identify it as a known SKU
- Object may not be in the catalog or model's training data
- Detection may be a false positive or unrecognized item

**Use cases:**
- Unrecognized items not in catalog
- Objects outside model's training scope
- New products not yet added to catalog

**Example scenario:**
```
Detection: "unknown" or "-9999" (confidence: 0.75)
OCR: No text or no match
Result: detection_not_validated
```

---

#### 8. **`not_validated`** - Both Failed or Low Confidence

**When it occurs:**
- Detection confidence < `DETECTION_MIN` (default: 0.6)
- OCR match confidence < `OCR_MATCH_MIN` (default: 0.7)
- Both signals are below their respective thresholds
- OR both failed to provide reliable evidence

**What it means:**
- Neither detection nor OCR provided sufficient evidence
- Detection may be a false positive or low-quality
- **No SKU assigned:** `sku_id` will be `null` or "-9999"

**Use cases:**
- Poor image quality affecting both models
- Unrecognized items not in catalog
- Edge cases requiring manual review

**Example scenario:**
```
Detection: "Sku2_Boost" (confidence: 0.55) - below threshold
OCR: Matched "Sku2_Boost" (match_confidence: 0.60) - below threshold
Result: not_validated, sku_id: null, final_confidence: low
```

---

### Decision Reasons - How to Interpret

Each detection includes a `decision_reason` array that provides an **explainable audit trail** of why the prediction was made. This is crucial for debugging, auditing, and understanding model behavior.

#### Decision Reason Format

Decision reasons are strings that follow a pattern: `"reason_type"` or `"reason_type_value_comparison"`

**Common Decision Reason Patterns:**

1. **Agreement Reasons:**
   - `"detection_ocr_agreement"` - Both models predicted the same SKU
   - `"both_predicted_{sku_id}"` - Specific SKU both models agreed on

2. **Signal Strength Reasons:**
   - `"detection_strong_ocr_weak"` - Detection above threshold, OCR below
   - `"ocr_strong_detection_weak"` - OCR above threshold, detection below
   - `"detection_confidence_{value}_above_{threshold}"` - Detection confidence details
   - `"ocr_match_confidence_{value}_above_{threshold}"` - OCR match confidence details
   - `"detection_confidence_{value}_below_{threshold}"` - Detection below threshold
   - `"ocr_match_confidence_{value}_below_{threshold}"` - OCR below threshold

3. **Conflict Reasons:**
   - `"detection_ocr_conflict"` - Models disagree on SKU
   - `"detection_class_{class}_vs_ocr_sku_{sku_id}"` - Specific conflict details
   - `"ocr_stronger_in_conflict"` - OCR won the conflict
   - `"detection_wins_conflict"` - Detection won the conflict
   - `"ocr_delta_{delta}_above_{threshold}"` - OCR significantly stronger
   - `"ocr_delta_{delta}_below_{threshold}"` - OCR not strong enough to override

4. **Penalty Reasons:**
   - `"confidence_penalty_applied_{value}"` - Penalty applied (e.g., `0.6` for detection_only, `0.7` for ocr_detected)
   - `"conflict_penalty_applied_0.15"` - Conflict penalty applied

5. **Data Quality Reasons:**
   - `"ocr_text_empty"` - No OCR text extracted
   - `"both_signals_weak"` - Both below thresholds
   - `"no_ocr_match_available"` - OCR didn't match any SKU

6. **Fallback Reasons:**
   - `"detection_fallback"` - Used detection as fallback
   - `"legacy_fallback"` - Legacy code path (should not occur in normal operation)

#### Example Decision Reason Arrays

**Example 1: Fully Validated**
```json
"decision_reason": [
  "detection_ocr_agreement",
  "both_predicted_Sku1_Dettol Soap"
]
```
**Interpretation:** Both models agreed on the same SKU - highest confidence scenario.

**Example 2: Detection Only (OCR Failed)**
```json
"decision_reason": [
  "detection_strong_ocr_weak",
  "detection_confidence_0.85_above_0.6",
  "ocr_text_empty",
  "confidence_penalty_applied_0.6"
]
```
**Interpretation:** Detection was strong (0.85), but OCR found no text. Penalty applied (×0.6).

**Example 3: OCR Stronger in Conflict**
```json
"decision_reason": [
  "ocr_stronger_in_conflict",
  "ocr_delta_0.25_above_0.2",
  "detection_ocr_conflict",
  "detection_class_Sku1_Dettol Soap_vs_ocr_sku_Sku3_Dettol Handwash",
  "conflict_penalty_applied_0.15"
]
```
**Interpretation:** Models disagreed. OCR was 0.25 points stronger, so OCR prediction chosen. Conflict penalty applied (-0.15).

**Example 4: Low Confidence**
```json
"decision_reason": [
  "both_signals_weak",
  "detection_confidence_0.55_below_0.6",
  "ocr_match_confidence_0.60_below_0.7"
]
```
**Interpretation:** Both signals below thresholds. Detection too weak (0.55 < 0.6), OCR match too weak (0.60 < 0.7).

#### How to Use Decision Reasons

1. **Debugging:** Check reasons to understand why a detection got a specific status
2. **Auditing:** Track which detections had conflicts or penalties
3. **Tuning:** Identify patterns (e.g., frequent OCR failures) to improve system
4. **Quality Control:** Filter detections by validation status (e.g., only accept `fully_validated`, `ocr_validated`, `detection_validated`)
5. **Monitoring:** Alert on high conflict rates or frequent low confidence detections

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.5 | Minimum detection confidence to process |
| `min_validation_confidence` | 0.6 | Minimum final confidence for validation |
| `enable_ocr` | true | Enable OCR text extraction |
| `enable_sku_matching` | true | Enable SKU matching via OCR |

### Best Practices

1. **For High Accuracy:** Use `min_validation_confidence=0.7` to require stronger evidence
2. **For Speed:** Disable OCR (`enable_ocr=false`) for detection-only mode
3. **For Catalog Matching:** Ensure SKU catalog is up-to-date for best matching results
4. **For Debugging:** Check `decision_reason` array to understand prediction logic

---

## 🏗️ Project Structure

```
fastapi-inventory-detector/
├── app/                         # ✅ Application package (v2.0 with OCR and Decision Engine)
│   ├── main.py                 # FastAPI application (v2.0) - PRIMARY ENTRY POINT
│   │                           #   - Defines all API endpoints (/detect, /health, etc.)
│   │                           #   - Orchestrates detection, OCR, and decision engine
│   │                           #   - Builds hierarchical JSON responses
│   │
│   ├── config.py               # Configuration management
│   │                           #   - Environment variable loading
│   │                           #   - Model paths, thresholds, weights
│   │                           #   - Decision engine parameters
│   │
│   ├── models/                 # Model wrappers (YOLO and OCR)
│   │   ├── detection_model.py  # YOLO detection/segmentation model wrapper
│   │   │                       #   - Loads YOLOv11 model
│   │   │                       #   - Handles detection and segmentation
│   │   │                       #   - Returns bounding boxes and masks
│   │   └── ocr_model.py        # OCR model wrapper (PaddleOCR)
│   │                           #   - Text extraction from image crops
│   │                           #   - Returns text and confidence scores
│   │
│   ├── services/               # Business logic services
│   │   ├── detection_service.py # Detection orchestration
│   │   │                       #   - Processes images through YOLO
│   │   │                       #   - Extracts bounding boxes and crops
│   │   │                       #   - Handles model inference
│   │   │
│   │   ├── ocr_service.py      # OCR orchestration
│   │   │                       #   - Processes image crops through OCR
│   │   │                       #   - Extracts text and token confidences
│   │   │                       #   - Returns structured OCR results
│   │   │
│   │   ├── sku_matching_service.py # SKU matching logic
│   │   │                           #   - Fuzzy matching of OCR text to catalog
│   │   │                           #   - Keyword similarity matching
│   │   │                           #   - Returns matched SKU with confidence
│   │   │
│   │   ├── catalog_service.py  # Catalog management
│   │   │                       #   - Loads SKU master data (JSON/DuckDB)
│   │   │                       #   - Provides SKU lookup by ID
│   │   │                       #   - Handles large catalogs efficiently
│   │   │
│   │   └── decision_engine.py  # ⭐ Decision Engine (Core Fusion Logic)
│   │                           #   - Fuses detection + OCR signals
│   │                           #   - Implements 5-case decision logic
│   │                           #   - Dynamic weight adjustment
│   │                           #   - Generates explainable decision reasons
│   │                           #   - Pure function (no model calls)
│   │
│   ├── schemas/                # Pydantic models (API contracts)
│   │   ├── request.py          # Request schemas
│   │   │                       #   - API input validation
│   │   │                       #   - Form parameters
│   │   │
│   │   └── response.py         # Response schemas
│   │                           #   - Hierarchical 3-level structure
│   │                           #   - ValidationStatus enum
│   │                           #   - DetectionDetail, FinalPrediction, etc.
│   │                           #   - JSON schema examples
│   │
│   ├── utils/                  # Utility functions
│   │   ├── image_utils.py      # Image processing utilities
│   │   │                       #   - Crop extraction from bboxes
│   │   │                       #   - Image format conversion
│   │   │                       #   - Annotation drawing
│   │   │
│   │   ├── inventory_summary.py # Summary generation
│   │   │                       #   - Aggregates detections by SKU
│   │   │                       #   - Calculates min/max/avg confidence
│   │   │                       #   - Counts validated detections
│   │   │
│   │   └── inventory_tracker.py # Legacy tracker (ByteTrack)
│   │                           #   - Object tracking across video frames
│   │                           #   - Used by /predict/video endpoint
│   │
│   └── data/                   # Data files
│       └── sku_master.json     # SKU catalog (JSON format)
│                               #   - Contains SKU IDs, names, metadata
│                               #   - Can be converted to DuckDB for large datasets
│
├── models/                     # Model weights directory
│   └── segmentation-model.pt   # YOLO model file (detection/segmentation)
│                               #   - Place your trained YOLOv11 model here
│                               #   - Supports both detection and segmentation
│
├── test/                       # Test suite
│   ├── test_local.py           # Local testing (without API server)
│   │                           #   - Direct function calls
│   │                           #   - Generates api_response.json files
│   │                           #   - Processes test_images/ directory
│   │
│   ├── test_api.py             # API testing (with running server)
│   │                           #   - Tests FastAPI endpoints
│   │                           #   - Validates response structure
│   │                           #   - Generates test output files
│   │
│   ├── test_new_structure.py   # Structure validation tests
│   │
│   ├── test_images/            # Test image files
│   │                           #   - Sample images for testing
│   │
│   └── test_output/            # Test output directory
│       └── local_call/         # Output from test_local.py
│                               #   - api_response.json (API response structure)
│                               #   - detection_response.jpg (annotated image)
│                               #   - ocr_response.txt (OCR text output)
│
├── documents/                  # Documentation
│   ├── IMPLEMENTATION_GUIDE.md # Detailed implementation guide
│   ├── TEST_DOCUMENTATION.md   # Test documentation
│   └── README.md               # Deployment/remote connection docs
│
├── deployment/                 # Remote deployment scripts
│   ├── connect.sh              # SSH connection helper
│   ├── upload.sh               # File upload script
│   ├── view_server.sh          # Server status viewer
│   └── cleanup.sh              # Cleanup script
│
├── requirements.txt            # Python dependencies
│                               #   - FastAPI, Uvicorn
│                               #   - Ultralytics (YOLO)
│                               #   - PaddleOCR
│                               #   - Pydantic, etc.
│
├── deployment/
│   └── Dockerfile             # Docker configuration
│                               #   - Container image definition
│                               #   - Production deployment
│
├── setup_environment.sh        # Environment setup script
│                               #   - Creates virtual environment
│                               #   - Installs dependencies
│
└── start_fastapi.sh            # FastAPI startup script
                                #   - Activates venv
                                #   - Starts uvicorn server
                                #   - Uses app/main.py as entry point
```

### Key Components Overview

**Entry Point:** `app/main.py`
- FastAPI application with all endpoints
- Orchestrates the entire detection pipeline
- Handles request/response serialization

**Core Logic Flow:**
1. **Detection Service** → YOLO model detects objects
2. **OCR Service** → Extracts text from detected object crops
3. **SKU Matching Service** → Matches OCR text to catalog
4. **Decision Engine** → Fuses signals into final prediction
5. **Response Builder** → Constructs hierarchical JSON response

**Decision Engine (`app/services/decision_engine.py`):**
- Pure function (no model calls, deterministic)
- Implements weighted evidence fusion
- 5-case decision logic with explainable reasons
- Configurable thresholds and weights

**Schemas (`app/schemas/response.py`):**
- Defines API contract (request/response structure)
- 3-level hierarchical JSON structure
- ValidationStatus enum (5 statuses)
- Pydantic models for type safety

**Configuration (`app/config.py`):**
- Centralized configuration management
- Environment variable support
- Default thresholds and weights
- Model metadata

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -f deployment/Dockerfile -t fastapi-inventory-detector:latest .

# Run the container
docker run -d -p 8000:8000 fastapi-inventory-detector:latest

# Access API at http://localhost:8000
```

---

## ⚡ Performance

### Inference Speed (CPU)
- **YOLO Detection**: ~50-80ms per frame
- **YOLO Segmentation**: ~60-95ms per frame (+10-15% slower)

### Memory Usage
- **YOLO Detection Model**: ~500MB GPU VRAM
- **YOLO Segmentation Model**: ~600MB GPU VRAM (+20% higher)

### Video Processing
- Frame skip: Every 5th frame (configurable)
- Max frames: 100 processed frames (configurable via API)
- Example: 30-second video @ 30fps = 900 frames → 180 processed frames

---

## 🔧 Configuration

### Model Selection

**Model File:**
- Place your YOLO model file at: `models/segmentation-model.pt`
- Supports both detection and segmentation models
- The system automatically detects model type

**Using a Different Model:**
```bash
# Replace the model file
mv your-model.pt models/segmentation-model.pt
```

### Label Modes

Available modes: `item_name`, `category`, `sub_category`, `brand`, `sku_code`

Change via API endpoint `/model/update-label-mode` or update in code.

### Confidence Thresholds

**YOLO Confidence:**
- Lower threshold (0.3) = more detections (more false positives)
- Higher threshold (0.7) = fewer detections (more accurate)
- Default: 0.5 (50% confidence)

---

## 🧪 Testing

### Quick Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Image prediction
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5"

# Video prediction
curl -X POST "http://localhost:8000/predict/video" \
  -F "file=@test_video.mp4" \
  -F "confidence_threshold=0.5" \
  -F "max_frames=100"

# Model info
curl http://localhost:8000/model/info
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Image prediction
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    data = {"confidence_threshold": 0.5, "label_mode": "item_name"}
    response = requests.post(
        "http://localhost:8000/predict/image",
        files=files,
        data=data
    )
    result = response.json()
    print(f"Detected {len(result['detections'])} item types")
    for item, info in result['detections'].items():
        print(f"  - {item}: {info['count']} items (avg confidence: {info['confidence_avg']}%)")
```

---

## 🔧 Troubleshooting

### Issue: Model Not Loading

**Solution:** Ensure model file exists in `models/` directory:
```bash
ls models/segmentation-model.pt
```

### Issue: Import Errors

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Slow Performance

**Solutions:**
- Use GPU if available
- Reduce image resolution before processing
- Increase frame skip for videos (currently every 5th frame)
- Increase confidence threshold to reduce detections

### Issue: API Timeout

**Solutions:**
- Increase timeout in client requests
- Reduce `max_frames` for video processing
- Check server logs for errors

---

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Supervision Library](https://roboflow.github.io/supervision/)

### Project Documentation
All documentation files are located in the `documents/` directory:
- `documents/IMPLEMENTATION_GUIDE.md` - Complete implementation guide
- `documents/TEST_DOCUMENTATION.md` - Comprehensive test documentation

---

## 📄 License

This project is licensed under the MIT License.

---

**Ready to deploy your inventory detection API! 🚀**
