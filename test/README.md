# Test Suite Documentation

## Overview

The test suite is organized into two main test types:

1. **Local Tests** (`test_local.py`) - Run detection, OCR, and visualization locally using project components
2. **API Tests** (`test_api.py`) - Test via FastAPI endpoint (automatically starts server if needed)

Both tests perform the same operations:
- Process each image from `test_images/` directory
- Detect all objects using the segmentation model
- Create visualization with bounding boxes and masks
- Crop detected objects
- Run OCR on each crop
- Match OCR text to SKU catalog
- Save all results to `test_output/<image_name>/`

## Test Files

### `test_local.py`

Runs the complete pipeline locally without requiring the API server.

**Features:**
- Uses project components directly (DetectionModel, OCRModel, etc.)
- No network required
- Faster execution (no HTTP overhead)
- Better for debugging

**Usage:**
```bash
# Process all images
python test/test_local.py

# Process single image
python test/test_local.py --image 20251010-1.jpeg

# Custom confidence threshold
python test/test_local.py --confidence 0.6
```

### `test_api.py`

Tests the complete pipeline via FastAPI endpoint.

**Features:**
- Automatically starts server if not running
- Tests the actual API endpoints
- Validates API response format
- Better for integration testing

**Usage:**
```bash
# Process all images (auto-starts server)
python test/test_api.py

# Use custom API URL
python test/test_api.py --api-url http://localhost:8000

# Process single image
python test/test_api.py --image 20251010-1.jpeg

# Don't auto-start server
python test/test_api.py --no-auto-start
```

## Output Structure

Both tests create the same output structure:

```
test_output/
└── <image_name>/
    ├── detection_response.jpg  # Visualization with bounding boxes and masks
    ├── ocr_response.txt        # Detailed results with OCR and SKU matching
    └── api_response.json       # API response in JSON format
```

**Note:** All detections, OCR results, and API responses are linked by consistent Detection IDs (tracker_id) across all three files for performance tracking.

### `ocr_response.txt` Contents

The results file contains:
- Image metadata (filename, processing time, confidence threshold)
- Detection summary (counts by item type)
- Detailed results for each object:
  - Detection information (SKU, confidence, bounding box)
  - OCR extracted text and confidence scores
  - SKU matching results and confidence
  - Validation status

## Shell Scripts

### `run_test_local.sh` (to be created)
Wrapper script for local tests.

### `run_test_api.sh` (to be created)
Wrapper script for API tests.

## Test Files Overview

### Main Test Suites (Primary)
- **test_local.py** - Complete local test suite (use this for local testing)
- **test_api.py** - Complete API test suite (use this for API testing)

### Specialized Tests (Secondary)
- **test_ocr_local.py** - OCR-only testing (useful for debugging OCR without full pipeline)
- **test_new_structure.py** - Structure verification (checks imports, SKU master, model paths)

### Shell Scripts
- **run_test_ocr_batch_local.sh** - Batch local testing wrapper (calls `test_local.py`)
- **run_test_ocr_batch.sh** - Batch API testing via curl (alternative to Python test)
- **run_test.sh** - API testing with markdown report output (curl-based)
- **run_test_api_curl.sh** - Remote API testing (tests production API at render.com)

## Requirements

Both tests require:
- Python 3.8+
- All project dependencies installed
- Model file at `models/segmentation-model.pt`
- SKU master catalog at `app/data/sku_master.json`
- Test images in `test/test_images/` directory

For API tests:
- FastAPI server accessible (auto-started if not running)
- Port 8000 available (or specify custom URL)

## Examples

### Run Local Test on All Images
```bash
cd /path/to/project
python test/test_local.py
```

### Run API Test on Single Image
```bash
cd /path/to/project
python test/test_api.py --image 20251010-1.jpeg
```

### Run with Custom Confidence
```bash
python test/test_local.py --confidence 0.7
python test/test_api.py --confidence 0.7
```

## Troubleshooting

### Local Test Issues

**Error: Model not found**
- Ensure `models/segmentation-model.pt` exists
- Check `app/config.py` for correct model path

**Error: OCR model initialization failed**
- Install PaddleOCR: `pip install paddlepaddle paddleocr`
- Check GPU availability if using GPU

### API Test Issues

**Error: Server not starting**
- Check if port 8000 is already in use
- Manually start server: `uvicorn app.main:app --reload`
- Use `--no-auto-start` flag and start server manually

**Error: Connection refused**
- Ensure server is running
- Check API URL is correct
- Verify firewall settings

## Notes

- Both tests create output in the same `test_output/` directory
- Results from different test runs will overwrite previous results for the same image
- The API test automatically stops the server if it started it
- Use `--no-auto-start` if you want to manage the server manually

