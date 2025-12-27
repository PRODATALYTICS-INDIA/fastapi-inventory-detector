"""
FastAPI application with integrated detection and OCR-based SKU identification.

This service provides:
- Detection/segmentation model for product localization
- PaddleOCR for text extraction from product crops
- Fuzzy matching for SKU identification
- Unified API endpoint for detection and identification with validation
"""
import os
import io
import time
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler

import uvicorn
import numpy as np
# Lazy import cv2 to prevent OOM during module load
# cv2 will be imported only when needed during request handling
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Response
from fastapi.middleware.cors import CORSMiddleware

# Import configuration
from app.config import (
    get_model_path,
    get_model_metadata,
    MODEL_VERSION,
    MODEL_BASE_NAME,
    APP_VERSION,
    CATALOG_PATH,
    SKU_MASTER_PATH,
    USE_GPU,
    OCR_ENABLED,
    USE_CATALOG_DATABASE,
    SKU_MATCHING_MIN_CONFIDENCE,
    LOG_DIR,
    LOG_FILE,
    LOG_LEVEL,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    ENABLE_FILE_LOGGING
)

# Import new modular components
from app.models.detection_model import DetectionModel
from app.models.ocr_model import OCRModel
from app.services.catalog_service import CatalogService
from app.services.detection_service import DetectionService
from app.services.ocr_service import OCRService
from app.services.sku_matching_service import SKUMatchingService
from app.services.decision_engine import DecisionEngine
from app.schemas.response import (
    DetectionResponse,
    ProductDetection,
    HealthResponse,
    ModelInfoResponse,
    DetectionDetail,
    DetectionModelOutput,
    OCRModelOutput,
    FinalPrediction,
    ValidationStatus,
    InputInfo
)
from app.utils.inventory_summary import generate_summary

# Import legacy tracker for backward compatibility
from app.utils.inventory_tracker import InventoryTracker

# Configure logging with file handler
def setup_logging():
    """
    Set up logging configuration with both console and file handlers.
    File logging can be enabled/disabled via ENABLE_FILE_LOGGING environment variable.
    """
    # Get log level
    log_level = getattr(logging, LOG_LEVEL, logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if ENABLE_FILE_LOGGING:
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        
        log_file_path = os.path.join(LOG_DIR, LOG_FILE)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        logger = logging.getLogger(__name__)
        logger.info(f"📝 File logging enabled: {log_file_path}")
        logger.info(f"   Max file size: {LOG_MAX_BYTES / 1024 / 1024:.1f}MB")
        logger.info(f"   Backup count: {LOG_BACKUP_COUNT}")
    else:
        logger = logging.getLogger(__name__)
        logger.info("📝 File logging disabled (ENABLE_FILE_LOGGING=false)")
    
    return logger

# Initialize logging
logger = setup_logging()

# Global model instances
detection_model: Optional[DetectionModel] = None
ocr_model: Optional[OCRModel] = None
catalog_service: Optional[CatalogService] = None
detection_service: Optional[DetectionService] = None
ocr_service: Optional[OCRService] = None
sku_matching_service: Optional[SKUMatchingService] = None
decision_engine: Optional[DecisionEngine] = None

# Legacy tracker for backward compatibility
legacy_tracker: Optional[InventoryTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global detection_model, ocr_model, catalog_service, detection_service, ocr_service, sku_matching_service, decision_engine, legacy_tracker
    
    try:
        # Initialize catalog service first (used by detection and SKU matching)
        if os.path.exists(CATALOG_PATH):
            logger.info(f"Loading catalog from: {CATALOG_PATH}")
            catalog_service = CatalogService(
                catalog_path=CATALOG_PATH,
                use_database=USE_CATALOG_DATABASE
            )
            catalog_stats = catalog_service.get_catalog_stats()
            logger.info(f"✅ Catalog service loaded: {catalog_stats['total_skus']} SKUs")
            if catalog_stats['using_database']:
                logger.info(f"   Using DuckDB database: {catalog_stats['database_path']}")
            else:
                logger.info("   Using in-memory lookup")
        else:
            logger.warning(f"⚠️ Catalog file not found: {CATALOG_PATH}")
            logger.warning("Catalog service will not be available")
            catalog_service = None
        
        # Initialize detection model
        model_path = get_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_metadata = get_model_metadata()
        logger.info(f"Loading detection model from: {model_path}")
        logger.info(f"Model name: {model_metadata['name']}")
        logger.info(f"Model version: {model_metadata['version']}")
        detection_model = DetectionModel(model_path)
        detection_service = DetectionService(detection_model, catalog_service=catalog_service)
        logger.info("✅ Detection model loaded successfully")
        
        # Initialize OCR model (lazy loading, but prepare instance)
        if OCR_ENABLED:
            try:
                logger.info("Initializing OCR model...")
                ocr_model = OCRModel(use_gpu=USE_GPU)
                # Use shorter timeout to prevent hanging (3 seconds per crop)
                ocr_service = OCRService(ocr_model, use_multiprocessing=True, timeout_per_crop=3.0)
                logger.info("✅ OCR model initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ OCR model initialization failed: {str(e)}")
                logger.warning("OCR features will be disabled")
                ocr_model = None
                ocr_service = None
        else:
            logger.info("OCR disabled via configuration")
            ocr_model = None
            ocr_service = None
        
        # Initialize SKU matching service (uses catalog service)
        if catalog_service:
            logger.info("Initializing SKU matching service with catalog...")
            sku_matching_service = SKUMatchingService(
                catalog_service=catalog_service,
                min_confidence=SKU_MATCHING_MIN_CONFIDENCE
            )
            logger.info("✅ SKU matching service loaded successfully")
        elif os.path.exists(SKU_MASTER_PATH):
            # Fallback to JSON file if catalog service not available
            logger.warning(f"Using legacy JSON SKU master: {SKU_MASTER_PATH}")
            sku_matching_service = SKUMatchingService(
                catalog_path=None,  # Will use JSON fallback
                min_confidence=SKU_MATCHING_MIN_CONFIDENCE
            )
            logger.info("✅ SKU matching service loaded (legacy mode)")
        else:
            logger.warning(f"⚠️ No catalog or SKU master file found")
            logger.warning("SKU matching will be disabled")
            sku_matching_service = None
        
        # Initialize Decision Engine (required for final prediction fusion)
        logger.info("Initializing Decision Engine...")
        decision_engine = DecisionEngine(catalog_service=catalog_service)
        logger.info("✅ Decision Engine initialized successfully")
        
        # Initialize legacy tracker for backward compatibility
        try:
            legacy_tracker = InventoryTracker(model_path=model_path, label_mode="item_name")
            logger.info("✅ Legacy tracker initialized for backward compatibility")
        except Exception as e:
            logger.warning(f"⚠️ Legacy tracker initialization failed: {str(e)}")
        
        logger.info(f"Application version: {APP_VERSION}")
        logger.info("✅ Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Log full error details before exiting
        logger.error("=" * 80)
        logger.error("CRITICAL: Application startup failed")
        logger.error("Container will exit. Check the error above for details.")
        logger.error("=" * 80)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if catalog_service:
        catalog_service.close()


# Initialize FastAPI app with lifespan
# Hide Schemas section in Swagger UI for cleaner interface
app = FastAPI(
    title="Inventory Detection API",
    description="REST API for automated inventory detection with validation from detection and OCR services",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}  # Hide Schemas section
)


# Customize OpenAPI schema to fix 422 validation error examples and ensure correct field order
def custom_openapi():
    """Custom OpenAPI schema with proper 422 error examples and correct field order."""
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Ensure servers are set correctly for Swagger UI
    # Use empty string for relative URLs (works for both localhost and Docker)
    # This tells Swagger UI to use the same origin as the page, avoiding CORS issues
    openapi_schema["servers"] = [{"url": "", "description": "Current server"}]
    
    # Update 422 error examples and ensure correct field order for 200 responses
    for path_key, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if isinstance(operation, dict) and "responses" in operation:
                # Fix 422 error examples
                if "422" in operation["responses"]:
                    # Set proper 422 validation error example based on endpoint
                    if "detect-and-count" in path_key:
                        # For file upload endpoint, show missing file error
                        example = {
                            "detail": [
                                {
                                    "loc": ["body", "file"],
                                    "msg": "field required",
                                    "type": "value_error.missing"
                                }
                            ]
                        }
                    else:
                        # Generic validation error example
                        example = {
                            "detail": [
                                {
                                    "loc": ["body", "field_name"],
                                    "msg": "field required",
                                    "type": "value_error.missing"
                                }
                            ]
                        }
                    
                    operation["responses"]["422"]["content"] = {
                        "application/json": {
                            "example": example
                        }
                    }
                
                # Ensure 200 response example has correct field order
                if "200" in operation["responses"]:
                    response_200 = operation["responses"]["200"]
                    if "content" in response_200 and "application/json" in response_200["content"]:
                        # Get the example from the schema if it exists
                        json_content = response_200["content"]["application/json"]
                        if "example" in json_content:
                            example = json_content["example"]
                            # Reorder the example to match the desired order
                            if isinstance(example, dict):
                                ordered_example = {}
                                # Level 1 parameters first
                                for key in ["success", "status", "timestamp", "model_version", "app_version", "processing_time_ms"]:
                                    if key in example:
                                        ordered_example[key] = example[key]
                                # Input section
                                if "input" in example:
                                    ordered_example["input"] = example["input"]
                                # Summary section
                                if "summary" in example:
                                    ordered_example["summary"] = example["summary"]
                                # Details section at the end
                                if "details" in example:
                                    ordered_example["details"] = example["details"]
                                # Update the example with ordered version
                                json_content["example"] = ordered_example
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override the OpenAPI schema generator
app.openapi = custom_openapi

# CORS middleware - configure for Docker and local development
# Allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],  # Explicit methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)


@app.get("/favicon.ico")
async def favicon():
    """
    Handle favicon requests to prevent 404 errors in browser logs.
    This is a standard endpoint that browsers automatically request.
    """
    return Response(status_code=204)  # No Content


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - verifies service and model status.
    Use this endpoint for monitoring and load balancer health checks.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=detection_model is not None,
        ocr_loaded=ocr_model is not None
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded models."""
    sku_count = 0
    if sku_matching_service:
        sku_count = len(sku_matching_service.sku_master)
    
    model_path = get_model_path() if detection_model else None
    model_metadata = get_model_metadata() if detection_model else {}
    
    # Get detectable classes from model
    detectable_classes = []
    if detection_model:
        class_mapping = detection_model.get_class_mapping()
        detectable_classes = [
            {"class_id": class_id, "class_name": class_name}
            for class_id, class_name in sorted(class_mapping.items())
        ]
    
    return ModelInfoResponse(
        model_path=model_path,
        model_name=model_metadata.get("name"),
        model_version=model_metadata.get("version"),
        is_segmentation=detection_model.is_segmentation if detection_model else False,
        ocr_available=ocr_model is not None,
        sku_master_count=sku_count,
        app_version=APP_VERSION,
        detectable_classes=detectable_classes
    )


@app.post("/detect-and-count", response_model=DetectionResponse)
async def detect_and_count(
    file: UploadFile = File(...)
):
    """
    Main endpoint for inventory detection and counting.
    
    Takes an image as input and returns a complete detection response with:
    - Detected products with bounding boxes
    - SKU identification via OCR and matching
    - Confidence scores and validation status
    - Summary statistics
    
    Pipeline:
    1. Detection → bounding boxes and masks
    2. Crop extraction → individual product images
    3. OCR extraction → text from each crop
    4. SKU matching → fuzzy match to master list
    5. Decision Engine → fuses detection and OCR signals
    6. Response → hierarchical JSON with summary and details
    
    Args:
        file: Image file (jpg, jpeg, png)
        
    Returns:
        DetectionResponse with hierarchical structure:
        - Top level: success, status, timestamp, model_version, app_version, processing_time_ms
        - input: Input information (type, filename, frame_count)
        - summary: Summary statistics (total_detections, validated_detections, sku_counts)
        - details: List of detection details with final_prediction, detection_model, ocr_model
    """
    start_time = time.time()
    
    # Default values (not exposed as parameters)
    confidence_threshold = 0.5
    enable_ocr = OCR_ENABLED  # Use config value
    enable_sku_matching = True
    min_validation_confidence = 0.6
    
    # Validate model is loaded
    if detection_model is None or detection_service is None:
        raise HTTPException(status_code=500, detail="Detection model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image (async I/O)
        t0 = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Lazy import cv2 only when needed
        import cv2
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frame_h, frame_w = frame.shape[:2]
        logger.debug(f"Image loading: {(time.time() - t0)*1000:.1f}ms")
        
        # Step 1: Run detection (CPU-bound, but fast)
        t1 = time.time()
        logger.debug("Running detection...")
        raw_detections, crops = detection_service.detect(
            frame,
            confidence_threshold=confidence_threshold
        )
        detection_time = (time.time() - t1) * 1000
        logger.info(f"Detected {len(raw_detections)} products in {detection_time:.1f}ms")
        
        # Step 2: Run OCR on crops (if enabled) - optimized batch processing
        ocr_results = []
        t2 = time.time()
        if enable_ocr and ocr_service is not None and crops:
            logger.debug(f"Running OCR on {len(crops)} product crops...")
            ocr_results = ocr_service.extract_text_batch(crops)
            ocr_time = (time.time() - t2) * 1000
            logger.info(f"OCR completed in {ocr_time:.1f}ms ({ocr_time/len(crops):.1f}ms per crop)")
        else:
            ocr_results = [("", [])] * len(crops)
            logger.debug("OCR skipped")
        
        # Step 3: Build evidence-based detections
        t3 = time.time()
        detection_objects = []
        
        for i, raw_detection in enumerate(raw_detections):
            ocr_text, ocr_details = ocr_results[i] if i < len(ocr_results) else ("", [])
            
            # ====================================================================
            # DETECTION MODEL OUTPUT (Level 3 - detailed detection model data)
            # ====================================================================
            detection_conf = raw_detection['confidence']
            bbox = raw_detection['bbox']
            detection_class = raw_detection.get('detection_class', raw_detection.get('sku', 'unknown'))
            
            # Note: mask is used internally for crop extraction but not included in API response
            # to keep the response clean and avoid large payloads
            
            detection_model_output = DetectionModelOutput(
                matched_sku_id=detection_class,
                match_confidence=detection_conf
            )
            
            # ====================================================================
            # OCR MODEL OUTPUT (Level 3 - combines OCR text extraction and SKU matching)
            # ====================================================================
            ocr_confidence_avg = 0.0
            
            if ocr_details:
                confidences = [detail.get('confidence', 0.0) for detail in ocr_details if detail.get('text')]
                if confidences:
                    ocr_confidence_avg = sum(confidences) / len(confidences)
            
            # SKU matching (part of OCR model output)
            matched_sku_id = None
            matched_sku_name = None
            match_confidence = None
            match_method = "none"
            
            # SKU matching (can be slow with large catalogs)
            if enable_sku_matching and sku_matching_service is not None and ocr_text:
                match_start = time.time()
                matched_sku, match_conf, match_info = sku_matching_service.find_best_sku_match(ocr_text)
                match_time = (time.time() - match_start) * 1000
                if match_time > 100:  # Log if matching takes > 100ms
                    logger.debug(f"SKU matching took {match_time:.1f}ms for: '{ocr_text}'")
                if matched_sku:
                    matched_sku_id = matched_sku
                    match_confidence = match_conf
                    match_method = match_info.get('match_method', 'fuzzy')
                    
                    # Get SKU name from catalog
                    sku_info = sku_matching_service.get_sku_info(matched_sku)
                    if sku_info:
                        matched_sku_name = sku_info.get('item_name', matched_sku)
                    else:
                        matched_sku_name = matched_sku
            
            ocr_model_output = OCRModelOutput(
                text_raw=ocr_text,
                method=match_method,
                matched_sku_id=matched_sku_id,
                match_confidence=match_confidence
            )
            
            # ====================================================================
            # FINAL PREDICTION (Level 2 - final output with bbox)
            # ====================================================================
            # Use decision engine to fuse detection and OCR evidence
            if decision_engine is None:
                raise HTTPException(status_code=500, detail="Decision engine not initialized")
            
            # Make final decision using fusion logic
            final_prediction_base = decision_engine.make_final_decision(
                detection_class=detection_class,
                detection_confidence=detection_conf,
                ocr_text_raw=ocr_text,
                ocr_token_confidence_avg=ocr_confidence_avg,
                matched_sku_id=matched_sku_id,
                ocr_match_confidence=match_confidence
            )
            
            # Handle unknown objects: ensure sku_id is "-9999" with proper placeholders
            if detection_class == "-9999" or detection_class == "unknown":
                final_sku_id = "-9999"
                final_sku_name = "unknown"
                # Use the confidence from decision engine or fallback to detection confidence
                final_confidence = final_prediction_base.confidence if final_prediction_base.confidence > 0 else detection_conf
                final_validation_status = ValidationStatus.NOT_VALIDATED
                final_decision_reason = list(final_prediction_base.decision_reason) if final_prediction_base.decision_reason else ["unknown_object"]
            else:
                final_sku_id = final_prediction_base.sku_id
                final_sku_name = final_prediction_base.sku_name
                final_confidence = final_prediction_base.confidence
                final_validation_status = final_prediction_base.validation_status
                final_decision_reason = list(final_prediction_base.decision_reason) if final_prediction_base.decision_reason else []
            
            # Ensure we always have valid values (no None)
            if final_sku_id is None:
                final_sku_id = "-9999"
                final_sku_name = "unknown"
                final_validation_status = ValidationStatus.NOT_VALIDATED
                if not final_decision_reason:
                    final_decision_reason = ["unknown_object", "no_sku_match"]
            
            # Add bbox to final prediction (Level 2 requirement)
            final_prediction = FinalPrediction(
                sku_id=final_sku_id,
                sku_name=final_sku_name,
                confidence=final_confidence,
                bbox=bbox,  # Move bbox to final_prediction
                validation_status=final_validation_status,
                decision_reason=final_decision_reason
            )
            
            # ====================================================================
            # CREATE DETECTION DETAIL (Level 2)
            # ====================================================================
            detection_id = f"f0_d{i}"  # Frame 0, detection i
            
            detection_detail = DetectionDetail(
                detection_id=detection_id,
                final_prediction=final_prediction,
                detection_model=detection_model_output,
                ocr_model=ocr_model_output
            )
            
            detection_objects.append(detection_detail)
        
        processing_time = (time.time() - t3) * 1000
        logger.debug(f"Detection processing loop: {processing_time:.1f}ms")
        
        # ====================================================================
        # GENERATE SUMMARY (Derived from base output)
        # ====================================================================
        t4 = time.time()
        # Convert DetectionDetail to old Detection format for summary generation
        # TODO: Update generate_summary to work with new structure
        summary = generate_summary(detection_objects, min_confidence_threshold=min_validation_confidence)
        summary_time = (time.time() - t4) * 1000
        logger.debug(f"Summary generation: {summary_time:.1f}ms")
        
        # ====================================================================
        # BUILD RESPONSE (Level 1)
        # ====================================================================
        total_processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        model_metadata = get_model_metadata()
        
        # Log performance summary
        ocr_time_total = (time.time() - t2) * 1000 if enable_ocr and ocr_service is not None and crops else 0
        logger.info(
            f"Total processing: {total_processing_time:.1f}ms | "
            f"Detection: {detection_time:.1f}ms | "
            f"OCR: {ocr_time_total:.1f}ms | "
            f"Processing: {processing_time:.1f}ms | "
            f"Summary: {summary_time:.1f}ms"
        )
        
        response = DetectionResponse(
            success=True,
            status="success",
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get("version"),
            app_version=APP_VERSION,
            processing_time_ms=total_processing_time,
            input=InputInfo(
                type="image",
                filename=file.filename,
                frame_count=1
            ),
            summary=summary,
            details=detection_objects  # Level 2 - details section
        )
        
        logger.info(f"Processed image: {file.filename}, found {len(detection_objects)} products ({summary.validated_detections} validated) in {processing_time:.1f}ms")
        
        return response
        
    except MemoryError as e:
        logger.error(f"❌ Out of memory error processing image {file.filename}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=507,  # 507 Insufficient Storage
            detail="Out of memory while processing image. Please try a smaller image or increase container memory."
        )
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}", exc_info=True)
        # Log full error details for debugging
        import traceback
        logger.error(f"❌ Error processing image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Don't let exceptions crash the server - return error response instead
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    """
    Entry point when running directly: python -m app.main
    or: python app/main.py (from project root)
    """
    print("=" * 80)
    print("FastAPI Inventory Detector - Starting Service")
    print("=" * 80)
    print("📍 Service Access Points:")
    print("   API:         http://localhost:8000")
    print("   Docs:        http://localhost:8000/docs")
    print("   ReDoc:       http://localhost:8000/redoc")
    print("   Health:      http://localhost:8000/health")
    print("   Model Info:  http://localhost:8000/model/info")
    print()
    print("=" * 80)
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
