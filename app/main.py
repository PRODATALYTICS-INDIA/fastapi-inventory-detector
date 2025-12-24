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

import uvicorn
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
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
    SKU_MATCHING_MIN_CONFIDENCE
)

# Import new modular components
from app.models.detection_model import DetectionModel
from app.models.ocr_model import OCRModel
from app.services.catalog_service import CatalogService
from app.services.detection_service import DetectionService
from app.services.ocr_service import OCRService
from app.services.sku_matching_service import SKUMatchingService
from app.schemas.response import (
    DetectionResponse,
    ProductDetection,
    HealthResponse,
    ModelInfoResponse
)

# Import legacy tracker for backward compatibility
from py.InventoryTracker import InventoryTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Inventory Detection API",
    description="REST API for automated inventory detection with validation from detection and OCR services",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
detection_model: Optional[DetectionModel] = None
ocr_model: Optional[OCRModel] = None
catalog_service: Optional[CatalogService] = None
detection_service: Optional[DetectionService] = None
ocr_service: Optional[OCRService] = None
sku_matching_service: Optional[SKUMatchingService] = None

# Legacy tracker for backward compatibility
legacy_tracker: Optional[InventoryTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    global detection_model, ocr_model, catalog_service, detection_service, ocr_service, sku_matching_service, legacy_tracker
    
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
                ocr_service = OCRService(ocr_model, use_multiprocessing=True)
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
        
        # Initialize legacy tracker for backward compatibility
        try:
            legacy_tracker = InventoryTracker(model_path=model_path, label_mode="item_name")
            logger.info("✅ Legacy tracker initialized for backward compatibility")
        except Exception as e:
            logger.warning(f"⚠️ Legacy tracker initialization failed: {str(e)}")
        
        logger.info(f"Application version: {APP_VERSION}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if catalog_service:
        catalog_service.close()


async def startup_event():
    """Initialize models on application startup."""
    global detection_model, ocr_model, catalog_service, detection_service, ocr_service, sku_matching_service, legacy_tracker
    
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
                ocr_service = OCRService(ocr_model, use_multiprocessing=True)
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
        
        # Initialize legacy tracker for backward compatibility
        try:
            legacy_tracker = InventoryTracker(model_path=model_path, label_mode="item_name")
            logger.info("✅ Legacy tracker initialized for backward compatibility")
        except Exception as e:
            logger.warning(f"⚠️ Legacy tracker initialization failed: {str(e)}")
        
        logger.info(f"Application version: {APP_VERSION}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - Health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=detection_model is not None,
        ocr_loaded=ocr_model is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
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


@app.post("/detect", response_model=DetectionResponse)
async def detect(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    enable_ocr: bool = Form(True),
    enable_sku_matching: bool = Form(True),
    min_validation_confidence: float = Form(0.6)
):
    """
    Main unified endpoint for inventory detection with validation.
    
    Takes an image as input and returns final count of objects validated by both
    detection and OCR services.
    
    Pipeline:
    1. Detection → bounding boxes and masks
    2. Crop extraction → individual product images
    3. OCR extraction → text from each crop
    4. SKU matching → fuzzy match to master list
    5. Validation → combine detection and OCR confidence
    
    Args:
        file: Image file (jpg, jpeg, png)
        confidence_threshold: Detection confidence threshold (0.0-1.0)
        enable_ocr: Whether to run OCR on crops
        enable_sku_matching: Whether to match SKUs using OCR text
        min_validation_confidence: Minimum combined confidence for validation
        
    Returns:
        DetectionResponse with:
        - products: List of all detected products
        - total_detections: Total number of detections
        - validated_count: Count of validated detections (detection + OCR)
        - validation_summary: Summary by validation status
    """
    start_time = time.time()
    
    # Validate model is loaded
    if detection_model is None or detection_service is None:
        raise HTTPException(status_code=500, detail="Detection model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image (async I/O)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Run detection (CPU-bound, but fast)
        logger.debug("Running detection...")
        detections, crops = detection_service.detect(
            frame,
            confidence_threshold=confidence_threshold
        )
        
        logger.info(f"Detected {len(detections)} products")
        
        # Step 2: Run OCR on crops (if enabled) - optimized batch processing
        ocr_results = []
        if enable_ocr and ocr_service is not None and crops:
            logger.debug("Running OCR on product crops...")
            ocr_results = ocr_service.extract_text_batch(crops)
        else:
            ocr_results = [("", [])] * len(crops)
        
        # Step 3: Process detections with validation
        products = []
        validated_count = 0
        validation_summary = {
            "detection_only": 0,
            "ocr_validated": 0,
            "sku_matched": 0,
            "fully_validated": 0
        }
        
        for i, detection in enumerate(detections):
            ocr_text, ocr_details = ocr_results[i] if i < len(ocr_results) else ("", [])
            
            # Get SKU match
            sku_code = None
            sku_name = None
            sku_match_confidence = None
            is_validated = False
            validation_status = "detection_only"
            
            # Calculate combined confidence for validation
            detection_conf = detection['confidence']
            ocr_conf = 0.0
            if ocr_text and ocr_details:
                # Use average OCR confidence if available
                ocr_conf = sum(d.get('confidence', 0.0) for d in ocr_details) / len(ocr_details) if ocr_details else 0.0
            
            combined_confidence = (detection_conf * 0.6 + ocr_conf * 0.4) if ocr_text else detection_conf
            
            if enable_sku_matching and sku_matching_service is not None and ocr_text:
                matched_sku, match_conf, match_info = sku_matching_service.find_best_sku_match(ocr_text)
                if matched_sku:
                    sku_code = matched_sku
                    sku_info = sku_matching_service.get_sku_info(matched_sku)
                    if sku_info:
                        sku_name = sku_info.get('item_name', matched_sku)
                    sku_match_confidence = match_conf
                    validation_status = "sku_matched"
                    if combined_confidence >= min_validation_confidence:
                        validation_status = "fully_validated"
                        is_validated = True
                elif ocr_text:
                    validation_status = "ocr_validated"
                    if combined_confidence >= min_validation_confidence:
                        is_validated = True
            elif ocr_text and combined_confidence >= min_validation_confidence:
                validation_status = "ocr_validated"
                is_validated = True
            
            # If no SKU match, use detection class as fallback
            if not sku_code:
                sku_code = detection['sku']
                sku_name = detection['sku']
            
            # Update validation summary
            validation_summary[validation_status] = validation_summary.get(validation_status, 0) + 1
            if is_validated:
                validated_count += 1
            
            # Convert mask to list format if present (optimize for large masks)
            mask_list = None
            if detection.get('mask') is not None:
                mask = detection['mask']
                if isinstance(mask, np.ndarray):
                    try:
                        # Only convert if mask is reasonable size
                        if mask.size < 1000000:  # ~1M pixels
                            mask_list = mask.tolist()
                    except Exception:
                        mask_list = None
                elif isinstance(mask, list):
                    mask_list = mask
            
            product = ProductDetection(
                detection_class=detection.get('detection_class', detection.get('sku', 'unknown')),
                sku=sku_code,
                sku_name=sku_name,
                confidence=combined_confidence,  # Combined confidence
                detection_confidence=detection['confidence'],
                sku_match_confidence=sku_match_confidence,
                bbox=detection['bbox'],
                mask=mask_list,
                ocr_text=ocr_text,
                ocr_details=ocr_details,
                tracker_id=detection.get('tracker_id'),
                class_id=detection['class_id']
            )
            products.append(product)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        model_metadata = get_model_metadata()
        response = DetectionResponse(
            success=True,
            filename=file.filename,
            products=products,
            total_detections=len(products),
            validated_count=validated_count,
            validation_summary=validation_summary,
            processing_time_ms=processing_time,
            model_version=model_metadata.get("version"),
            app_version=APP_VERSION
        )
        
        logger.info(f"Processed image: {file.filename}, found {len(products)} products ({validated_count} validated) in {processing_time:.1f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Alias for backward compatibility
@app.post("/detect-and-identify", response_model=DetectionResponse)
async def detect_and_identify(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    enable_ocr: bool = Form(True),
    enable_sku_matching: bool = Form(True)
):
    """Legacy endpoint name - redirects to /detect"""
    return await detect(file, confidence_threshold, enable_ocr, enable_sku_matching, 0.6)


# Legacy endpoints for backward compatibility
@app.post("/predict/image")
async def predict_image_legacy(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    label_mode: str = Form("item_name")
):
    """
    Legacy endpoint for image prediction (backward compatibility).
    Uses the old InventoryTracker class.
    """
    if legacy_tracker is None:
        raise HTTPException(status_code=500, detail="Legacy tracker not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if legacy_tracker.label_mode != label_mode:
            legacy_tracker.label_mode = label_mode
        
        legacy_tracker.reset_output_stats()
        annotated_frame, live_summary = legacy_tracker.track_picture_stream(frame, confidence_threshold)
        stats_df = legacy_tracker.get_output_stats()
        
        import base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "filename": file.filename,
            "confidence_threshold": confidence_threshold,
            "label_mode": label_mode,
            "frame_count": legacy_tracker.frame_count,
            "detections": live_summary,
            "statistics": stats_df.to_dict('records') if not stats_df.empty else [],
            "annotated_image": f"data:image/jpeg;base64,{annotated_image_b64}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
