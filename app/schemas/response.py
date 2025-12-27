"""
Pydantic response models for API responses.

Redesigned with evidence-based, frame-first architecture:
- Evidence vs Decision Separation
- Summary First, Details Below
- Frame-First Abstraction (supports video/multi-frame)
- Stable Identifiers (sku_id as primary)
- Forward Compatibility
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


# ============================================================================
# Validation Status Enum (FIXED)
# ============================================================================

class ValidationStatus(str, Enum):
    """Validation status enum for detections."""
    FULLY_VALIDATED = "fully_validated"  # Visual detection and OCR agree on the same SKU (highest confidence)
    OCR_VALIDATED = "ocr_validated"  # OCR matched a SKU in the catalog
    OCR_NOT_VALIDATED = "ocr_not_validated"  # OCR found text but didn't match a SKU
    ONLY_OCR_VALIDATED = "only_ocr_validated"  # Only OCR model matched but not DETECTION MODEL
    DETECTION_VALIDATED = "detection_validated"  # Visual detection matched a SKU in the catalog
    DETECTION_NOT_VALIDATED = "detection_not_validated"  # Visual detection not matched any SKU in the catalog
    ONLY_DETECTION_VALIDATED = "only_detection_validated"  # Only detection model found something, OCR model not
    NOT_VALIDATED = "not_validated"  # Visual detection and OCR both failed or with low confidence or providing conflicting signals


# ============================================================================
# Detection-Level Detailed Output (Source of Truth)
# ============================================================================

class DetectionModelOutput(BaseModel):
    """Output from detection model (Level 3 - detailed detection model data)."""
    model_config = ConfigDict(extra="forbid")  # Prevent extra fields (like mask) from being included
    
    matched_sku_id: str = Field(..., description="Matched SKU ID from detection model")
    match_confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")


class SKUMatchingInfo(BaseModel):
    """SKU matching information (Level 3 - detailed OCR data)."""
    method: str = Field(..., description="Matching method used (e.g., 'fuzzy', 'keyword_similarity')")
    matched_sku_id: Optional[str] = Field(None, description="Matched SKU ID (stable identifier)")
    matched_sku_name: Optional[str] = Field(None, description="Matched SKU display name")
    match_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="SKU match confidence score")


class OCRModelOutput(BaseModel):
    """Output from OCR model (Level 2 - combines OCR text extraction and SKU matching)."""
    text_raw: str = Field(..., description="Concatenated text extracted by OCR")
    method: str = Field(..., description="SKU matching method used (e.g., 'fuzzy', 'keyword_similarity', 'none')")
    matched_sku_id: Optional[str] = Field(None, description="Matched SKU ID from OCR (stable identifier)")
    match_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="SKU match confidence score")


class FinalPrediction(BaseModel):
    """Final SKU prediction derived from all evidence (Level 2 - final output for each SKU)."""
    sku_id: Optional[str] = Field(None, description="Final SKU ID (stable identifier)")
    sku_name: Optional[str] = Field(None, description="Final SKU display name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final aggregated confidence score")
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="Bounding box [x1, y1, x2, y2]")
    validation_status: ValidationStatus = Field(..., description="Validation status enum")
    decision_reason: List[str] = Field(default_factory=list, description="List of reasons for this prediction")


class DetectionDetail(BaseModel):
    """
    Complete detection detail (Level 2 - one per detected object).
    
    Structure:
    - final_prediction: Final output for this SKU (includes bbox, confidence, etc.)
    - detection_model: Detection model output (Level 3 details)
    - ocr_model: OCR model output (Level 3 details including text extraction and SKU matching)
    """
    detection_id: str = Field(..., description="Unique detection identifier (e.g., 'f0_d0')")
    
    final_prediction: FinalPrediction = Field(..., description="Final SKU prediction with bbox and confidence")
    detection_model: DetectionModelOutput = Field(..., description="Detection model output")
    ocr_model: OCRModelOutput = Field(..., description="OCR model output (includes text extraction and SKU matching)")


# ============================================================================
# Frame-Level Structure
# ============================================================================

class FrameDimensions(BaseModel):
    """Frame dimensions."""
    width: int = Field(..., gt=0, description="Frame width in pixels")
    height: int = Field(..., gt=0, description="Frame height in pixels")


class Frame(BaseModel):
    """Single frame with detections (legacy - kept for backward compatibility)."""
    frame_id: int = Field(..., ge=0, description="Frame identifier (0-indexed)")
    timestamp_ms: int = Field(..., ge=0, description="Frame timestamp in milliseconds")
    dimensions: FrameDimensions = Field(..., description="Frame dimensions")
    detections: List[DetectionDetail] = Field(default_factory=list, description="List of detections in this frame")


# ============================================================================
# Summary Structure (Derived from Base Output)
# ============================================================================

class SKUCount(BaseModel):
    """SKU count statistics for summary."""
    sku_id: str = Field(..., description="SKU ID (stable identifier)")
    count: int = Field(..., ge=0, description="Number of detections for this SKU")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence across all detections")


class Summary(BaseModel):
    """Summary derived from base output."""
    sku_counts: Dict[str, SKUCount] = Field(..., description="SKU counts keyed by sku_slug (for display)")
    total_detections: int = Field(..., ge=0, description="Total number of detections")
    validated_detections: int = Field(..., ge=0, description="Number of validated detections")


# ============================================================================
# Input Structure
# ============================================================================

class InputInfo(BaseModel):
    """Input information."""
    type: Literal["image", "video"] = Field(..., description="Input type")
    filename: Optional[str] = Field(None, description="Input filename")
    frame_count: int = Field(..., ge=1, description="Number of frames processed")


# ============================================================================
# Metadata Structure
# ============================================================================

class Metadata(BaseModel):
    """Processing metadata."""
    model_version: Optional[str] = Field(None, description="Detection model version")
    app_version: Optional[str] = Field(None, description="Application version")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")


# ============================================================================
# Top-Level Response
# ============================================================================

class DetectionResponse(BaseModel):
    """
    Main API response with hierarchical structure (Level 1).
    
    Architecture:
    - Level 1: Top-level response with status, timestamp, metadata fields, input, summary, and details
    - Level 2: Details section with final_prediction, detection_model, ocr_model for each object
    - Level 3: Detailed data within detection_model and ocr_model (matching fields)
    
    Structure (field order):
    1. Level 1 parameters: success, status, timestamp, model_version, app_version, processing_time_ms
    2. input: Input information (type, filename, frame_count)
    3. summary: Summary statistics
    4. details: List of detection details (Level 2)
    """
    # Level 1 parameters (first)
    success: bool = Field(..., description="Whether processing was successful")
    status: str = Field(default="success", description="Processing status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Processing timestamp")
    model_version: Optional[str] = Field(None, description="Detection model version")
    app_version: Optional[str] = Field(None, description="Application version")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    
    # Input section
    input: InputInfo = Field(..., description="Input information")
    
    # Summary section
    summary: Summary = Field(..., description="Summary derived from base output")
    
    # Details section (at the end)
    details: List[DetectionDetail] = Field(..., description="List of detection details (Level 2)")
    
    class Config:
        json_schema_extra = {
            "example": {
                # Level 1 parameters first
                "success": True,
                "status": "success",
                "timestamp": "2025-12-26T10:30:00.123456",
                "model_version": "25.11.0",
                "app_version": "2.0.0",
                "processing_time_ms": 1234.5,
                # Input section
                "input": {
                    "type": "image",
                    "filename": "test_image.jpg",
                    "frame_count": 1
                },
                # Summary section
                "summary": {
                    "sku_counts": {
                        "sku1_dettol_soap": {
                            "sku_id": "Sku1_Dettol Soap",
                            "count": 2,
                            "confidence": 0.85
                        },
                        "sku2_boost": {
                            "sku_id": "Sku2_Boost",
                            "count": 1,
                            "confidence": 0.78
                        }
                    },
                    "total_detections": 3,
                    "validated_detections": 2
                },
                # Details section at the end
                "details": [
                    {
                        "detection_id": "f0_d0",
                        "final_prediction": {
                            "sku_id": "Sku1_Dettol Soap",
                            "sku_name": "dettol-soap",
                            "confidence": 0.85,
                            "bbox": [100.0, 200.0, 300.0, 400.0],
                            "validation_status": "fully_validated",
                            "decision_reason": ["detection_ocr_agreement", "high_confidence"]
                        },
                        "detection_model": {
                            "matched_sku_id": "Sku1_Dettol Soap",
                            "match_confidence": 0.90
                        },
                        "ocr_model": {
                            "text_raw": "dettol soap",
                            "method": "fuzzy",
                            "matched_sku_id": "Sku1_Dettol Soap",
                            "match_confidence": 0.9
                        }
                    },
                    {
                        "detection_id": "f0_d1",
                        "final_prediction": {
                            "sku_id": "Sku2_Boost",
                            "sku_name": "boost",
                            "confidence": 0.78,
                            "bbox": [150.0, 250.0, 350.0, 450.0],
                            "validation_status": "ocr_validated",
                            "decision_reason": ["ocr_match_found", "detection_below_threshold"]
                        },
                        "detection_model": {
                            "matched_sku_id": "Sku2_Boost",
                            "match_confidence": 0.65
                        },
                        "ocr_model": {
                            "text_raw": "boost energy drink",
                            "method": "fuzzy",
                            "matched_sku_id": "Sku2_Boost",
                            "match_confidence": 0.85
                        }
                    }
                ]
            }
        }


# ============================================================================
# Legacy Models (for backward compatibility)
# ============================================================================

class ProductDetection(BaseModel):
    """Legacy single product detection result (deprecated - use Detection instead)."""
    detection_class: str = Field(..., description="Detected class/SKU code")
    sku: Optional[str] = Field(None, description="Matched SKU code from master list")
    sku_name: Optional[str] = Field(None, description="Matched SKU item name")
    confidence: float = Field(..., description="Overall confidence score (0.0-1.0)")
    detection_confidence: float = Field(..., description="Detection confidence score")
    sku_match_confidence: Optional[float] = Field(None, description="SKU matching confidence")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    mask: Optional[List[List[float]]] = Field(None, description="Segmentation mask coordinates")
    ocr_text: str = Field(..., description="OCR-extracted text from product crop")
    ocr_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed OCR results")
    tracker_id: Optional[int] = Field(None, description="Object tracker ID")
    class_id: int = Field(..., description="Model class ID")


class LegacyDetectionResponse(BaseModel):
    """Legacy response model (deprecated - use DetectionResponse instead)."""
    success: bool = Field(..., description="Whether processing was successful")
    filename: Optional[str] = Field(None, description="Original filename")
    products: List[ProductDetection] = Field(default_factory=list, description="List of detected products")
    total_detections: int = Field(0, description="Total number of detections")
    validated_count: int = Field(0, description="Number of detections validated by both detection and OCR")
    validation_summary: Dict[str, int] = Field(default_factory=dict, description="Summary of validation status")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Processing timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    model_version: Optional[str] = Field(None, description="Detection model version used")
    app_version: Optional[str] = Field(None, description="Application version")


# ============================================================================
# Health and Info Responses
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether detection model is loaded")
    ocr_loaded: bool = Field(..., description="Whether OCR model is loaded")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_path: Optional[str] = Field(None, description="Detection model file path")
    model_name: Optional[str] = Field(None, description="Detection model name")
    model_version: Optional[str] = Field(None, description="Detection model version")
    is_segmentation: bool = Field(False, description="Whether model supports segmentation")
    ocr_available: bool = Field(False, description="Whether OCR is available")
    sku_master_count: int = Field(0, description="Number of SKUs in master list")
    app_version: Optional[str] = Field(None, description="Application version")
    detectable_classes: List[Dict[str, Any]] = Field(default_factory=list, description="List of detectable classes with class_id and class_name")
