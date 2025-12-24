"""
Pydantic response models for API responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ProductDetection(BaseModel):
    """Single product detection result."""
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


class DetectionResponse(BaseModel):
    """Response model for /detect-and-identify endpoint."""
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
