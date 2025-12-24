"""
Pydantic request models for API requests.
"""
from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Request model for detection endpoint."""
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    enable_ocr: bool = Field(True, description="Enable OCR text extraction")
    enable_sku_matching: bool = Field(True, description="Enable SKU matching")
    min_validation_confidence: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence for validation")


