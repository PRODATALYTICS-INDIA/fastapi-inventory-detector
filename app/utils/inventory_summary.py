"""
Inventory Summary Utility

Derives summary statistics from base detection output.
Follows the principle: "Base output is evidence. Summary is opinion."
"""
from typing import List, Dict
from collections import defaultdict
import logging

from app.schemas.response import (
    Summary,
    SKUCount,
    DetectionDetail,
    ValidationStatus
)

logger = logging.getLogger(__name__)


def generate_summary(detections: List[DetectionDetail], min_confidence_threshold: float = 0.0) -> Summary:
    """
    Generate summary statistics from list of detection details.
    
    The summary is derived programmatically from the base output (detections).
    This ensures consistency and prevents summary values that don't match the evidence.
    
    Args:
        detections: List of DetectionDetail objects (source of truth)
        min_confidence_threshold: Minimum confidence to include in validated count
        
    Returns:
        Summary object with SKU counts and statistics
    """
    # Group detections by SKU ID
    sku_groups: Dict[str, List[DetectionDetail]] = defaultdict(list)
    
    for detection in detections:
        sku_id = detection.final_prediction.sku_id
        # Use "unknown" as fallback for None SKU IDs
        if not sku_id:
            sku_id = "unknown"
        sku_groups[sku_id].append(detection)
    
    # Calculate statistics per SKU
    sku_counts: Dict[str, SKUCount] = {}
    
    for sku_id, sku_detections in sku_groups.items():
        if not sku_detections:
            continue
        
        # Get SKU name from first detection (all should have same name for same ID)
        sku_name = sku_detections[0].final_prediction.sku_name or sku_id
        
        # Generate slug for dictionary key (for display purposes)
        sku_slug = _generate_sku_slug(sku_id, sku_name)
        
        # Extract confidence values
        confidences = [d.final_prediction.confidence for d in sku_detections]
        
        # Calculate statistics
        count = len(sku_detections)
        avg_confidence = sum(confidences) / count if count > 0 else 0.0
        
        sku_counts[sku_slug] = SKUCount(
            sku_id=sku_id,
            count=count,
            confidence=avg_confidence
        )
    
    # Count validated detections (based on validation status and confidence)
    # Validated means: detection or OCR matched a SKU (excluding NOT_VALIDATED, OCR_NOT_VALIDATED, DETECTION_NOT_VALIDATED)
    validated_count = sum(
        1 for d in detections
        if d.final_prediction.validation_status in [
            ValidationStatus.FULLY_VALIDATED,
            ValidationStatus.OCR_VALIDATED,
            ValidationStatus.ONLY_OCR_VALIDATED,
            ValidationStatus.DETECTION_VALIDATED,
            ValidationStatus.ONLY_DETECTION_VALIDATED
        ] and d.final_prediction.confidence >= min_confidence_threshold
    )
    
    return Summary(
        sku_counts=sku_counts,
        total_detections=len(detections),
        validated_detections=validated_count
    )


def _generate_sku_slug(sku_id: str, sku_name: str) -> str:
    """
    Generate a slug from SKU ID and name for dictionary key.
    
    Args:
        sku_id: SKU ID
        sku_name: SKU display name
        
    Returns:
        Slug string suitable for dictionary key
    """
    # Use SKU ID as base, normalize it
    slug = sku_id.lower().replace(" ", "_").replace("-", "_")
    
    # Remove special characters
    import re
    slug = re.sub(r'[^a-z0-9_]', '', slug)
    
    # Ensure it's not empty
    if not slug:
        slug = sku_name.lower().replace(" ", "_").replace("-", "_")
        slug = re.sub(r'[^a-z0-9_]', '', slug)
        if not slug:
            slug = "unknown_sku"
    
    return slug

