"""
Detection Service
Handles detection pipeline and result processing.
Uses CatalogService for SKU metadata lookup.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import supervision as sv
from collections import defaultdict
import logging

from app.models.detection_model import DetectionModel
from app.utils.image_utils import extract_crop_from_bbox, extract_crop_from_mask

# Type hint for CatalogService (avoid circular import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.services.catalog_service import CatalogService

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Service for running detection and extracting product crops.
    Uses CatalogService for efficient SKU metadata lookup.
    """
    
    def __init__(
        self, 
        detection_model: DetectionModel,
        catalog_service: Optional['CatalogService'] = None
    ):
        """
        Initialize detection service.
        
        Args:
            detection_model: Initialized detection model instance
            catalog_service: Optional CatalogService instance for SKU metadata lookup
        """
        self.detection_model = detection_model
        self.catalog_service = catalog_service
        self.tracker = sv.ByteTrack()
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        min_detection_area_ratio: float = 0.0001,
        strict_sku_validation: bool = True
    ) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Run detection on an image and extract product crops.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Detection confidence threshold
            min_detection_area_ratio: Minimum detection area as ratio of image
            strict_sku_validation: Filter out detections not matching known SKUs
            
        Returns:
            Tuple of (detections, crops)
            - detections: List of detection dicts with bbox, mask, class_id, confidence, sku
            - crops: List of cropped images for OCR processing
        """
        # Run detection inference
        results = self.detection_model.predict(
            image,
            confidence_threshold=confidence_threshold
        )
        
        # Convert to Supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter detections
        if len(detections) > 0:
            detections = self._filter_detections(
                detections,
                image,
                min_detection_area_ratio,
                strict_sku_validation
            )
        
        # Update tracker
        tracked_detections = self.tracker.update_with_detections(detections)
        
        # Extract detection information and crops
        detection_list = []
        crops = []
        
        frame_h, frame_w = image.shape[:2]
        
        for i in range(len(tracked_detections)):
            # Get detection properties
            bbox = tracked_detections.xyxy[i]
            class_id = int(tracked_detections.class_id[i])
            tracker_id = int(tracked_detections.tracker_id[i]) if hasattr(tracked_detections, 'tracker_id') else i
            
            # Get confidence
            if hasattr(tracked_detections, 'confidence') and tracked_detections.confidence is not None:
                confidence = float(tracked_detections.confidence[i])
            else:
                confidence = 0.5
            
            # Get SKU name from model
            sku = self.detection_model.get_class_name(class_id)
            
            # Get SKU metadata from catalog if available
            sku_metadata = None
            if self.catalog_service:
                try:
                    sku_metadata = self.catalog_service.get_sku_info(sku)
                except Exception as e:
                    logger.debug(f"Failed to get SKU metadata for {sku}: {str(e)}")
            
            # Get mask if available
            mask = None
            if self.detection_model.is_segmentation:
                if hasattr(tracked_detections, 'mask') and tracked_detections.mask is not None:
                    if len(tracked_detections.mask) > i:
                        mask = tracked_detections.mask[i]
            
            # Extract crop for OCR
            crop = None
            if mask is not None:
                crop = extract_crop_from_mask(image, mask, bbox)
            else:
                crop = extract_crop_from_bbox(image, bbox)
            
            # Store detection info
            detection_info = {
                'bbox': bbox.tolist(),
                'mask': mask.tolist() if mask is not None else None,
                'class_id': class_id,
                'tracker_id': tracker_id,
                'confidence': confidence,
                'sku': sku,
                'detection_class': sku,  # Generic detection class name
                'sku_metadata': sku_metadata  # Full SKU metadata from catalog
            }
            
            detection_list.append(detection_info)
            
            # Store crop if valid
            if crop is not None:
                crops.append(crop)
            else:
                crops.append(None)
        
        return detection_list, crops
    
    def _filter_detections(
        self,
        detections: sv.Detections,
        image: np.ndarray,
        min_detection_area_ratio: float,
        strict_sku_validation: bool
    ) -> sv.Detections:
        """
        Filter detections based on size and SKU validation.
        
        Args:
            detections: Supervision Detections object
            image: Input image
            min_detection_area_ratio: Minimum area ratio threshold
            strict_sku_validation: Whether to validate against known SKUs
            
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        frame_h, frame_w = image.shape[:2]
        image_area = frame_h * frame_w
        min_area = image_area * min_detection_area_ratio
        
        valid_mask = np.ones(len(detections), dtype=bool)
        
        for i in range(len(detections)):
            # Check SKU validation
            if strict_sku_validation and hasattr(detections, 'class_id'):
                class_id = int(detections.class_id[i])
                detected_sku = self.detection_model.get_class_name(class_id)
                if detected_sku not in self.detection_model.known_skus:
                    valid_mask[i] = False
                    continue
            
            # Check size validation
            if hasattr(detections, 'xyxy'):
                x1, y1, x2, y2 = detections.xyxy[i]
                detection_area = (x2 - x1) * (y2 - y1)
                if detection_area < min_area:
                    valid_mask[i] = False
                    continue
        
        # Apply filter
        filtered_count = np.sum(~valid_mask)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} detections")
            detections = detections[valid_mask]
        
        return detections
